"""可预览、可应用、可撤销的 episode 整理与手工生命周期操作。"""
from __future__ import annotations

import json
import asyncio
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from astrbot.api import logger

MAINTENANCE_LOGIC_VERSION = 2


MAINTENANCE_PROMPT = """你是 Astra Memoir 的记忆整理器。你不是写日报，而是在同一小批独立 episode 中判断长期价值。
只能输出严格 JSON，只允许 keep / archive / merge，绝不 delete。
判断重点：几天到几周后重新提起，对 Astra 是否仍有价值？
普通画图请求、表情包、一次性玩笑、重复闲聊、无后续小事优先 archive；明确决定、人物状态变化、项目进展、新设定、持续话题和有后续影响的互动保留。
同一持续事件的碎片可以 merge；不同主题绝不能因为同一天而合并。merge 必须写成一条自然、完整、忠实的事件记忆，不添加原文没有的事实。
人称规则是硬规则：只有输入中 astra_participated=true，且事实确实是 role=assistant 的 Astra 所做或所说，才能用“我”指代 Astra；群友或其他 AI 的言行必须使用 participants 中的正式名字。若 astra_participated=false，标题和正文都禁止使用“我/我们”，必须客观第三人称记录，也不要虚构“我看到/我得知”。
每个输入 id 必须且只能出现一次。keep/archive 的 ids 必须只有一个；merge 至少两个。
edited_by_user=true 的正文是用户手工事实，绝不能 merge 或改写；过保护期后只能 keep 或 archive。
importance 为 1~5 的重新评估结果。
输出：{"decisions":[{"action":"keep","ids":[1],"importance":3,"reason":"..."},{"action":"archive","ids":[2],"importance":1,"reason":"一次性闲聊"},{"action":"merge","ids":[3,4],"title":"...","content":"...","importance":4,"reason":"同一持续事件"}]}"""


class MaintenanceError(RuntimeError):
    pass


class MaintenanceManager:
    def __init__(self, db, vec, writer, extractor, *, batch_size: int = 12):
        self.db, self.vec, self.writer, self.extractor = db, vec, writer, extractor
        self.batch_size = max(4, min(int(batch_size), 20))
        self._run_lock = asyncio.Lock()

    def backup(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        folder = Path(self.db.db_path).parent / "backups"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"memoir-before-maintenance-{stamp}.db"
        import sqlite3
        dst = sqlite3.connect(target)
        try:
            self.db.conn.backup(dst)
        finally:
            dst.close()
        return str(target)

    @staticmethod
    def _topic_tokens(row) -> set[str]:
        text = f"{row['title']} {row['content']} {row.get('keywords', '')}".lower()
        compact = re.sub(r"\s+", "", text)
        return {compact[i:i+2] for i in range(max(0, len(compact)-1))}

    def _small_batches(self, rows: list[dict]) -> list[list[dict]]:
        """先按本地日期和来源隔开，再按主题相似度贪心排序并限制小批。"""
        buckets: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            day = datetime.fromtimestamp(int(row["event_start_at"])).strftime("%Y-%m-%d")
            buckets[(day, row["chat_type"], row["session_id"], row.get("group_id"))].append(row)
        result = []
        for items in buckets.values():
            remaining = items[:]
            while remaining:
                batch = [remaining.pop(0)]
                union = self._topic_tokens(batch[0])
                while remaining and len(batch) < self.batch_size:
                    best_i, best = 0, -1.0
                    for i, candidate in enumerate(remaining):
                        tokens = self._topic_tokens(candidate)
                        score = len(union & tokens) / max(1, len(union | tokens))
                        if score > best:
                            best_i, best = i, score
                    chosen = remaining.pop(best_i)
                    batch.append(chosen)
                    union |= self._topic_tokens(chosen)
                result.append(batch)
        return result

    async def _ask(self, rows: list[dict]) -> list[dict]:
        provider = self.extractor._get_provider()
        if provider is None:
            raise MaintenanceError("整理模型不可用")
        payload = [{"id": r["id"], "title": r["title"], "content": r["content"],
                    "importance": r["importance"], "source": r["chat_type"],
                    "edited_by_user": bool(r.get("edited_by_user")),
                    "participants": r.get("participants", []),
                    "astra_participated": any(
                        p.get("role") == "assistant" for p in r.get("participants", [])
                    ),
                    "time": datetime.fromtimestamp(r["event_start_at"]).isoformat()}
                   for r in rows]
        expected = {r["id"] for r in rows}
        validation_error = ""
        for attempt in range(3):
            prompt = "<episodes>\n" + json.dumps(payload, ensure_ascii=False) + "\n</episodes>"
            if attempt:
                prompt += ("\n上次输出被硬校验拒绝，原因：" + validation_error +
                           "。请修正后重新输出完整 JSON；覆盖每个 id 恰好一次。")
            try:
                resp = await asyncio.wait_for(
                    provider.text_chat(prompt=prompt, system_prompt=MAINTENANCE_PROMPT),
                    timeout=120,
                )
            except asyncio.TimeoutError:
                if attempt < 2:
                    validation_error = "模型调用超过120秒"
                    continue
                raise MaintenanceError("整理模型单批连续三次超过 120 秒")
            text = getattr(resp, "completion_text", "") or ""
            match = re.search(r"\{[\s\S]*\}", text)
            try:
                decisions = json.loads(match.group(0))["decisions"] if match else []
                seen, valid = set(), []
                for d in decisions:
                    action, ids = d.get("action"), d.get("ids")
                    importance = d.get("importance")
                    if action not in ("keep", "archive", "merge") or not isinstance(ids, list):
                        raise ValueError("action 必须是 keep/archive/merge，ids 必须为数组")
                    try:
                        ids = [int(i) for i in ids]
                    except (TypeError, ValueError):
                        raise ValueError("ids 必须全部是整数")
                    if not ids or set(ids) - expected or seen & set(ids):
                        raise ValueError("ids 为空、越界或重复出现")
                    if action in ("keep", "archive") and len(ids) != 1:
                        raise ValueError("keep/archive 每项只能包含一个 id")
                    if action == "merge" and (len(ids) < 2 or not str(d.get("title", "")).strip()
                                              or not str(d.get("content", "")).strip()):
                        raise ValueError("merge 至少需要两个 id，且必须有完整 title/content")
                    if action == "merge" and any(r["id"] in ids and r.get("edited_by_user") for r in rows):
                        raise ValueError("merge 包含 edited_by_user=true 的用户手工记忆")
                    if action == "merge":
                        astra_participated = any(
                            r["id"] in ids and any(
                                p.get("role") == "assistant" for p in r.get("participants", [])
                            ) for r in rows
                        )
                        merged_text = f"{d.get('title', '')}\n{d.get('content', '')}"
                        if not astra_participated and "我" in merged_text:
                            message = (
                                f"merge ids={ids} 没有 Astra(role=assistant) 参与，禁止使用‘我/我们’；"
                                "请用 participants 中的正式名字作第三人称记录")
                            if attempt < 2:
                                raise ValueError(message)
                            # 第三次仍犯同一人称错误时，仅拆回该 merge；同批其它合法判断保留。
                            by_id = {r["id"]: r for r in rows}
                            seen.update(ids)
                            valid.extend({
                                "action": "keep", "ids": [eid],
                                "importance": int(by_id[eid]["importance"]),
                                "reason": f"合并摘要人称不安全，安全保留：{message}",
                            } for eid in ids)
                            continue
                    if type(importance) is not int or not 1 <= importance <= 5:
                        raise ValueError("importance 必须是 1~5 的整数")
                    seen.update(ids)
                    valid.append({**d, "ids": ids})
                if seen != expected:
                    raise ValueError(f"未恰好覆盖全部输入 id，缺少 {sorted(expected-seen)}")
                return valid
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                validation_error = str(exc) or "JSON/schema 不合法"
                continue
        logger.warning(
            "[Memoir] maintenance batch ids=%s failed validation 3 times (%s); safe-keep fallback",
            sorted(expected), validation_error,
        )
        return [{"action": "keep", "ids": [r["id"]],
                 "importance": int(r["importance"]),
                 "reason": f"模型连续输出无效，安全保留：{validation_error}"}
                for r in rows]

    async def preview(self, start_ts: int, end_ts: int, *, run_type: str = "history",
                      target_date: str | None = None) -> dict:
        async with self._run_lock:
            return await self._preview_unlocked(
                start_ts, end_ts, run_type=run_type, target_date=target_date)

    async def _preview_unlocked(self, start_ts: int, end_ts: int, *, run_type: str,
                                target_date: str | None) -> dict:
        if run_type not in ("history", "nightly") or end_ts <= start_ts:
            raise ValueError("无效整理范围")
        backup_path = await self.db.run(self.backup)
        now = int(time.time())
        def _create():
            cur = self.db.execute(
                "INSERT INTO maintenance_runs(run_type,target_start,target_end,status,backup_path,created_at,logic_version) "
                "VALUES(?,?,?,?,?,?,?)", (run_type, start_ts, end_ts, "generating", backup_path,
                                          now, MAINTENANCE_LOGIC_VERSION))
            if target_date:
                self.db.execute(
                    "INSERT OR REPLACE INTO maintenance_days(target_date,run_id,status,updated_at) VALUES(?,?,?,?)",
                    (target_date, cur.lastrowid, "running", now))
            return cur.lastrowid
        run_id = await self.db.run(_create)
        try:
            def _load():
                rows = self.db.fetchall(
                    "SELECT e.*, COALESCE(group_concat(k.keyword,' '),'') keywords FROM episodes e "
                    "LEFT JOIN episode_keywords k ON k.episode_id=e.id "
                    "WHERE e.status='active' AND e.event_start_at>=? AND e.event_start_at<? "
                    "AND (e.protected_until IS NULL OR e.protected_until<=?) "
                    "GROUP BY e.id ORDER BY e.event_start_at,e.id", (start_ts, end_ts, now))
                result = [dict(r) for r in rows]
                if not result:
                    return result
                ids = [r["id"] for r in result]
                parts = self.db.fetchall(
                    f"SELECT episode_id,speaker_id,speaker_name,role FROM episode_participants "
                    f"WHERE episode_id IN ({','.join('?' * len(ids))})", ids)
                by_id: dict[int, list[dict]] = defaultdict(list)
                for p in parts:
                    name = self.db.identities.display(p["speaker_id"], p["speaker_name"])[0]
                    by_id[p["episode_id"]].append({
                        "speaker_id": p["speaker_id"], "name": name, "role": p["role"]})
                for row in result:
                    row["participants"] = by_id.get(row["id"], [])
                return result
            rows = await self.db.run(_load)
            batches = self._small_batches(rows)
            await self.db.run(lambda: self.db.execute(
                "UPDATE maintenance_runs SET source_count=?,batch_count=? WHERE id=?",
                (len(rows), len(batches), run_id)))
            decisions = []
            for completed, batch in enumerate(batches, 1):
                decisions.extend(await self._ask(batch))
                await self.db.run(lambda n=completed: self.db.execute(
                    "UPDATE maintenance_runs SET completed_batches=? WHERE id=?", (n, run_id)))
            row_map = {r["id"]: r for r in rows}
            def _save():
                counts = {"keep": 0, "archive": 0, "merge_sources": 0, "merge_results": 0}
                with self.db.transaction():
                    for d in decisions:
                        group = uuid.uuid4().hex if d["action"] == "merge" else None
                        if d["action"] == "merge":
                            counts["merge_results"] += 1
                            counts["merge_sources"] += len(d["ids"])
                        else:
                            counts[d["action"]] += 1
                        for eid in d["ids"]:
                            self.db.execute(
                                "INSERT INTO maintenance_actions(run_id,action,source_episode_id,merge_group,"
                                "proposed_title,proposed_content,proposed_importance,reason,before_status,"
                                "before_archive_reason,before_merged_into,before_importance) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                (run_id,d["action"],eid,group,d.get("title"),d.get("content"),
                                 d["importance"],str(d.get("reason", ""))[:300],row_map[eid]["status"],
                                 row_map[eid]["archive_reason"],row_map[eid]["merged_into"],row_map[eid]["importance"]))
                    self.db.execute(
                        "UPDATE maintenance_runs SET status='preview',source_count=?,keep_count=?,"
                        "archive_count=?,merge_source_count=?,merge_result_count=? WHERE id=?",
                        (len(rows),counts["keep"],counts["archive"],counts["merge_sources"],
                         counts["merge_results"],run_id))
                    if target_date:
                        self.db.execute(
                            "INSERT OR REPLACE INTO maintenance_days(target_date,run_id,status,updated_at) "
                            "VALUES(?,?,?,?)", (target_date,run_id,"preview",now))
                return self.run_detail(run_id)
            return await self.db.run(_save)
        except Exception as exc:
            def _failed():
                self.db.execute("UPDATE maintenance_runs SET status='failed',error=? WHERE id=?",
                                (str(exc),run_id))
                if target_date:
                    self.db.execute(
                        "INSERT OR REPLACE INTO maintenance_days(target_date,run_id,status,updated_at) VALUES(?,?,?,?)",
                        (target_date,run_id,"failed",int(time.time())))
            await self.db.run(_failed)
            raise

    def run_detail(self, run_id: int) -> dict:
        run = self.db.fetchone("SELECT * FROM maintenance_runs WHERE id=?", (run_id,))
        if not run:
            raise ValueError("整理记录不存在")
        actions = self.db.fetchall(
            "SELECT a.*,e.title source_title,e.content source_content FROM maintenance_actions a "
            "LEFT JOIN episodes e ON e.id=a.source_episode_id WHERE a.run_id=? ORDER BY a.id", (run_id,))
        return {**dict(run), "actions": [dict(a) for a in actions]}

    def list_runs(self) -> list[dict]:
        return [dict(r) for r in self.db.fetchall(
            "SELECT * FROM maintenance_runs ORDER BY created_at DESC,id DESC LIMIT 100")]

    def delete_run_record(self, run_id: int) -> dict:
        """Discard an unused Preview or remove a failed run from panel history.

        The database backup is deliberately preserved. Applied/undone runs remain as
        the audit trail needed to explain or verify episode state changes.
        """
        run = self.db.fetchone("SELECT * FROM maintenance_runs WHERE id=?", (run_id,))
        if not run:
            raise ValueError("整理记录不存在")
        if run["status"] not in ("preview", "failed"):
            raise ValueError("只能删除失败记录或放弃尚未应用的 Preview")
        backup_path = run["backup_path"]
        with self.db.transaction():
            self.db.execute("DELETE FROM maintenance_days WHERE run_id=?", (run_id,))
            self.db.execute("DELETE FROM maintenance_actions WHERE run_id=?", (run_id,))
            self.db.execute("DELETE FROM maintenance_runs WHERE id=?", (run_id,))
        return {
            "deleted": True,
            "run_id": run_id,
            "backup_path": backup_path,
            "backup_preserved": True,
        }

    async def apply(self, run_id: int) -> dict:
        async with self._run_lock:
            return await self._apply_unlocked(run_id)

    async def _apply_unlocked(self, run_id: int) -> dict:
        detail = await self.db.run(self.run_detail, run_id)
        if detail["status"] != "preview":
            raise ValueError("只有 Preview 方案可以应用")
        if int(detail.get("logic_version") or 1) != MAINTENANCE_LOGIC_VERSION:
            raise ValueError("该 Preview 使用旧版整理规则，已禁止应用；请重新生成")
        grouped: dict[str, list[dict]] = defaultdict(list)
        for a in detail["actions"]:
            if a["action"] == "merge": grouped[a["merge_group"]].append(a)
        embeddings = {}
        for key, actions in grouped.items():
            text = f"{actions[0]['proposed_title']}\n{actions[0]['proposed_content']}"
            emb = await self.writer._embed(text)
            if emb is None: raise MaintenanceError("合并记忆 embedding 失败，未应用任何修改")
            embeddings[key] = emb
        now = int(time.time())
        def _validate_sources():
            for a in detail["actions"]:
                ep = self.db.fetchone("SELECT * FROM episodes WHERE id=?", (a["source_episode_id"],))
                if (not ep or ep["status"] != a["before_status"]
                        or ep["importance"] != a["before_importance"]
                        or (ep["protected_until"] and ep["protected_until"] > now)
                        or self.db.fetchone("SELECT 1 FROM episode_versions WHERE episode_id=? AND saved_at>?",
                                           (a["source_episode_id"], detail["created_at"]))):
                    raise ValueError(f"episode #{a['source_episode_id']} 在 Preview 后已变化，请重新生成方案")
        await self.db.run(_validate_sources)
        def _apply():
            with self.db.transaction():
                for a in detail["actions"]:
                    ep = self.db.fetchone("SELECT * FROM episodes WHERE id=?", (a["source_episode_id"],))
                    if a["action"] == "keep":
                        if not ep["edited_by_user"]:
                            self.db.execute("UPDATE episodes SET importance=? WHERE id=?",
                                            (a["proposed_importance"],ep["id"]))
                    elif a["action"] == "archive":
                        if not ep["edited_by_user"]:
                            self.db.execute("UPDATE episodes SET importance=? WHERE id=?",
                                            (a["proposed_importance"],ep["id"]))
                        self._set_state(ep["id"], "archived", a["reason"] or "低长期价值")
                for key, actions in grouped.items():
                    sources = [self.db.fetchone("SELECT * FROM episodes WHERE id=?",
                                                (a["source_episode_id"],)) for a in actions]
                    base, first = sources[0], actions[0]
                    raw_ids = sorted({rid for e in sources for rid in json.loads(e["source_raw_ids"] or "[]")})
                    eid = self.db.insert_episode(platform=base["platform"],chat_type=base["chat_type"],
                        session_id=base["session_id"],group_id=base["group_id"],title=first["proposed_title"],
                        content=first["proposed_content"],source_raw_ids=raw_ids,
                        event_start_at=min(e["event_start_at"] for e in sources),
                        event_end_at=max(e["event_end_at"] for e in sources),extracted_at=now,
                        importance=first["proposed_importance"])
                    self.db.execute("UPDATE episodes SET created_by_run_id=? WHERE id=?", (run_id,eid))
                    participant_rows = self.db.fetchall(
                        f"SELECT DISTINCT speaker_id,speaker_name,role FROM episode_participants WHERE episode_id IN ({','.join('?'*len(sources))})",
                        [e["id"] for e in sources])
                    self.db.insert_participants(eid,[dict(p) for p in participant_rows])
                    keywords = [r["keyword"] for r in self.db.fetchall(
                        f"SELECT DISTINCT keyword FROM episode_keywords WHERE episode_id IN ({','.join('?'*len(sources))})",
                        [e["id"] for e in sources])]
                    self.db.insert_keywords(eid,keywords); self.db.insert_fts(eid,first["proposed_title"],first["proposed_content"],keywords)
                    self.vec.upsert(eid,embeddings[key],chat_type=base["chat_type"],session_id=base["session_id"],group_id=base["group_id"])
                    for e,a in zip(sources,actions):
                        self._set_state(e["id"],"archived",f"已合并到 #{eid}",merged_into=eid)
                        self.db.execute("INSERT INTO episode_merge_sources VALUES(?,?)",(eid,e["id"]))
                        self.db.execute("UPDATE maintenance_actions SET result_episode_id=? WHERE id=?",(eid,a["id"]))
                self.db.execute("UPDATE maintenance_runs SET status='applied',applied_at=? WHERE id=?",(now,run_id))
                self.db.execute("UPDATE maintenance_days SET status='applied',updated_at=? WHERE run_id=?",(now,run_id))
            return self.run_detail(run_id)
        return await self.db.run(_apply)

    def _set_state(self, eid: int, status: str, reason: str | None = None,
                   *, merged_into=None, user_restore=False):
        archived = 0 if status == "active" else 1
        protected = int(time.time()) + 30*86400 if user_restore else None
        self.db.execute("UPDATE episodes SET status=?,is_archived=?,archive_reason=?,merged_into=?,"
                        "restored_by_user=CASE WHEN ? THEN 1 ELSE restored_by_user END,"
                        "protected_until=COALESCE(?,protected_until) WHERE id=?",
                        (status,archived,reason,merged_into,int(user_restore),protected,eid))
        self.db.execute("UPDATE episode_vec SET is_archived=? WHERE episode_id=?",(archived,eid))

    async def undo(self, run_id: int) -> dict:
        async with self._run_lock:
            return await self._undo_unlocked(run_id)

    async def _undo_unlocked(self, run_id: int) -> dict:
        detail = await self.db.run(self.run_detail, run_id)
        if detail["status"] != "applied": raise ValueError("只有已应用的整理可以撤销")
        now = int(time.time())
        def _validate_undo():
            for a in detail["actions"]:
                ep = self.db.fetchone("SELECT * FROM episodes WHERE id=?", (a["source_episode_id"],))
                expected = "active" if a["action"] == "keep" else "archived"
                if (not ep or ep["status"] != expected
                        or (ep["protected_until"] and ep["protected_until"] > now)
                        or self.db.fetchone("SELECT 1 FROM episode_versions WHERE episode_id=? AND saved_at>?",
                                           (a["source_episode_id"], detail["applied_at"] or 0))):
                    raise ValueError(f"episode #{a['source_episode_id']} 在整理后被用户修改，拒绝覆盖；请先人工检查")
            for eid in {a["result_episode_id"] for a in detail["actions"] if a["result_episode_id"]}:
                ep = self.db.fetchone("SELECT * FROM episodes WHERE id=?", (eid,))
                if (not ep or ep["status"] != "active" or ep["edited_by_user"]
                        or (ep["protected_until"] and ep["protected_until"] > now)):
                    raise ValueError(f"合并 episode #{eid} 已被用户修改，拒绝覆盖；请先人工检查")
        await self.db.run(_validate_undo)
        def _undo():
            with self.db.transaction():
                result_ids = {a["result_episode_id"] for a in detail["actions"] if a["result_episode_id"]}
                for eid in result_ids:
                    self._set_state(eid,"trashed","撤销整理产生的合并记忆")
                for a in detail["actions"]:
                    archived = 0 if a["before_status"] == "active" else 1
                    self.db.execute("UPDATE episodes SET status=?,is_archived=?,archive_reason=?,merged_into=?,importance=? WHERE id=?",
                        (a["before_status"],archived,a["before_archive_reason"],a["before_merged_into"],a["before_importance"],a["source_episode_id"]))
                    self.db.execute("UPDATE episode_vec SET is_archived=? WHERE episode_id=?",(archived,a["source_episode_id"]))
                self.db.execute("UPDATE maintenance_runs SET status='undone',undone_at=? WHERE id=?",(now,run_id))
            return self.run_detail(run_id)
        return await self.db.run(_undo)

    async def edit_episode(self, eid: int, title: str, content: str, importance: int) -> dict:
        if not title.strip() or not content.strip() or not 1 <= importance <= 5:
            raise ValueError("标题、正文和 importance(1~5) 必须有效")
        emb = await self.writer._embed(f"{title.strip()}\n{content.strip()}")
        if emb is None: raise MaintenanceError("编辑后的 embedding 失败，未保存")
        now = int(time.time())
        def _edit():
            ep=self.db.fetchone("SELECT * FROM episodes WHERE id=?",(eid,))
            if not ep: raise ValueError("episode 不存在")
            with self.db.transaction():
                self.db.execute("INSERT INTO episode_versions(episode_id,title,content,importance,saved_at) VALUES(?,?,?,?,?)",
                                (eid,ep["title"],ep["content"],ep["importance"],now))
                self.db.execute("UPDATE episodes SET title=?,content=?,importance=?,edited_by_user=1,protected_until=? WHERE id=?",
                                (title.strip(),content.strip(),importance,now+30*86400,eid))
                keywords=[r["keyword"] for r in self.db.fetchall("SELECT keyword FROM episode_keywords WHERE episode_id=?",(eid,))]
                self.db.execute("DELETE FROM episodes_fts WHERE rowid=?",(eid,)); self.db.insert_fts(eid,title.strip(),content.strip(),keywords)
                self.vec.upsert(eid,emb,chat_type=ep["chat_type"],session_id=ep["session_id"],group_id=ep["group_id"],is_archived=ep["is_archived"])
            return dict(self.db.fetchone("SELECT * FROM episodes WHERE id=?",(eid,)))
        return await self.db.run(_edit)

    async def undo_latest_edit(self, eid: int) -> dict:
        def _load():
            ep = self.db.fetchone("SELECT * FROM episodes WHERE id=?", (eid,))
            version = self.db.fetchone(
                "SELECT * FROM episode_versions WHERE episode_id=? ORDER BY id DESC LIMIT 1", (eid,))
            return (dict(ep) if ep else None, dict(version) if version else None)
        ep, version = await self.db.run(_load)
        if not ep or not version: raise ValueError("没有可撤销的编辑版本")
        emb = await self.writer._embed(f"{version['title']}\n{version['content']}")
        if emb is None: raise MaintenanceError("旧版本 embedding 失败，未撤销")
        def _restore():
            with self.db.transaction():
                self.db.execute("UPDATE episodes SET title=?,content=?,importance=?,protected_until=? WHERE id=?",
                    (version["title"],version["content"],version["importance"],int(time.time())+30*86400,eid))
                self.db.execute("DELETE FROM episode_versions WHERE id=?", (version["id"],))
                keywords=[r["keyword"] for r in self.db.fetchall("SELECT keyword FROM episode_keywords WHERE episode_id=?",(eid,))]
                self.db.execute("DELETE FROM episodes_fts WHERE rowid=?",(eid,)); self.db.insert_fts(eid,version["title"],version["content"],keywords)
                self.vec.upsert(eid,emb,chat_type=ep["chat_type"],session_id=ep["session_id"],group_id=ep["group_id"],is_archived=ep["is_archived"])
            return dict(self.db.fetchone("SELECT * FROM episodes WHERE id=?",(eid,)))
        return await self.db.run(_restore)

    def manual_state(self, eid: int, action: str, reason: str | None = None) -> dict:
        ep=self.db.fetchone("SELECT * FROM episodes WHERE id=?",(eid,))
        if not ep: raise ValueError("episode 不存在")
        if action == "archive": self._set_state(eid,"archived",reason or "用户手动归档")
        elif action == "trash": self._set_state(eid,"trashed",reason or "用户删除")
        elif action == "restore": self._set_state(eid,"active",None,user_restore=True)
        elif action == "unprotect": self.db.execute("UPDATE episodes SET protected_until=NULL,restored_by_user=0 WHERE id=?",(eid,))
        else: raise ValueError("无效操作")
        return dict(self.db.fetchone("SELECT * FROM episodes WHERE id=?",(eid,)))

    def permanent_delete(self, eid: int) -> None:
        ep=self.db.fetchone("SELECT status FROM episodes WHERE id=?",(eid,))
        if not ep or ep["status"] != "trashed": raise ValueError("只能永久删除回收站内容")
        with self.db.transaction():
            self.vec.delete(eid); self.db.execute("DELETE FROM episodes_fts WHERE rowid=?",(eid,))
            self.db.execute("DELETE FROM episode_keywords WHERE episode_id=?",(eid,))
            self.db.execute("DELETE FROM episode_participants WHERE episode_id=?",(eid,))
            self.db.execute("DELETE FROM episode_versions WHERE episode_id=?",(eid,))
            self.db.execute("DELETE FROM episode_merge_sources WHERE merged_episode_id=? OR source_episode_id=?",(eid,eid))
            self.db.execute("DELETE FROM episodes WHERE id=?",(eid,))
