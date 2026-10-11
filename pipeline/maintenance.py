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
from .extractor import suspected_name_drift

MAINTENANCE_LOGIC_VERSION = 4


MAINTENANCE_PROMPT = """你是 Astra Memoir 的记忆整理器。你不是写日报，而是在同一小批独立 episode 中判断长期价值。
只能输出严格 JSON，只允许 keep / archive / merge，绝不 delete。
判断重点：几天到几周后重新提起，对 Astra 是否仍有价值？
普通画图请求、表情包、一次性玩笑、重复闲聊、无后续小事优先 archive；明确决定、人物状态变化、项目进展、新设定、持续话题和有后续影响的互动保留。
同一持续事件的碎片可以 merge；不同主题绝不能因为同一天而合并。merge 必须写成一条自然、完整、忠实的事件记忆，不添加原文没有的事实。
人称规则是硬规则：只有输入中 astra_participated=true，且事实确实是 role=assistant 的 Astra 所做或所说，才能用“我”指代 Astra；群友或其他 AI 的言行必须使用 participants 中的正式名字。若 astra_participated=false，标题和正文都禁止使用“我/我们”，必须客观第三人称记录，也不要虚构“我看到/我得知”。
姓名规则是硬规则：participants 中的正式名字必须逐字复制，标题和正文必须一致；不得改同音字、形近字或自行润色姓名。
每个输入 id 必须且只能出现一次。keep/archive 的 ids 必须只有一个；merge 至少两个。
edited_by_user=true 表示用户修正过事实：允许参与整理与 merge，但必须把当前最新版正文视为高可信事实，不能使用旧版本；合并结果不得与它冲突。
importance 为 1~5 的重新评估结果。
输出：{"decisions":[{"action":"keep","ids":[1],"importance":3,"reason":"..."},{"action":"archive","ids":[2],"importance":1,"reason":"一次性闲聊"},{"action":"merge","ids":[3,4],"title":"...","content":"...","importance":4,"reason":"同一持续事件"}]}"""

LONG_TERM_PROMPT = MAINTENANCE_PROMPT.replace("几天到几周后", "几个月后") + """
这是长期二次睡眠，不是每日整理或月报。只合并主题高度一致、主要人物一致、时间关系合理，且属于同一项目、持续事件或状态发展的记忆；embedding/措辞相似本身不构成合并理由。
stage=1 主要压缩30天以上的重复碎片，archive要保守；stage=2 把同一持续项目的具体事件压缩成阶段记忆并保留转折；stage=3 筛选半年后的长期价值。
importance=5 默认长期保留，禁止自动 archive；importance=4 不得轻易 archive。importance=2 且半年后没有发展可以建议 archive。
pinned 条目不会出现在输入中。输出仍必须是独立事件，严禁生成把无关主题揉在一起的月报。
若 merge 包含 edited_by_user=true 的来源，生成内容必须完整兼容该条当前 title/content；无法确认时 keep，不得猜测或覆盖用户修正事实。"""


class MaintenanceError(RuntimeError):
    pass


class MaintenanceManager:
    def __init__(self, db, vec, writer, extractor, *, batch_size: int = 12):
        self.db, self.vec, self.writer, self.extractor = db, vec, writer, extractor
        self.batch_size = max(4, min(int(batch_size), 20))
        self._run_lock = asyncio.Lock()

    @staticmethod
    def _long_term_archive_reason(reason: str) -> str:
        text = str(reason or "")
        if any(word in text for word in ("重复", "duplicate", "覆盖")):
            return "long_term_duplicate"
        if any(word in text for word in ("过期", "失去意义", "阶段细节")):
            return "long_term_expired"
        return "long_term_low_value"

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

    def _long_term_batches(self, rows: list[dict]) -> list[list[dict]]:
        """跨日期找持续主题，但先按来源类型分桶，再以主题/人物/会话贪心聚类。"""
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            buckets[row["chat_type"]].append(row)
        result = []
        for items in buckets.values():
            remaining = sorted(items, key=lambda r: (r["event_start_at"], r["id"]))
            while remaining:
                seed = remaining.pop(0)
                cluster = [seed]
                union = self._topic_tokens(seed)
                participants = {p.get("speaker_id") for p in seed.get("participants", [])}
                while remaining and len(cluster) < min(self.batch_size, 15):
                    best_i, best = -1, 0.0
                    for i, candidate in enumerate(remaining):
                        tokens = self._topic_tokens(candidate)
                        lexical = len(union & tokens) / max(1, len(union | tokens))
                        if lexical <= .02:
                            continue
                        cparts = {p.get("speaker_id") for p in candidate.get("participants", [])}
                        people = len(participants & cparts) / max(1, len(participants | cparts))
                        same_session = 1.0 if candidate["session_id"] == seed["session_id"] else 0.0
                        score = lexical * .70 + people * .20 + same_session * .10
                        if score > best:
                            best_i, best = i, score
                    # Keep unrelated themes out of the same LLM merge decision.
                    if best_i < 0 or best < .08:
                        break
                    chosen = remaining.pop(best_i)
                    cluster.append(chosen)
                    union |= self._topic_tokens(chosen)
                    participants |= {p.get("speaker_id") for p in chosen.get("participants", [])}
                result.append(cluster)
        return result

    async def _ask(self, rows: list[dict], *, long_term_stage: int | None = None) -> list[dict]:
        provider = self.extractor._get_provider()
        if provider is None:
            raise MaintenanceError("整理模型不可用")
        payload = [{"id": r["id"], "title": r["title"], "content": r["content"],
                    "importance": r["importance"], "source": r["chat_type"],
                    "edited_by_user": bool(r.get("edited_by_user") or
                                           r.get("contains_user_edited_source")),
                    "consolidation_stage": int(r.get("consolidation_stage") or 0),
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
            if long_term_stage:
                prompt = f"<target_stage>{long_term_stage}</target_stage>\n" + prompt
            if attempt:
                prompt += ("\n上次输出被硬校验拒绝，原因：" + validation_error +
                           "。请修正后重新输出完整 JSON；覆盖每个 id 恰好一次。")
            try:
                resp = await asyncio.wait_for(
                    provider.text_chat(prompt=prompt, system_prompt=(
                        LONG_TERM_PROMPT if long_term_stage else MAINTENANCE_PROMPT)),
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
                    if long_term_stage and action == "archive" and any(
                            r["id"] in ids and int(r.get("importance") or 0) >= 5 for r in rows):
                        raise ValueError("importance=5 的长期记忆禁止自动 archive")
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
                        participant_names = {
                            str(p.get("name", "")).strip()
                            for r in rows if r["id"] in ids
                            for p in r.get("participants", [])
                            if str(p.get("name", "")).strip()
                        }
                        name_error = suspected_name_drift(
                            str(d.get("title", "")), str(d.get("content", "")),
                            participant_names,
                        )
                        if name_error:
                            raise ValueError(name_error)
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
                      target_date: str | None = None, stage: int | None = None) -> dict:
        async with self._run_lock:
            return await self._preview_unlocked(
                start_ts, end_ts, run_type=run_type, target_date=target_date, stage=stage)

    async def preview_long_term(self, stage: int) -> dict:
        if stage not in (1, 2, 3):
            raise ValueError("长期整理阶段必须是 1、2 或 3")
        now = int(time.time())
        async with self._run_lock:
            return await self._preview_unlocked(
                0, now, run_type="long_term", target_date=None, stage=stage)

    async def _preview_unlocked(self, start_ts: int, end_ts: int, *, run_type: str,
                                target_date: str | None, stage: int | None = None) -> dict:
        if run_type not in ("history", "nightly", "long_term") or end_ts <= start_ts:
            raise ValueError("无效整理范围")
        if run_type == "long_term" and stage not in (1, 2, 3):
            raise ValueError("长期整理缺少有效 stage")
        backup_path = await self.db.run(self.backup)
        now = int(time.time())
        def _create():
            cur = self.db.execute(
                "INSERT INTO maintenance_runs(run_type,stage,target_start,target_end,status,backup_path,created_at,logic_version) "
                "VALUES(?,?,?,?,?,?,?,?)", (run_type, stage, start_ts, end_ts, "generating",
                                             backup_path, now, MAINTENANCE_LOGIC_VERSION))
            if target_date:
                self.db.execute(
                    "INSERT OR REPLACE INTO maintenance_days(target_date,run_id,status,updated_at) VALUES(?,?,?,?)",
                    (target_date, cur.lastrowid, "running", now))
            return cur.lastrowid
        run_id = await self.db.run(_create)
        try:
            def _load():
                if run_type == "long_term":
                    minimum_age = {1: 30, 2: 90, 3: 180}[stage]
                    rows = self.db.fetchall(
                        "SELECT e.*, COALESCE(group_concat(k.keyword,' '),'') keywords FROM episodes e "
                        "LEFT JOIN episode_keywords k ON k.episode_id=e.id "
                        "WHERE e.status='active' AND e.pinned=0 AND e.consolidation_stage=? "
                        "AND e.event_end_at<=? AND (e.protected_until IS NULL OR e.protected_until<=?) "
                        "AND (?=1 OR e.last_consolidated_at IS NULL OR e.last_consolidated_at<=?) "
                        "GROUP BY e.id ORDER BY e.event_start_at,e.id",
                        (stage - 1, now - minimum_age * 86400, now, stage, now - 28 * 86400))
                else:
                    rows = self.db.fetchall(
                        "SELECT e.*, COALESCE(group_concat(k.keyword,' '),'') keywords FROM episodes e "
                        "LEFT JOIN episode_keywords k ON k.episode_id=e.id "
                        "WHERE e.status='active' AND e.pinned=0 AND e.event_start_at>=? AND e.event_start_at<? "
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
            batches = (self._long_term_batches(rows) if run_type == "long_term"
                       else self._small_batches(rows))
            await self.db.run(lambda: self.db.execute(
                "UPDATE maintenance_runs SET source_count=?,batch_count=? WHERE id=?",
                (len(rows), len(batches), run_id)))
            decisions = []
            for completed, batch in enumerate(batches, 1):
                decisions.extend(await self._ask(batch, long_term_stage=stage if run_type == "long_term" else None))
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
                                "before_archive_reason,before_merged_into,before_importance,"
                                "before_consolidation_stage,before_last_consolidated_at,before_pinned,"
                                "before_title,before_content) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (run_id,d["action"],eid,group,d.get("title"),d.get("content"),
                                 d["importance"],str(d.get("reason", ""))[:300],row_map[eid]["status"],
                                 row_map[eid]["archive_reason"],row_map[eid]["merged_into"],row_map[eid]["importance"],
                                 row_map[eid].get("consolidation_stage", 0),
                                 row_map[eid].get("last_consolidated_at"), row_map[eid].get("pinned", 0),
                                 row_map[eid]["title"], row_map[eid]["content"]))
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
            "SELECT a.*,e.title source_title,e.content source_content,"
            "(e.edited_by_user OR e.contains_user_edited_source) source_edited_by_user "
            "FROM maintenance_actions a "
            "LEFT JOIN episodes e ON e.id=a.source_episode_id WHERE a.run_id=? ORDER BY a.id", (run_id,))
        return {**dict(run), "actions": [dict(a) for a in actions]}

    def list_runs(self) -> list[dict]:
        return [dict(r) for r in self.db.fetchall(
            "SELECT * FROM maintenance_runs ORDER BY created_at DESC,id DESC LIMIT 100")]

    def long_term_candidate_counts(self, now: int | None = None) -> dict:
        now = int(time.time()) if now is None else int(now)
        return {str(stage): self.db.fetchone(
            "SELECT COUNT(*) n FROM episodes WHERE status='active' AND pinned=0 "
            "AND consolidation_stage=? AND event_end_at<=? "
            "AND (protected_until IS NULL OR protected_until<=?) "
            "AND (?=1 OR last_consolidated_at IS NULL OR last_consolidated_at<=?)",
            (stage - 1, now - days * 86400, now, stage, now - 28 * 86400))["n"]
            for stage, days in ((1, 30), (2, 90), (3, 180))}

    def _recount_run(self, run_id: int) -> None:
        rows = self.db.fetchall(
            "SELECT action,merge_group,COUNT(*) n FROM maintenance_actions "
            "WHERE run_id=? GROUP BY action,merge_group", (run_id,))
        keep = sum(r["n"] for r in rows if r["action"] == "keep")
        archive = sum(r["n"] for r in rows if r["action"] == "archive")
        merge_sources = sum(r["n"] for r in rows if r["action"] == "merge")
        merge_results = sum(1 for r in rows if r["action"] == "merge")
        self.db.execute(
            "UPDATE maintenance_runs SET keep_count=?,archive_count=?,merge_source_count=?,"
            "merge_result_count=? WHERE id=?",
            (keep, archive, merge_sources, merge_results, run_id))

    def revise_preview_action(self, run_id: int, action_id: int, new_action: str) -> dict:
        if new_action not in ("keep", "archive"):
            raise ValueError("Preview 只能改为 keep 或 archive")
        run = self.db.fetchone("SELECT * FROM maintenance_runs WHERE id=?", (run_id,))
        action = self.db.fetchone(
            "SELECT * FROM maintenance_actions WHERE id=? AND run_id=?", (action_id, run_id))
        if not run or run["status"] != "preview" or not action:
            raise ValueError("只有待确认 Preview 的有效项目可以调整")
        if new_action == "archive":
            ids = ([action["source_episode_id"]] if action["action"] != "merge" else
                   [r["source_episode_id"] for r in self.db.fetchall(
                       "SELECT source_episode_id FROM maintenance_actions WHERE run_id=? AND merge_group=?",
                       (run_id, action["merge_group"]))])
            placeholders = ",".join("?" * len(ids))
            extra = " OR importance=5" if run["run_type"] == "long_term" else ""
            if self.db.fetchone(
                    f"SELECT 1 FROM episodes WHERE id IN ({placeholders}) AND (pinned=1{extra})", ids):
                raise ValueError("已保护或 importance=5 的长期记忆不能建议归档")
        with self.db.transaction():
            if action["action"] == "merge":
                self.db.execute(
                    "UPDATE maintenance_actions SET action=?,merge_group=NULL,proposed_title=NULL,"
                    "proposed_content=NULL,reason=? WHERE run_id=? AND merge_group=?",
                    (new_action, "用户在 Preview 中调整", run_id, action["merge_group"]))
            else:
                self.db.execute(
                    "UPDATE maintenance_actions SET action=?,merge_group=NULL,reason=? WHERE id=?",
                    (new_action, "用户在 Preview 中调整", action_id))
            self._recount_run(run_id)
        return self.run_detail(run_id)

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
            embeddings[key] = emb
        now = int(time.time())
        is_long_term = detail["run_type"] == "long_term"
        target_stage = int(detail.get("stage") or 0)
        def _validate_sources():
            for a in detail["actions"]:
                ep = self.db.fetchone("SELECT * FROM episodes WHERE id=?", (a["source_episode_id"],))
                if (not ep or ep["status"] != a["before_status"]
                        or ep["importance"] != a["before_importance"]
                        or ep["pinned"] != a.get("before_pinned", 0)
                        or ep["title"] != (a.get("before_title") or ep["title"])
                        or ep["content"] != (a.get("before_content") or ep["content"])
                        or (ep["protected_until"] and ep["protected_until"] > now)
                        or self.db.fetchone("SELECT 1 FROM episode_versions WHERE episode_id=? AND saved_at>?",
                                           (a["source_episode_id"], detail["created_at"]))):
                    raise ValueError(f"episode #{a['source_episode_id']} 在 Preview 后已变化，请重新生成方案")
                if ep["pinned"]:
                    raise ValueError(f"episode #{ep['id']} 已保护，不能应用自动整理")
                if is_long_term and a["action"] == "archive" and ep["importance"] >= 5:
                    raise ValueError(f"episode #{ep['id']} importance=5，禁止长期自动归档")
        await self.db.run(_validate_sources)
        def _apply():
            with self.db.transaction():
                for a in detail["actions"]:
                    ep = self.db.fetchone("SELECT * FROM episodes WHERE id=?", (a["source_episode_id"],))
                    if a["action"] == "keep":
                        self.db.execute("UPDATE episodes SET importance=? WHERE id=?",
                                        (a["proposed_importance"],ep["id"]))
                    elif a["action"] == "archive":
                        self.db.execute("UPDATE episodes SET importance=? WHERE id=?",
                                        (a["proposed_importance"],ep["id"]))
                        reason = (self._long_term_archive_reason(a["reason"]) if is_long_term
                                  else (a["reason"] or "低长期价值"))
                        self._set_state(ep["id"], "archived", reason)
                    if is_long_term:
                        self.db.execute(
                            "UPDATE episodes SET consolidation_stage=?,last_consolidated_at=? WHERE id=?",
                            (target_stage, now, ep["id"]))
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
                    if is_long_term:
                        has_user_edits = int(any(
                            e["edited_by_user"] or e["contains_user_edited_source"]
                            for e in sources))
                        self.db.execute(
                            "UPDATE episodes SET consolidation_stage=?,last_consolidated_at=?,"
                            "contains_user_edited_source=? WHERE id=?",
                            (target_stage, now, has_user_edits, eid))
                    participant_rows = self.db.fetchall(
                        f"SELECT DISTINCT speaker_id,speaker_name,role FROM episode_participants WHERE episode_id IN ({','.join('?'*len(sources))})",
                        [e["id"] for e in sources])
                    self.db.insert_participants(eid,[dict(p) for p in participant_rows])
                    keywords = [r["keyword"] for r in self.db.fetchall(
                        f"SELECT DISTINCT keyword FROM episode_keywords WHERE episode_id IN ({','.join('?'*len(sources))})",
                        [e["id"] for e in sources])]
                    self.db.insert_keywords(eid,keywords); self.db.insert_fts(eid,first["proposed_title"],first["proposed_content"],keywords)
                    self.vec.optional_upsert(eid,embeddings[key],chat_type=base["chat_type"],session_id=base["session_id"],group_id=base["group_id"])
                    for e,a in zip(sources,actions):
                        reason = "long_term_merged" if is_long_term else f"已合并到 #{eid}"
                        self._set_state(e["id"],"archived",reason,merged_into=eid)
                        if is_long_term:
                            self.db.execute(
                                "UPDATE episodes SET consolidation_stage=?,last_consolidated_at=? WHERE id=?",
                                (target_stage, now, e["id"]))
                        self.db.execute("INSERT INTO episode_merge_sources VALUES(?,?)",(eid,e["id"]))
                        self.db.execute("UPDATE maintenance_actions SET result_episode_id=? WHERE id=?",(eid,a["id"]))
                self.db.execute("UPDATE maintenance_runs SET status='applied',applied_at=? WHERE id=?",(now,run_id))
                self.db.execute("UPDATE maintenance_days SET status='applied',updated_at=? WHERE run_id=?",(now,run_id))
            return self.run_detail(run_id)
        return await self.db.run(_apply)

    def _set_state(self, eid: int, status: str, reason: str | None = None,
                   *, merged_into=None, user_restore=False):
        archived = 0 if status == "active" else 1
        self.db.execute("UPDATE episodes SET status=?,is_archived=?,archive_reason=?,merged_into=?,"
                        "restored_by_user=CASE WHEN ? THEN 1 ELSE restored_by_user END WHERE id=?",
                        (status,archived,reason,merged_into,int(user_restore),eid))
        if self.db.vector_ready:
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
                        or self.db.fetchone("SELECT 1 FROM episode_versions WHERE episode_id=? AND saved_at>?",
                                           (a["source_episode_id"], detail["applied_at"] or 0))):
                    raise ValueError(f"episode #{a['source_episode_id']} 在整理后被用户修改，拒绝覆盖；请先人工检查")
            for eid in {a["result_episode_id"] for a in detail["actions"] if a["result_episode_id"]}:
                ep = self.db.fetchone("SELECT * FROM episodes WHERE id=?", (eid,))
                if (not ep or ep["status"] != "active" or ep["edited_by_user"]
                        or ep["pinned"]):
                    raise ValueError(f"合并 episode #{eid} 已被用户修改，拒绝覆盖；请先人工检查")
        await self.db.run(_validate_undo)
        def _undo():
            with self.db.transaction():
                result_ids = {a["result_episode_id"] for a in detail["actions"] if a["result_episode_id"]}
                for eid in result_ids:
                    self._set_state(eid,"trashed","撤销整理产生的合并记忆")
                for a in detail["actions"]:
                    archived = 0 if a["before_status"] == "active" else 1
                    self.db.execute("UPDATE episodes SET status=?,is_archived=?,archive_reason=?,"
                        "merged_into=?,importance=?,consolidation_stage=?,last_consolidated_at=? WHERE id=?",
                        (a["before_status"],archived,a["before_archive_reason"],a["before_merged_into"],
                         a["before_importance"],a.get("before_consolidation_stage",0),
                         a.get("before_last_consolidated_at"),a["source_episode_id"]))
                    if self.db.vector_ready:
                        self.db.execute("UPDATE episode_vec SET is_archived=? WHERE episode_id=?",(archived,a["source_episode_id"]))
                self.db.execute("UPDATE maintenance_runs SET status='undone',undone_at=? WHERE id=?",(now,run_id))
            return self.run_detail(run_id)
        return await self.db.run(_undo)

    async def edit_episode(self, eid: int, title: str, content: str, importance: int) -> dict:
        if not title.strip() or not content.strip() or not 1 <= importance <= 5:
            raise ValueError("标题、正文和 importance(1~5) 必须有效")
        emb = await self.writer._embed(f"{title.strip()}\n{content.strip()}")
        now = int(time.time())
        def _edit():
            ep=self.db.fetchone("SELECT * FROM episodes WHERE id=?",(eid,))
            if not ep: raise ValueError("episode 不存在")
            with self.db.transaction():
                self.db.execute("INSERT INTO episode_versions(episode_id,title,content,importance,saved_at) VALUES(?,?,?,?,?)",
                                (eid,ep["title"],ep["content"],ep["importance"],now))
                self.db.execute("UPDATE episodes SET title=?,content=?,importance=?,edited_by_user=1 WHERE id=?",
                                (title.strip(),content.strip(),importance,eid))
                keywords=[r["keyword"] for r in self.db.fetchall("SELECT keyword FROM episode_keywords WHERE episode_id=?",(eid,))]
                self.db.execute("DELETE FROM episodes_fts WHERE rowid=?",(eid,)); self.db.insert_fts(eid,title.strip(),content.strip(),keywords)
                self.vec.optional_upsert(eid,emb,chat_type=ep["chat_type"],session_id=ep["session_id"],group_id=ep["group_id"],is_archived=ep["is_archived"])
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
        def _restore():
            with self.db.transaction():
                self.db.execute("UPDATE episodes SET title=?,content=?,importance=? WHERE id=?",
                    (version["title"],version["content"],version["importance"],eid))
                self.db.execute("DELETE FROM episode_versions WHERE id=?", (version["id"],))
                keywords=[r["keyword"] for r in self.db.fetchall("SELECT keyword FROM episode_keywords WHERE episode_id=?",(eid,))]
                self.db.execute("DELETE FROM episodes_fts WHERE rowid=?",(eid,)); self.db.insert_fts(eid,version["title"],version["content"],keywords)
                self.vec.optional_upsert(eid,emb,chat_type=ep["chat_type"],session_id=ep["session_id"],group_id=ep["group_id"],is_archived=ep["is_archived"])
            return dict(self.db.fetchone("SELECT * FROM episodes WHERE id=?",(eid,)))
        return await self.db.run(_restore)

    def manual_state(self, eid: int, action: str, reason: str | None = None) -> dict:
        ep=self.db.fetchone("SELECT * FROM episodes WHERE id=?",(eid,))
        if not ep: raise ValueError("episode 不存在")
        if action == "archive": self._set_state(eid,"archived",reason or "用户手动归档")
        elif action == "trash": self._set_state(eid,"trashed",reason or "用户删除")
        elif action == "restore": self._set_state(eid,"active",None,user_restore=True)
        elif action == "unprotect": self.db.execute("UPDATE episodes SET protected_until=NULL,restored_by_user=0 WHERE id=?",(eid,))
        elif action == "pin": self.db.execute("UPDATE episodes SET pinned=1 WHERE id=?",(eid,))
        elif action == "unpin": self.db.execute("UPDATE episodes SET pinned=0 WHERE id=?",(eid,))
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
