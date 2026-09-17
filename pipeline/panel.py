"""
pipeline/panel.py — 管理面板 API endpoint

只读调试面板（Phase 1e.5），注册到 AstrBot dashboard 上。
访问路径：/api/v1/plugins/extensions/astrbot_plugin_astra_memoir/*

Endpoints:
- GET  /stats           首页数字（episode 总数、今日新增、raw 未处理数等）
- GET  /episodes        Episode 列表，支持筛选和分页
- GET  /episodes/{id}   Episode 详情 + 原始证据回查
- GET  /raw             近期原文列表，支持筛选
- POST /debug/recall    模拟检索，返回候选和各路分数

view_handler 签名: async def h(...) -> dict[str, Any]
AstrBot 会自动 JSON 序列化。Request 通过 bind_request_context 里的 context 取。
"""
from __future__ import annotations

import json
import time
from datetime import datetime

from astrbot.api import logger

from ..storage import MemoirDB, VecStore
from .retriever import Retriever, RetrieverConfig


def _row_to_dict(row) -> dict:
    return dict(row) if row is not None else {}


class MemoirPanel:
    def __init__(
        self,
        db: MemoirDB,
        vec: VecStore,
        retriever: Retriever,
        scheduler,  # BatchScheduler
    ):
        self.db = db
        self.vec = vec
        self.retriever = retriever
        self.scheduler = scheduler

    # ---------- stats ----------

    async def get_stats(self) -> dict:
        def _q():
            now = int(time.time())
            today_start = now - now % 86400  # UTC，够用
            episodes_total = self.db.fetchone(
                "SELECT COUNT(*) AS n FROM episodes"
            )["n"]
            episodes_today = self.db.fetchone(
                "SELECT COUNT(*) AS n FROM episodes WHERE extracted_at >= ?",
                (today_start,),
            )["n"]
            raw_total = self.db.fetchone(
                "SELECT COUNT(*) AS n FROM recent_messages"
            )["n"]
            raw_unproc = self.db.fetchone(
                "SELECT COUNT(*) AS n FROM recent_messages WHERE processed_at IS NULL"
            )["n"]
            missing_vec = len(self.vec.missing_episode_ids(limit=1000))
            last_extracted = self.db.fetchone(
                "SELECT MAX(extracted_at) AS t FROM episodes"
            )["t"]

            sessions_active = self.db.fetchall(
                "SELECT session_id, chat_type, group_id, COUNT(*) AS unprocessed_count "
                "FROM recent_messages WHERE processed_at IS NULL "
                "GROUP BY session_id ORDER BY unprocessed_count DESC LIMIT 20"
            )

            return {
                "episodes_total": episodes_total,
                "episodes_today": episodes_today,
                "raw_total": raw_total,
                "raw_unprocessed": raw_unproc,
                "vec_missing": missing_vec,
                "last_extracted_at": last_extracted,
                "scheduler_running": self.scheduler._main_task is not None,
                "active_sessions": [_row_to_dict(r) for r in sessions_active],
            }

        return {"status": "ok", "data": await self.db.run(_q)}

    # ---------- episodes list ----------

    async def list_episodes(self, request) -> dict:
        q = _query_params(request)
        chat_type = q.get("chat_type")  # private / group / None
        participant = q.get("participant")  # speaker_id
        session_id = q.get("session_id")
        group_id = q.get("group_id")
        search = q.get("q")  # FTS
        limit = min(int(q.get("limit", 50)), 200)
        offset = int(q.get("offset", 0))

        def _q():
            conditions = []
            params: list = []

            if chat_type:
                conditions.append("chat_type = ?")
                params.append(chat_type)
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)
            if group_id:
                conditions.append("group_id = ?")
                params.append(group_id)

            if participant:
                # 用 JOIN participants
                sql_base = (
                    "SELECT DISTINCT e.* FROM episodes e "
                    "JOIN episode_participants p ON p.episode_id = e.id "
                    "WHERE p.speaker_id = ?"
                )
                params_base: list = [participant]
                if conditions:
                    sql_base += " AND " + " AND ".join(f"e.{c}" for c in conditions)
                    params_base.extend(params)
                sql = sql_base + " ORDER BY e.event_end_at DESC LIMIT ? OFFSET ?"
                params_final = [*params_base, limit, offset]
            elif search:
                # FTS
                sql = (
                    "SELECT e.* FROM episodes_fts f "
                    "JOIN episodes e ON e.id = f.rowid "
                    "WHERE episodes_fts MATCH ?"
                )
                params_final = [search]
                if conditions:
                    sql += " AND " + " AND ".join(f"e.{c}" for c in conditions)
                    params_final.extend(params)
                sql += " ORDER BY e.event_end_at DESC LIMIT ? OFFSET ?"
                params_final.extend([limit, offset])
            else:
                sql = "SELECT * FROM episodes"
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                sql += " ORDER BY event_end_at DESC LIMIT ? OFFSET ?"
                params_final = [*params, limit, offset]

            eps = self.db.fetchall(sql, tuple(params_final))
            # 附上 participants
            result = []
            for e in eps:
                ep_dict = _row_to_dict(e)
                parts = self.db.fetchall(
                    "SELECT speaker_id, speaker_name, role FROM episode_participants "
                    "WHERE episode_id = ?",
                    (e["id"],),
                )
                ep_dict["participants"] = [_row_to_dict(p) for p in parts]
                result.append(ep_dict)
            return result

        eps = await self.db.run(_q)
        return {"status": "ok", "data": eps, "count": len(eps)}

    # ---------- episode detail ----------

    async def episode_detail(self, episode_id: str) -> dict:
        try:
            eid = int(episode_id)
        except (ValueError, TypeError):
            return {"status": "error", "message": "invalid episode_id"}

        def _q():
            ep = self.db.fetchone("SELECT * FROM episodes WHERE id = ?", (eid,))
            if ep is None:
                return None
            ep_dict = _row_to_dict(ep)

            parts = self.db.fetchall(
                "SELECT speaker_id, speaker_name, role FROM episode_participants "
                "WHERE episode_id = ?",
                (eid,),
            )
            ep_dict["participants"] = [_row_to_dict(p) for p in parts]

            keywords = self.db.fetchall(
                "SELECT keyword FROM episode_keywords WHERE episode_id = ?",
                (eid,),
            )
            ep_dict["keywords"] = [k["keyword"] for k in keywords]

            # 回查 raw evidence
            try:
                source_ids = json.loads(ep["source_raw_ids"])
            except Exception:
                source_ids = []

            if source_ids:
                placeholders = ",".join("?" * len(source_ids))
                raws = self.db.fetchall(
                    f"SELECT id, speaker_id, speaker_name, role, content, created_at, "
                    f"processed_at IS NOT NULL AS is_processed "
                    f"FROM recent_messages WHERE id IN ({placeholders}) "
                    f"ORDER BY created_at ASC",
                    source_ids,
                )
                ep_dict["raw_evidence"] = [_row_to_dict(r) for r in raws]
                # 缺失的 raw（已被 TTL 清理）
                found_ids = {r["id"] for r in raws}
                ep_dict["raw_missing_ids"] = [
                    i for i in source_ids if i not in found_ids
                ]
            else:
                ep_dict["raw_evidence"] = []
                ep_dict["raw_missing_ids"] = []

            # vec 状态
            vec_row = self.db.fetchone(
                "SELECT episode_id FROM episode_vec WHERE episode_id = ?", (eid,)
            )
            ep_dict["vec_present"] = vec_row is not None

            return ep_dict

        ep = await self.db.run(_q)
        if ep is None:
            return {"status": "error", "message": "episode not found"}
        return {"status": "ok", "data": ep}

    # ---------- recent raw ----------

    async def list_raw(self, request) -> dict:
        q = _query_params(request)
        chat_type = q.get("chat_type")
        session_id = q.get("session_id")
        role = q.get("role")
        processed = q.get("processed")  # "true" / "false" / None
        limit = min(int(q.get("limit", 100)), 500)
        offset = int(q.get("offset", 0))

        def _q():
            conditions = []
            params: list = []
            if chat_type:
                conditions.append("chat_type = ?")
                params.append(chat_type)
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)
            if role:
                conditions.append("role = ?")
                params.append(role)
            if processed == "true":
                conditions.append("processed_at IS NOT NULL")
            elif processed == "false":
                conditions.append("processed_at IS NULL")

            sql = "SELECT * FROM recent_messages"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = self.db.fetchall(sql, tuple(params))
            return [_row_to_dict(r) for r in rows]

        return {"status": "ok", "data": await self.db.run(_q)}

    # ---------- debug recall ----------

    async def debug_recall(self, request) -> dict:
        body = await _json_body(request)
        query = str(body.get("query", "")).strip()
        chat_type = body.get("chat_type", "private")
        session_id = body.get("session_id", "debug_session")
        group_id = body.get("group_id") or None
        current_speaker_id = str(body.get("current_speaker_id", ""))

        if not query:
            return {"status": "error", "message": "missing query"}

        results = await self.retriever.recall(
            query,
            session_id=session_id,
            chat_type=chat_type,
            group_id=group_id,
            current_speaker_id=current_speaker_id,
        )

        # 返回带调试字段的结果
        return {
            "status": "ok",
            "data": {
                "query": query,
                "scope": {
                    "chat_type": chat_type,
                    "session_id": session_id,
                    "group_id": group_id,
                    "current_speaker_id": current_speaker_id,
                },
                "results": [
                    {
                        "episode_id": r.episode_id,
                        "title": r.title,
                        "content": r.content,
                        "chat_type": r.chat_type,
                        "event_start_at": r.event_start_at,
                        "participants": r.participants,
                        "vec_distance": r.debug_vec_distance,
                        "fts_hit": r.debug_fts_hit,
                        "rrf_score": r.debug_rrf_score,
                    }
                    for r in results
                ],
            },
        }


# ---------- request helpers ----------

def _query_params(request) -> dict:
    """兼容 fastapi.Request / quart.Request / 我们自己 mock 的 dict-like。"""
    try:
        return dict(request.query_params)
    except AttributeError:
        pass
    try:
        return dict(request.args)
    except AttributeError:
        pass
    return {}


async def _json_body(request) -> dict:
    """兼容 fastapi.Request.json() / quart.Request.get_json()。"""
    try:
        return await request.json()
    except AttributeError:
        pass
    try:
        return await request.get_json()
    except AttributeError:
        pass
    return {}


# ---------- register on context ----------

def register_panel_routes(context, panel: MemoirPanel):
    """把 5 个 endpoint 注册到 AstrBot dashboard 上。"""
    async def h_stats():
        return await panel.get_stats()

    async def h_list_episodes(request):
        return await panel.list_episodes(request)

    async def h_episode_detail(episode_id: str):
        return await panel.episode_detail(episode_id)

    async def h_list_raw(request):
        return await panel.list_raw(request)

    async def h_debug_recall(request):
        return await panel.debug_recall(request)

    context.register_web_api(
        "/stats", h_stats, methods=["GET"],
        desc="Memoir status: episode 总数、今日、未处理 raw 等",
    )
    context.register_web_api(
        "/episodes", h_list_episodes, methods=["GET"],
        desc="Memoir episodes list with filtering",
    )
    context.register_web_api(
        "/episodes/<episode_id>", h_episode_detail, methods=["GET"],
        desc="Memoir episode detail with raw evidence",
    )
    context.register_web_api(
        "/raw", h_list_raw, methods=["GET"],
        desc="Memoir recent raw messages",
    )
    context.register_web_api(
        "/debug/recall", h_debug_recall, methods=["POST"],
        desc="Memoir retrieval simulator (debug)",
    )

    logger.info("[Memoir] panel API endpoints registered (5 routes)")
