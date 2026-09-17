"""
pipeline/panel.py — 管理面板 API endpoint

只读调试面板（Phase 1e.5），注册到 AstrBot dashboard 上。
访问路径：/api/v1/plugins/extensions/astrbot_plugin_astra_memoir/*

Endpoints:
- GET  /stats                   首页数字
- GET  /episodes                Episode 列表，支持筛选和分页
- GET  /episodes/<episode_id>   Episode 详情 + 原始证据回查
- GET  /raw                     近期原文列表，支持筛选
- POST /debug/recall            模拟检索

View handler 签名: view_func(**path_params) —— AstrBot 只把 URL path 参数作为
kwargs 传入。要拿 query string / body，用 astrbot.api.web.request module-level proxy。
"""
from __future__ import annotations

import json
import time

from astrbot.api import logger
from astrbot.api.web import request as astr_request

from ..storage import MemoirDB, VecStore
from .retriever import Retriever


def _row_to_dict(row) -> dict:
    return dict(row) if row is not None else {}


def _query(key: str, default=None):
    try:
        return astr_request.query.get(key, default)
    except Exception:
        return default


async def _json_body() -> dict:
    try:
        body = await astr_request.json()
        if isinstance(body, dict):
            return body
    except Exception:
        pass
    return {}


class MemoirPanel:
    def __init__(
        self,
        db: MemoirDB,
        vec: VecStore,
        retriever: Retriever,
        scheduler,
    ):
        self.db = db
        self.vec = vec
        self.retriever = retriever
        self.scheduler = scheduler

    # ---------- stats ----------

    async def get_stats(self) -> dict:
        def _q():
            now = int(time.time())
            today_start = now - now % 86400
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

        try:
            return {"status": "ok", "data": await self.db.run(_q)}
        except Exception as e:
            logger.exception("[Memoir] get_stats 异常")
            return {"status": "error", "message": str(e), "data": {}}

    # ---------- episodes list ----------

    async def list_episodes(self) -> dict:
        chat_type = _query("chat_type")
        participant = _query("participant")
        session_id = _query("session_id")
        group_id = _query("group_id")
        search = _query("q")
        try:
            limit = min(int(_query("limit", 50)), 200)
        except (TypeError, ValueError):
            limit = 50
        try:
            offset = int(_query("offset", 0))
        except (TypeError, ValueError):
            offset = 0

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
                sql = (
                    "SELECT e.* FROM episodes_fts "
                    "JOIN episodes e ON e.id = episodes_fts.rowid "
                    "WHERE episodes_fts MATCH ?"
                )
                params_final: list = [search]
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

        try:
            eps = await self.db.run(_q)
            return {"status": "ok", "data": eps, "count": len(eps)}
        except Exception as e:
            logger.exception("[Memoir] list_episodes 异常")
            return {"status": "error", "message": str(e), "data": []}

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
                found_ids = {r["id"] for r in raws}
                ep_dict["raw_missing_ids"] = [
                    i for i in source_ids if i not in found_ids
                ]
            else:
                ep_dict["raw_evidence"] = []
                ep_dict["raw_missing_ids"] = []

            vec_row = self.db.fetchone(
                "SELECT episode_id FROM episode_vec WHERE episode_id = ?", (eid,)
            )
            ep_dict["vec_present"] = vec_row is not None

            return ep_dict

        try:
            ep = await self.db.run(_q)
            if ep is None:
                return {"status": "error", "message": "episode not found"}
            return {"status": "ok", "data": ep}
        except Exception as e:
            logger.exception("[Memoir] episode_detail 异常")
            return {"status": "error", "message": str(e)}

    # ---------- recent raw ----------

    async def list_raw(self) -> dict:
        chat_type = _query("chat_type")
        session_id = _query("session_id")
        role = _query("role")
        processed = _query("processed")
        try:
            limit = min(int(_query("limit", 100)), 500)
        except (TypeError, ValueError):
            limit = 100
        try:
            offset = int(_query("offset", 0))
        except (TypeError, ValueError):
            offset = 0

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

        try:
            return {"status": "ok", "data": await self.db.run(_q)}
        except Exception as e:
            logger.exception("[Memoir] list_raw 异常")
            return {"status": "error", "message": str(e), "data": []}

    # ---------- debug recall ----------

    async def debug_recall(self) -> dict:
        body = await _json_body()
        query = str(body.get("query", "")).strip()
        chat_type = body.get("chat_type", "private")
        session_id = body.get("session_id", "debug_session")
        group_id = body.get("group_id") or None
        current_speaker_id = str(body.get("current_speaker_id", ""))

        if not query:
            return {"status": "error", "message": "missing query"}

        try:
            results = await self.retriever.recall(
                query,
                session_id=session_id,
                chat_type=chat_type,
                group_id=group_id,
                current_speaker_id=current_speaker_id,
            )
        except Exception as e:
            logger.exception("[Memoir] debug_recall 异常")
            return {"status": "error", "message": str(e)}

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


# ---------- register on context ----------

_ROUTE_PREFIX = "/astrbot_plugin_astra_memoir"


def register_panel_routes(context, panel: MemoirPanel):
    """
    View handler 只接受 URL path 参数作为 kwargs（AstrBot 用 view_func(**path_params) 调用）。
    query string / body 通过 astrbot.api.web.request module-level proxy 获取。
    Route 必须带 plugin_name 前缀。
    """
    async def h_stats():
        return await panel.get_stats()

    async def h_list_episodes():
        return await panel.list_episodes()

    async def h_episode_detail(episode_id: str):
        return await panel.episode_detail(episode_id)

    async def h_list_raw():
        return await panel.list_raw()

    async def h_debug_recall():
        return await panel.debug_recall()

    context.register_web_api(
        f"{_ROUTE_PREFIX}/stats", h_stats, methods=["GET"],
        desc="Memoir status: episode 总数、今日、未处理 raw 等",
    )
    context.register_web_api(
        f"{_ROUTE_PREFIX}/episodes", h_list_episodes, methods=["GET"],
        desc="Memoir episodes list with filtering",
    )
    context.register_web_api(
        f"{_ROUTE_PREFIX}/episodes/<episode_id>", h_episode_detail, methods=["GET"],
        desc="Memoir episode detail with raw evidence",
    )
    context.register_web_api(
        f"{_ROUTE_PREFIX}/raw", h_list_raw, methods=["GET"],
        desc="Memoir recent raw messages",
    )
    context.register_web_api(
        f"{_ROUTE_PREFIX}/debug/recall", h_debug_recall, methods=["POST"],
        desc="Memoir retrieval simulator (debug)",
    )

    logger.info("[Memoir] panel API endpoints registered (5 routes)")
