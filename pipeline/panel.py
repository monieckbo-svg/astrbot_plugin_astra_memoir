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
import sqlite3
import time
import asyncio
from datetime import datetime, timedelta

from astrbot.api import logger
from astrbot.api.web import request as astr_request

from ..storage import MemoirDB, VecStore
from .embedding import provider_display_name
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
        writer,
        embedding_provider,
        embedding_provider_id: str,
        embedding_dim: int,
        maintenance=None,
    ):
        self.db = db
        self.vec = vec
        self.retriever = retriever
        self.scheduler = scheduler
        self.writer = writer
        self.embedding_provider = embedding_provider
        self.embedding_provider_id = embedding_provider_id
        self.embedding_dim = embedding_dim
        self.maintenance = maintenance
        self._repair_lock = asyncio.Lock()
        self._maintenance_lock = asyncio.Lock()
        self._maintenance_tasks: set[asyncio.Task] = set()

    # ---------- identities ----------

    async def list_identities(self) -> dict:
        try:
            cards = await self.db.run(self.db.identities.list_all)
            return {"status": "ok", "data": cards}
        except Exception as e:
            logger.exception("[Memoir] list_identities failed")
            return {"status": "error", "message": str(e)}

    async def save_identities(self) -> dict:
        body = await _json_body()
        try:
            cards = await self.db.run(
                self.db.identities.update_many, body.get("identities")
            )
            return {"status": "ok", "data": cards}
        except (ValueError, TypeError) as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.exception("[Memoir] save_identities failed")
            return {"status": "error", "message": str(e)}

    async def merge_identities(self) -> dict:
        body = await _json_body()
        try:
            card = await self.db.run(
                self.db.identities.merge,
                body.get("source_qq_id"), body.get("target_qq_id"),
            )
            return {"status": "ok", "data": card}
        except (ValueError, TypeError) as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.exception("[Memoir] merge_identities failed")
            return {"status": "error", "message": str(e)}

    # ---------- stats ----------

    async def get_stats(self) -> dict:
        def _q():
            now = int(time.time())
            today_start = now - now % 86400
            episodes_total = self.db.fetchone(
                "SELECT COUNT(*) AS n FROM episodes"
            )["n"]
            episodes_archived = self.db.fetchone(
                "SELECT COUNT(*) AS n FROM episodes WHERE status = 'archived'"
            )["n"]
            episodes_active = self.db.fetchone(
                "SELECT COUNT(*) AS n FROM episodes WHERE status = 'active'"
            )["n"]
            episodes_trashed = self.db.fetchone(
                "SELECT COUNT(*) AS n FROM episodes WHERE status = 'trashed'"
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
            missing_vec = self.db.fetchone(
                "SELECT COUNT(*) AS n FROM episodes e LEFT JOIN episode_vec v "
                "ON v.episode_id = e.id WHERE v.episode_id IS NULL AND e.status != 'trashed'"
            )["n"]
            vector_eligible = self.db.fetchone(
                "SELECT COUNT(*) AS n FROM episodes WHERE status != 'trashed'"
            )["n"]
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
                "episodes_archived": episodes_archived,
                "episodes_active": episodes_active,
                "episodes_trashed": episodes_trashed,
                "episodes_today": episodes_today,
                "raw_total": raw_total,
                "raw_unprocessed": raw_unproc,
                "vec_missing": missing_vec,
                "vec_covered": vector_eligible - missing_vec,
                "embedding_provider_id": self.embedding_provider_id,
                "embedding_provider_name": provider_display_name(
                    self.embedding_provider, self.embedding_provider_id
                ),
                "embedding_dim": self.embedding_dim,
                "embedding_status": "ok",
                "last_extracted_at": last_extracted,
                "scheduler_running": self.scheduler._main_task is not None,
                "active_sessions": [_row_to_dict(r) for r in sessions_active],
            }

        try:
            return {"status": "ok", "data": await self.db.run(_q)}
        except Exception as e:
            logger.exception("[Memoir] get_stats 异常")
            return {"status": "error", "message": str(e), "data": {}}

    async def repair_vectors(self) -> dict:
        async with self._repair_lock:
            try:
                result = await self.writer.reindex_missing_vectors()
                return {"status": "ok", "data": result}
            except Exception as e:
                logger.exception("[Memoir] vector repair failed")
                return {"status": "error", "message": str(e)}

    # ---------- episodes list ----------

    async def list_episodes(self) -> dict:
        chat_type = _query("chat_type")
        participant = _query("participant")
        session_id = _query("session_id")
        group_id = _query("group_id")
        archived = _query("archived")
        status = _query("status")
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
            if archived in ("0", "1"):
                conditions.append("is_archived = ?")
                params.append(int(archived))
            if status in ("active", "archived", "trashed"):
                conditions.append("status = ?")
                params.append(status)

            if search:
                sql = (
                    "SELECT e.* FROM episodes_fts "
                    "JOIN episodes e ON e.id = episodes_fts.rowid "
                    "WHERE episodes_fts MATCH ?"
                )
                params_final: list = [search]
                if participant:
                    sql += (" AND EXISTS (SELECT 1 FROM episode_participants p "
                            "WHERE p.episode_id = e.id AND p.speaker_id = ?)")
                    params_final.append(participant)
                if conditions:
                    sql += " AND " + " AND ".join(f"e.{c}" for c in conditions)
                    params_final.extend(params)
                sql += " ORDER BY e.event_end_at DESC LIMIT ? OFFSET ?"
                params_final.extend([limit, offset])
            elif participant:
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
            else:
                sql = "SELECT * FROM episodes"
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                sql += " ORDER BY event_end_at DESC LIMIT ? OFFSET ?"
                params_final = [*params, limit, offset]

            try:
                eps = self.db.fetchall(sql, tuple(params_final))
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if not search or not any(marker in message for marker in (
                    "fts5", "syntax error", "unterminated string", "no such column"
                )):
                    raise
                # MATCH parses user input as FTS syntax; arbitrary text may be invalid.
                # instr() treats the input literally and keeps all other filters.
                fallback_sql = (
                    "SELECT e.* FROM episodes e WHERE "
                    "(instr(e.title, ?) > 0 OR instr(e.content, ?) > 0 OR "
                    "EXISTS (SELECT 1 FROM episode_keywords k "
                    "WHERE k.episode_id = e.id AND instr(k.keyword, ?) > 0))"
                )
                fallback_params: list = [search, search, search]
                if participant:
                    fallback_sql += (" AND EXISTS (SELECT 1 FROM episode_participants p "
                                     "WHERE p.episode_id = e.id AND p.speaker_id = ?)")
                    fallback_params.append(participant)
                if conditions:
                    fallback_sql += " AND " + " AND ".join(f"e.{c}" for c in conditions)
                    fallback_params.extend(params)
                fallback_sql += " ORDER BY e.event_end_at DESC LIMIT ? OFFSET ?"
                fallback_params.extend([limit, offset])
                eps = self.db.fetchall(fallback_sql, tuple(fallback_params))
            result = []
            for e in eps:
                ep_dict = _row_to_dict(e)
                parts = self.db.fetchall(
                    "SELECT speaker_id, speaker_name, role FROM episode_participants "
                    "WHERE episode_id = ?",
                    (e["id"],),
                )
                ep_dict["participants"] = [
                    {**_row_to_dict(p), "canonical_name": self.db.identities.display(
                        p["speaker_id"], p["speaker_name"])[0]} for p in parts
                ]
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
            ep_dict["participants"] = [
                {**_row_to_dict(p), "canonical_name": self.db.identities.display(
                    p["speaker_id"], p["speaker_name"])[0]} for p in parts
            ]

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
                ep_dict["raw_evidence"] = [
                    {**_row_to_dict(r), "canonical_name": self.db.identities.display(
                        r["speaker_id"], r["speaker_name"])[0]} for r in raws
                ]
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
            ep_dict["versions"] = [_row_to_dict(v) for v in self.db.fetchall(
                "SELECT * FROM episode_versions WHERE episode_id=? ORDER BY id DESC", (eid,))]
            ep_dict["merged_from"] = [_row_to_dict(v) for v in self.db.fetchall(
                "SELECT e.* FROM episode_merge_sources m JOIN episodes e ON e.id=m.source_episode_id "
                "WHERE m.merged_episode_id=? ORDER BY e.event_start_at,e.id", (eid,))]

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
            return [
                {**_row_to_dict(r), "canonical_name": self.db.identities.display(
                    r["speaker_id"], r["speaker_name"])[0]} for r in rows
            ]

        try:
            return {"status": "ok", "data": await self.db.run(_q)}
        except Exception as e:
            logger.exception("[Memoir] list_raw 异常")
            return {"status": "error", "message": str(e), "data": []}

    # ---------- lifecycle / edit / maintenance ----------

    async def episode_edit(self, episode_id: str) -> dict:
        body = await _json_body()
        try:
            data = await self.maintenance.edit_episode(
                int(episode_id), str(body.get("title", "")), str(body.get("content", "")),
                int(body.get("importance", 0)))
            return {"status": "ok", "data": data}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def episode_action(self, episode_id: str) -> dict:
        body = await _json_body()
        try:
            action = str(body.get("action", ""))
            if action == "undo_edit":
                data = await self.maintenance.undo_latest_edit(int(episode_id))
                return {"status": "ok", "data": data}
            if action == "permanent_delete":
                await self.db.run(self.maintenance.permanent_delete, int(episode_id))
                return {"status": "ok", "data": {"deleted": True}}
            data = await self.db.run(self.maintenance.manual_state, int(episode_id), action,
                                     body.get("reason"))
            return {"status": "ok", "data": data}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def maintenance_preview(self) -> dict:
        body = await _json_body()
        try:
            start = datetime.fromisoformat(str(body.get("start_date"))).date()
            end = datetime.fromisoformat(str(body.get("end_date"))).date() + timedelta(days=1)
            if self._maintenance_lock.locked():
                return {"status": "error", "message": "已有历史整理正在生成 Preview"}
            await self._maintenance_lock.acquire()
            start_ts = int(datetime.combine(start, datetime.min.time()).timestamp())
            end_ts = int(datetime.combine(end, datetime.min.time()).timestamp())
            async def _background():
                try:
                    await self.maintenance.preview(start_ts, end_ts, run_type="history")
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("[Memoir] background maintenance preview failed")
                finally:
                    self._maintenance_lock.release()
            task = asyncio.create_task(_background(), name="memoir-history-preview")
            self._maintenance_tasks.add(task)
            task.add_done_callback(self._maintenance_tasks.discard)
            return {"status": "ok", "data": {"started": True}}
        except Exception as e:
            logger.exception("[Memoir] maintenance preview failed")
            return {"status": "error", "message": str(e)}

    async def stop_background(self):
        for task in list(self._maintenance_tasks):
            task.cancel()
        if self._maintenance_tasks:
            await asyncio.gather(*self._maintenance_tasks, return_exceptions=True)
        self._maintenance_tasks.clear()

    async def maintenance_runs(self) -> dict:
        try: return {"status": "ok", "data": await self.db.run(self.maintenance.list_runs)}
        except Exception as e: return {"status": "error", "message": str(e)}

    async def maintenance_detail(self, run_id: str) -> dict:
        try: return {"status": "ok", "data": await self.db.run(self.maintenance.run_detail, int(run_id))}
        except Exception as e: return {"status": "error", "message": str(e)}

    async def maintenance_apply(self, run_id: str) -> dict:
        try: return {"status": "ok", "data": await self.maintenance.apply(int(run_id))}
        except Exception as e: return {"status": "error", "message": str(e)}

    async def maintenance_undo(self, run_id: str) -> dict:
        try: return {"status": "ok", "data": await self.maintenance.undo(int(run_id))}
        except Exception as e: return {"status": "error", "message": str(e)}

    async def maintenance_delete(self, run_id: str) -> dict:
        try:
            data = await self.db.run(self.maintenance.delete_run_record, int(run_id))
            return {"status": "ok", "data": data}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ---------- debug recall ----------

    async def debug_recall(self) -> dict:
        body = await _json_body()
        query = str(body.get("query", "")).strip()
        chat_type = body.get("chat_type", "private")
        session_id = body.get("session_id", "debug_session")
        group_id = body.get("group_id") or None
        current_speaker_id = str(body.get("current_speaker_id", "") or "").strip()

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

        visibility = self.retriever.visibility_info(
            chat_type, session_id, group_id, current_speaker_id,
        )
        return {
            "status": "ok",
            "data": {
                "query": query,
                "scope": {
                    "chat_type": chat_type,
                    "session_id": session_id,
                    "group_id": group_id,
                    "current_speaker_id": current_speaker_id,
                    "resolved_owner_id": visibility["resolved_owner_id"],
                    "is_owner": visibility["is_owner"],
                    "visibility_scope": visibility["visibility_scope"],
                },
                "results": [
                    {
                        "episode_id": r.episode_id,
                        "title": r.title,
                        "content": r.content,
                        "chat_type": r.chat_type,
                        "session_id": r.session_id,
                        "group_id": r.group_id,
                        "source_partner": r.source_partner,
                        "event_start_at": r.event_start_at,
                        "participants": r.participants,
                        "vec_distance": r.debug_vec_distance,
                        "fts_hit": r.debug_fts_hit,
                        "rrf_score": r.debug_rrf_score,
                        "participant_bonus": r.debug_participant_bonus,
                        "source_bonus": r.debug_source_bonus,
                        "recency_bonus": r.debug_recency_bonus,
                        "final_score": r.debug_final_score,
                        "decay_factor": r.debug_decay_factor,
                        "structured": r.debug_structured,
                        "relevance_passed": r.debug_relevance_passed,
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

    async def h_repair_vectors():
        return await panel.repair_vectors()

    async def h_list_identities():
        return await panel.list_identities()

    async def h_save_identities():
        return await panel.save_identities()

    async def h_merge_identities():
        return await panel.merge_identities()

    async def h_episode_edit(episode_id: str): return await panel.episode_edit(episode_id)
    async def h_episode_action(episode_id: str): return await panel.episode_action(episode_id)
    async def h_maintenance_preview(): return await panel.maintenance_preview()
    async def h_maintenance_runs(): return await panel.maintenance_runs()
    async def h_maintenance_detail(run_id: str): return await panel.maintenance_detail(run_id)
    async def h_maintenance_apply(run_id: str): return await panel.maintenance_apply(run_id)
    async def h_maintenance_undo(run_id: str): return await panel.maintenance_undo(run_id)
    async def h_maintenance_delete(run_id: str): return await panel.maintenance_delete(run_id)

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

    context.register_web_api(
        f"{_ROUTE_PREFIX}/vectors/repair", h_repair_vectors, methods=["POST"],
        desc="Memoir repair missing vectors",
    )

    context.register_web_api(
        f"{_ROUTE_PREFIX}/identities", h_list_identities, methods=["GET"],
        desc="Memoir QQ identity cards",
    )
    context.register_web_api(
        f"{_ROUTE_PREFIX}/identities/batch", h_save_identities, methods=["POST"],
        desc="Memoir edit identity cards",
    )
    context.register_web_api(
        f"{_ROUTE_PREFIX}/identities/merge", h_merge_identities, methods=["POST"],
        desc="Memoir merge two QQ identities",
    )

    context.register_web_api(f"{_ROUTE_PREFIX}/episodes/<episode_id>/edit", h_episode_edit,
                             methods=["POST"], desc="Edit episode with version history")
    context.register_web_api(f"{_ROUTE_PREFIX}/episodes/<episode_id>/action", h_episode_action,
                             methods=["POST"], desc="Archive/restore/trash/permanently delete episode")
    context.register_web_api(f"{_ROUTE_PREFIX}/maintenance/preview", h_maintenance_preview,
                             methods=["POST"], desc="Backup and preview historical maintenance")
    context.register_web_api(f"{_ROUTE_PREFIX}/maintenance/runs", h_maintenance_runs,
                             methods=["GET"], desc="Maintenance run history")
    context.register_web_api(f"{_ROUTE_PREFIX}/maintenance/runs/<run_id>", h_maintenance_detail,
                             methods=["GET"], desc="Maintenance run detail")
    context.register_web_api(f"{_ROUTE_PREFIX}/maintenance/runs/<run_id>/apply", h_maintenance_apply,
                             methods=["POST"], desc="Apply preview")
    context.register_web_api(f"{_ROUTE_PREFIX}/maintenance/runs/<run_id>/undo", h_maintenance_undo,
                             methods=["POST"], desc="Undo applied maintenance")
    context.register_web_api(f"{_ROUTE_PREFIX}/maintenance/runs/<run_id>/delete", h_maintenance_delete,
                             methods=["POST"], desc="Discard preview or delete failed run record")

    logger.info("[Memoir] panel API endpoints registered (17 routes)")
