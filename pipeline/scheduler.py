"""
pipeline/scheduler.py — 后台调度器 + process_batch 主流程

改动（Phase 1f-fix）:

- bounded batch: threshold 和 idle 都按 N 个 inbound 分块连续消化；
  idle 私聊允许未闭合的 user
- extractor 失败（success=False）→ raw 保持 unprocessed，下次重试
- extractor 成功但 events=[] → 在同一逻辑事务里标 processed（防死循环）
- extractor 成功有 events → 走 writer.write_batch，写库+processed 一次原子完成
- 后台 task 全部追踪，terminate 时收干净
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from astrbot.api import logger

from ..storage import MemoirDB
from .extractor import EventExtractor
from .writer import EpisodeWriter, WriteBatchError


@dataclass
class SchedulerConfig:
    interval_seconds: int = 60
    private_batch_user_turns: int = 10
    private_idle_minutes: int = 30
    group_batch_inbound: int = 20
    group_idle_minutes: int = 15
    overlap_group_messages: int = 4
    overlap_private_messages: int = 2
    raw_retention_days: int = 30
    nightly_maintenance_enabled: bool = True
    nightly_maintenance_hour: int = 5


class BatchScheduler:
    def __init__(
        self,
        db: MemoirDB,
        extractor: EventExtractor,
        writer: EpisodeWriter,
        config: SchedulerConfig,
        maintenance=None,
    ):
        self.db = db
        self.extractor = extractor
        self.writer = writer
        self.config = config
        self.maintenance = maintenance

        self._main_task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()
        self._session_locks: dict[str, asyncio.Lock] = {}
        # 追踪所有后台 task，terminate 时收干净
        self._background_tasks: set[asyncio.Task] = set()
        self._last_ttl_at: float = 0.0
        self._last_nightly_check: float = 0.0

    # ---------- 生命周期 ----------

    def start(self) -> None:
        if self._main_task is not None:
            return
        self._stop_evt.clear()
        self._main_task = asyncio.create_task(self._run(), name="memoir-scheduler")
        logger.info("[Memoir] scheduler 已启动 (interval=%ds)", self.config.interval_seconds)

    async def stop(self) -> None:
        self._stop_evt.set()
        if self._main_task is not None:
            try:
                await asyncio.wait_for(self._main_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._main_task.cancel()
                try:
                    await self._main_task
                except (asyncio.CancelledError, Exception):
                    pass
            finally:
                self._main_task = None

        # 等所有后台 batch task 收尾
        if self._background_tasks:
            logger.info(
                "[Memoir] scheduler stop: waiting %d background task(s) to finish",
                len(self._background_tasks),
            )
            done, pending = await asyncio.wait(
                self._background_tasks, timeout=10.0,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            self._background_tasks.clear()

        logger.info("[Memoir] scheduler 已停止")

    def _spawn(self, coro, name: str) -> None:
        """追踪 background task。"""
        t = asyncio.create_task(coro, name=name)
        self._background_tasks.add(t)
        t.add_done_callback(self._background_tasks.discard)

    # ---------- 主循环 ----------

    async def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("[Memoir] scheduler tick 异常，继续下一轮")

            try:
                await asyncio.wait_for(
                    self._stop_evt.wait(), timeout=self.config.interval_seconds,
                )
                return
            except asyncio.TimeoutError:
                pass

    async def _tick(self) -> None:
        def _find_active():
            return self.db.get_active_sessions_with_unprocessed()

        active = await self.db.run(_find_active)
        now = int(time.time())

        for row in active:
            session_id = row["session_id"]
            chat_type = row["chat_type"]
            last_at = int(row["last_unprocessed_at"])
            user_turn_count = int(row["user_turn_count"] or 0)
            completed_user_turns = int(row["completed_user_turns"] or 0)
            group_inbound_count = int(row["group_inbound_count"] or 0)

            triggered, mode, reason = self._should_trigger(
                chat_type, now, last_at,
                user_turn_count=user_turn_count,
                completed_user_turns=completed_user_turns,
                group_inbound_count=group_inbound_count,
            )
            if triggered:
                logger.info(
                    "[Memoir] trigger process_batch: session=%s chat_type=%s mode=%s reason=%s",
                    session_id, chat_type, mode, reason,
                )
                self._spawn(
                    self.process_batch(session_id, mode=mode),
                    name=f"memoir-batch-{session_id[:16]}",
                )

        if now - self._last_ttl_at > 86400:
            self._last_ttl_at = now
            self._spawn(self._ttl_cleanup(), name="memoir-ttl")
        if (self.maintenance and self.config.nightly_maintenance_enabled
                and now - self._last_nightly_check > 3600
                and datetime.now().hour >= self.config.nightly_maintenance_hour):
            self._last_nightly_check = now
            self._spawn(self._nightly_maintenance(), name="memoir-nightly")

    def _should_trigger(
        self,
        chat_type: str,
        now: int,
        last_at: int,
        *,
        user_turn_count: int,
        completed_user_turns: int,
        group_inbound_count: int,
    ) -> tuple[bool, str, str]:
        """
        判断是否触发。返回 (触发?, mode, 原因)。
        mode: "threshold" | "idle"
        """
        idle_seconds = now - last_at

        if chat_type == "private":
            if completed_user_turns >= self.config.private_batch_user_turns:
                return True, "threshold", (
                    f"private completed_turns >= {self.config.private_batch_user_turns}"
                )
            if idle_seconds >= self.config.private_idle_minutes * 60:
                return True, "idle", (
                    f"private idle {idle_seconds}s "
                    f"(user_turns={user_turn_count}, completed={completed_user_turns})"
                )
        else:  # group
            if group_inbound_count >= self.config.group_batch_inbound:
                return True, "threshold", (
                    f"group inbound >= {self.config.group_batch_inbound}"
                )
            if idle_seconds >= self.config.group_idle_minutes * 60:
                return True, "idle", f"group idle {idle_seconds}s"
        return False, "", ""

    # ---------- process_batch ----------

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    async def process_batch(self, session_id: str, *, mode: str = "idle") -> None:
        """对指定 session 跑一次事件消化。"""
        lock = self._get_lock(session_id)
        if lock.locked():
            logger.debug("[Memoir] session %s 已在处理中，跳过", session_id)
            return

        async with lock:
            try:
                while await self._process_batch_locked(session_id, mode):
                    await asyncio.sleep(0)
            except Exception:
                logger.exception(
                    "[Memoir] process_batch 异常，session=%s，raw 保持 unprocessed",
                    session_id,
                )

    async def _process_batch_locked(self, session_id: str, mode: str) -> bool:
        # 1. 拉本批 new_msgs（threshold / idle 都 bounded）
        def _load_batch():
            # 需要先知道 chat_type 才能算 batch_size 对应字段
            # 先看看 session 的类型
            rows = self.db.get_unprocessed_by_session(session_id, limit=1)
            if not rows:
                return [], "private"
            ct = rows[0]["chat_type"]
            # bounded batch
            bs = (
                self.config.private_batch_user_turns if ct == "private"
                else self.config.group_batch_inbound
            )
            return self.db.get_bounded_batch(
                session_id, ct, mode=mode, batch_size=bs,
            ), ct

        new_msgs, chat_type = await self.db.run(_load_batch)
        if not new_msgs:
            return False
        if mode == "threshold":
            inbound_count = sum(m["role"] == "user" for m in new_msgs)
            required = (self.config.private_batch_user_turns if chat_type == "private"
                        else self.config.group_batch_inbound)
            if inbound_count < required:
                return False

        group_id = new_msgs[0]["group_id"]
        platform = new_msgs[0]["platform"]

        # 2. overlap
        overlap_n = (
            self.config.overlap_group_messages if chat_type == "group"
            else self.config.overlap_private_messages
        )

        def _load_overlap():
            return self.db.fetchall(
                "SELECT * FROM recent_messages "
                "WHERE session_id = ? AND processed_at IS NOT NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (session_id, overlap_n),
            )
        overlap_msgs = list(reversed(await self.db.run(_load_overlap)))

        # 3. extractor
        result = await self.extractor.extract(new_msgs, overlap_msgs, chat_type)

        raw_ids = [int(m["id"]) for m in new_msgs]

        # 4. 分支处理
        if not result.success:
            logger.warning(
                "[Memoir] extract failed for session=%s: %s — raw 保持 unprocessed，下次重试",
                session_id, result.error,
            )
            return False  # 不 mark_processed

        if not result.events:
            # 合法的空结果 → 标 processed（防死循环）
            logger.info(
                "[Memoir] batch done: session=%s new=%d events=0 (nothing worth remembering)",
                session_id, len(new_msgs),
            )
            processed_at = int(time.time())
            await self.db.run(
                lambda: self.db.mark_processed(raw_ids, processed_at)
            )
            return True

        # 5. 有事件 → writer 原子写入
        try:
            written = await self.writer.write_batch(
                result.events,
                raw_ids_to_mark=raw_ids,
                session_id=session_id,
                chat_type=chat_type,
                group_id=group_id,
                platform=platform,
            )
            logger.info(
                "[Memoir] batch done: session=%s new=%d events=%d written=%d (rejected=%d)",
                session_id, len(new_msgs), len(result.events), len(written),
                result.rejected_count,
            )
            return True
        except WriteBatchError as e:
            logger.error(
                "[Memoir] write_batch failed: %s — raw 保持 unprocessed", e,
            )
            # 不 mark_processed
        except Exception:
            logger.exception(
                "[Memoir] write_batch 异常 — raw 保持 unprocessed"
            )
        return False

    # ---------- TTL ----------

    async def _ttl_cleanup(self) -> None:
        cutoff = int(time.time()) - self.config.raw_retention_days * 86400

        def _del():
            return self.db.delete_expired_raw(cutoff)

        n = await self.db.run(_del)
        archived = await self.db.run(lambda: self.db.archive_expired_short_term(int(time.time())))
        if archived:
            logger.info("[Memoir] archived %d expired importance=2 episode(s)", archived)
        if n > 0:
            logger.info(
                "[Memoir] TTL: 清理过期原文 %d 条 (retention=%d days)",
                n, self.config.raw_retention_days,
            )

    async def _nightly_maintenance(self) -> None:
        """仅补跑启用日期之后、今天之前尚未处理的日期。历史库需面板手动 Preview。"""
        today = datetime.now().date()
        def _start_date():
            raw = self.db.get_meta("nightly_maintenance_started_date")
            if not raw:
                raw = today.isoformat()
                self.db.set_meta("nightly_maintenance_started_date", raw)
            return raw
        start = datetime.fromisoformat(await self.db.run(_start_date)).date()
        day = start
        while day < today:
            date_key = day.isoformat()
            state = await self.db.run(lambda d=date_key: self.db.fetchone(
                "SELECT status FROM maintenance_days WHERE target_date=?", (d,)))
            if not state or state["status"] == "failed":
                begin = int(datetime.combine(day, datetime.min.time()).timestamp())
                end = int(datetime.combine(day + timedelta(days=1), datetime.min.time()).timestamp())
                try:
                    preview = await self.maintenance.preview(
                        begin, end, run_type="nightly", target_date=date_key)
                    # 夜间方案由同一套逻辑自动应用；仍完整记录且可一键撤销。
                    await self.maintenance.apply(preview["id"])
                    logger.info("[Memoir] nightly maintenance applied for %s", date_key)
                except Exception as exc:
                    logger.exception("[Memoir] nightly maintenance failed for %s: %s", date_key, exc)
                    return
            day += timedelta(days=1)
