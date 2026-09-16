"""
pipeline/scheduler.py — 后台调度器 + process_batch 主流程

一个单独的 asyncio 长跑协程，每 N 秒扫一次"有未处理消息的活跃会话"，
判断是否达到触发条件（数量阈值 / idle timeout）。

per-session asyncio.Lock 保证同一会话同时只跑一次 process_batch，
两个触发条件（数量 vs idle）走同一入口。

process_batch 是整个 pipeline 的核心：
  拉未处理 → 拉 overlap → LLM 拆事件 → 校验 → 落库 → 标 processed

见 DESIGN §4、§5.4。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from astrbot.api import logger

from ..storage import MemoirDB
from .extractor import EventExtractor
from .writer import EpisodeWriter


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


class BatchScheduler:
    """
    后台调度器 + process_batch 主流程实现。

    - start(): 起后台协程
    - stop(): 停后台协程
    - process_batch(session_id): 手动触发（也被 tick 调用）
    """

    def __init__(
        self,
        db: MemoirDB,
        extractor: EventExtractor,
        writer: EpisodeWriter,
        config: SchedulerConfig,
    ):
        self.db = db
        self.extractor = extractor
        self.writer = writer
        self.config = config

        self._task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()
        # per-session lock，防止同一会话被同时触发两次
        self._session_locks: dict[str, asyncio.Lock] = {}
        # TTL 上次跑时间
        self._last_ttl_at: float = 0.0

    # ---------- 生命周期 ----------

    def start(self) -> None:
        if self._task is not None:
            return
        self._stop_evt.clear()
        self._task = asyncio.create_task(self._run(), name="memoir-scheduler")
        logger.info("[Memoir] scheduler 已启动 (interval=%ds)", self.config.interval_seconds)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_evt.set()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except asyncio.TimeoutError:
            self._task.cancel()
        finally:
            self._task = None
            logger.info("[Memoir] scheduler 已停止")

    # ---------- 主循环 ----------

    async def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("[Memoir] scheduler tick 异常，继续下一轮")

            # 可被 stop 中断的 sleep
            try:
                await asyncio.wait_for(
                    self._stop_evt.wait(),
                    timeout=self.config.interval_seconds,
                )
                return  # stop 被 set 了
            except asyncio.TimeoutError:
                pass

    async def _tick(self) -> None:
        """
        每 interval 秒的动作：
        1. 找所有有未处理消息的会话
        2. 判断是否达到触发条件，达到就调 process_batch
        3. 每天跑一次 TTL 清理
        """
        # 1. 找活跃会话
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

            triggered, reason = self._should_trigger(
                chat_type, now, last_at,
                user_turn_count=user_turn_count,
                completed_user_turns=completed_user_turns,
                group_inbound_count=group_inbound_count,
            )
            if triggered:
                logger.info(
                    "[Memoir] trigger process_batch: session=%s chat_type=%s reason=%s",
                    session_id, chat_type, reason,
                )
                # 不 await，让不同 session 并行；per-session lock 由 process_batch 内部处理
                asyncio.create_task(
                    self.process_batch(session_id),
                    name=f"memoir-batch-{session_id[:16]}",
                )

        # 2. TTL: 每 24h 跑一次
        if now - self._last_ttl_at > 86400:
            self._last_ttl_at = now
            asyncio.create_task(self._ttl_cleanup(), name="memoir-ttl")

    def _should_trigger(
        self,
        chat_type: str,
        now: int,
        last_at: int,
        *,
        user_turn_count: int,
        completed_user_turns: int,
        group_inbound_count: int,
    ) -> tuple[bool, str]:
        """
        判断是否达到触发条件。返回 (触发?, 原因)。

        私聊 race 处理:
          - 数量阈值检查用 completed_user_turns（已被 Astra 回复的 user 数）
            避免"第 10 条 user 刚到，Astra 还在生成回复" 时把它纳入 batch，
            导致 assistant reply 后落成孤 raw。
          - idle 阈值检查用 user_turn_count（含未闭合的）
            长时间没等到回复（异常/被拦）也允许消化，避免永远卡住。
        """
        idle_seconds = now - last_at

        if chat_type == "private":
            if completed_user_turns >= self.config.private_batch_user_turns:
                return True, (
                    f"private completed_turns >= {self.config.private_batch_user_turns}"
                )
            if idle_seconds >= self.config.private_idle_minutes * 60:
                return True, (
                    f"private idle {idle_seconds}s "
                    f"(user_turns={user_turn_count}, completed={completed_user_turns})"
                )
        else:  # group
            if group_inbound_count >= self.config.group_batch_inbound:
                return True, f"group inbound >= {self.config.group_batch_inbound}"
            if idle_seconds >= self.config.group_idle_minutes * 60:
                return True, f"group idle {idle_seconds}s"
        return False, ""

    # ---------- process_batch 主流程 ----------

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    async def process_batch(self, session_id: str) -> None:
        """
        对指定 session 跑一次事件消化。

        流程:
        1. 取 per-session lock（拿不到直接返回，等下一轮）
        2. 拉未处理 raw
        3. 拉 overlap
        4. LLM 拆事件（含硬规则校验）
        5. 全部 events 落库（含 events=[] 场景）
        6. 全部成功后标 processed
           任一失败 → 保持 unprocessed，下次重试
        """
        lock = self._get_lock(session_id)
        if lock.locked():
            logger.debug("[Memoir] session %s 已在处理中，跳过本次触发", session_id)
            return

        async with lock:
            try:
                await self._process_batch_locked(session_id)
            except Exception:
                logger.exception(
                    "[Memoir] process_batch 异常，session=%s，raw 保持 unprocessed", session_id,
                )

    async def _process_batch_locked(self, session_id: str) -> None:
        # 1. 拉未处理 raw
        def _load_new():
            return self.db.get_unprocessed_by_session(session_id)

        new_msgs = await self.db.run(_load_new)
        if not new_msgs:
            return

        chat_type = new_msgs[0]["chat_type"]
        group_id = new_msgs[0]["group_id"]
        platform = new_msgs[0]["platform"]

        # 2. 拉 overlap（上一批已 processed 的最后 N 条）
        overlap_n = (
            self.config.overlap_group_messages if chat_type == "group"
            else self.config.overlap_private_messages
        )
        overlap_msgs = await self.db.run(
            lambda: self.db.fetchall(
                "SELECT * FROM recent_messages "
                "WHERE session_id = ? AND processed_at IS NOT NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (session_id, overlap_n),
            )
        )
        # 反转成时间升序
        overlap_msgs = list(reversed(overlap_msgs))

        # 3. LLM 拆事件
        events = await self.extractor.extract(new_msgs, overlap_msgs, chat_type)

        # 4. 落库
        ok, fail = await self.writer.write_events(
            events,
            session_id=session_id,
            chat_type=chat_type,
            group_id=group_id,
            platform=platform,
        )

        # 5. 标 processed
        #    只要 extractor 成功返回（可能是 []），本批就算处理过了。
        #    单条 event 写入失败不影响标 processed —— 那部分内容我们已经无法恢复了，
        #    强行重试反而会 duplicate。这是权衡：宁可丢一条也不重复。
        raw_ids = [int(m["id"]) for m in new_msgs]
        processed_at = int(time.time())
        await self.db.run(
            lambda: self.db.mark_processed(raw_ids, processed_at)
        )

        logger.info(
            "[Memoir] batch done: session=%s new=%d events_ok=%d events_fail=%d",
            session_id, len(new_msgs), ok, fail,
        )

    # ---------- TTL ----------

    async def _ttl_cleanup(self) -> None:
        cutoff = int(time.time()) - self.config.raw_retention_days * 86400

        def _del():
            return self.db.delete_expired_raw(cutoff)

        n = await self.db.run(_del)
        if n > 0:
            logger.info("[Memoir] TTL: 清理过期原文 %d 条 (retention=%d days)",
                        n, self.config.raw_retention_days)
