"""
astrbot_plugin_astra_memoir — Astra 的自动记忆库

Phase 1d 完整装配：
- storage (SQLite + sqlite-vec)
- raw_cache (消息入库)
- scheduler (60s 轮询 + per-session lock)
- extractor (LLM 事件拆分)
- writer (episode 落库 + embedding)

Phase 1e 待补：retriever（检索 + on_llm_request 注入）
"""
from __future__ import annotations

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.provider.entities import LLMResponse

from .storage import MemoirDB, VecStore
from .pipeline import (
    RawCache,
    EventExtractor,
    EpisodeWriter,
    BatchScheduler,
    SchedulerConfig,
    Retriever,
    RetrieverConfig,
    MemoirPanel,
    register_panel_routes,
)


PLUGIN_NAME = "astrbot_plugin_astra_memoir"


@register(
    PLUGIN_NAME,
    "Astra & Celii",
    "星星的自动记忆库 - 私聊/群聊事件消化",
    "0.1.0",
    "https://github.com/monieckbo-svg/astrbot_plugin_astra_memoir",
)
class AstraMemoir(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config or {}

        # ---- 读配置 ----
        embedding_dim = int(self._cfg("embedding_dim", 1024))
        bot_display_name = str(self._cfg("bot_display_name", "星星"))
        extract_provider_id = str(self._cfg("extract_provider_id", ""))
        embedding_provider_id = str(self._cfg("embedding_provider_id", ""))

        sched_cfg = SchedulerConfig(
            interval_seconds=int(self._cfg("scheduler_interval_seconds", 60)),
            private_batch_user_turns=int(self._cfg("private_batch_user_turns", 10)),
            private_idle_minutes=int(self._cfg("private_idle_minutes", 30)),
            group_batch_inbound=int(self._cfg("group_batch_inbound", 20)),
            group_idle_minutes=int(self._cfg("group_idle_minutes", 15)),
            overlap_group_messages=int(self._cfg("overlap_group_messages", 4)),
            overlap_private_messages=int(self._cfg("overlap_private_messages", 2)),
            raw_retention_days=int(self._cfg("raw_retention_days", 30)),
        )

        retr_cfg = RetrieverConfig(
            top_k=int(self._cfg("retrieval_top_k", 4)),
            max_cosine_distance=float(self._cfg("retrieval_max_cosine_distance", 0.9)),
            enable_group_recall_in_private=bool(
                self._cfg("enable_group_recall_in_private", True)
            ),
        )

        # ---- 初始化 DB ----
        data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        db_path = data_dir / "memoir.db"
        logger.info("[Memoir] DB path: %s", db_path)

        self.db = MemoirDB(db_path, embedding_dim=embedding_dim)
        self.db.initialize()
        self.vec = VecStore(self.db)

        # ---- 装配 pipeline ----
        self.raw_cache = RawCache(self.db, bot_display_name=bot_display_name)
        self.extractor = EventExtractor(context, self.db, extract_provider_id)
        self.writer = EpisodeWriter(context, self.db, self.vec, embedding_provider_id)
        self.scheduler = BatchScheduler(self.db, self.extractor, self.writer, sched_cfg)
        self.retriever = Retriever(
            context, self.db, self.vec, retr_cfg, embedding_provider_id
        )

        # ---- 装配管理面板 + 注册 HTTP endpoint ----
        self.panel = MemoirPanel(self.db, self.vec, self.retriever, self.scheduler)
        try:
            register_panel_routes(context, self.panel)
        except Exception:
            logger.exception("[Memoir] 面板路由注册失败，其他功能不受影响")

        # ---- 起后台调度 ----
        self.scheduler.start()

        logger.info(
            "[Memoir] 插件加载完成 (embedding_dim=%d, extract_provider=%r, "
            "top_k=%d, max_cosine_distance=%.2f)",
            embedding_dim, extract_provider_id or "(using_provider)",
            retr_cfg.top_k, retr_cfg.max_cosine_distance,
        )

    def _cfg(self, key: str, default):
        """兼容 dict / AstrBotConfig 两种配置形态。"""
        try:
            return self.config.get(key, default)
        except AttributeError:
            return getattr(self.config, key, default)

    # ==========================================================
    # Hooks
    # ==========================================================

    @filter.event_message_type(EventMessageType.ALL)
    async def on_user_message(self, event: AstrMessageEvent):
        """
        所有用户消息入 raw cache（包括群里非 @Astra 的普通消息）。
        speaker_id = event.get_sender_id()（QQ 号，稳定主键）
        """
        await self.raw_cache.record_user_message(event)

    @filter.on_llm_response()
    async def on_astra_reply(self, event: AstrMessageEvent, resp: LLMResponse):
        """
        Astra 最终回复入 raw cache。
        speaker_id = event.get_self_id()（bot 自己的 QQ 号）
        trigger_raw_id = 反查触发本次 LLM 的用户 raw
        反查不到 → warning + skip
        """
        await self.raw_cache.record_assistant_reply(event, resp)

    @filter.on_llm_request()
    async def on_before_llm(self, event: AstrMessageEvent, req):
        """
        LLM 请求前检索相关记忆并注入。
        用 req.extra_user_content_parts + mark_as_temp（v4 官方姿势），
        不动 system_prompt / prompt / contexts，保护 prompt cache。
        """
        await self.retriever.inject(event, req)

    # ==========================================================
    # Lifecycle
    # ==========================================================

    async def terminate(self):
        """插件卸载：停调度，关 DB。"""
        try:
            await self.scheduler.stop()
        except Exception:
            logger.exception("[Memoir] scheduler.stop 异常")
        try:
            self.db.close()
        except Exception:
            logger.exception("[Memoir] db.close 异常")
        logger.info("[Memoir] 插件已卸载")
