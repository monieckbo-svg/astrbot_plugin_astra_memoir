"""
astrbot_plugin_astra_memoir — Astra 的自动记忆库

完整装配：消息缓存、分块事件提取、episode/FTS/向量存储、检索注入，
以及管理面板和缺失向量修复。
"""
from __future__ import annotations

import asyncio
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
)
from .pipeline.embedding import resolve_embedding_provider, embedding_dimension
from .pipeline.panel import MemoirPanel, register_panel_routes


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
            owner_qq_id=str(self._cfg("owner_qq_id", "") or "").strip(),
        )
        logger.info("[Memoir] private→group recall: enabled=%s, owner_qq_id=%r",
                    retr_cfg.enable_group_recall_in_private, retr_cfg.owner_qq_id)

        # ---- 异步初始化：先确定 provider/dim，才能创建 vec 表 ----
        data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        db_path = data_dir / "memoir.db"
        logger.info("[Memoir] DB path: %s", db_path)
        self.db = None
        self.scheduler = None
        self.startup_error = None
        self._startup_task = asyncio.create_task(self._initialize(
            context, db_path, embedding_provider_id, extract_provider_id,
            bot_display_name, sched_cfg, retr_cfg,
        ))

    async def _initialize(self, context, db_path, embedding_provider_id,
                          extract_provider_id, bot_display_name, sched_cfg, retr_cfg):
        try:
            provider = resolve_embedding_provider(context, embedding_provider_id)
            dim = await embedding_dimension(provider)
            # Probe even when get_dim() succeeds: fail early on broken credentials.
            probe = await provider.get_embedding("memoir startup probe")
            if len(probe) != dim:
                raise RuntimeError(f"Embedding provider 维度不一致: get_dim={dim}, probe={len(probe)}")
            self.db = MemoirDB(db_path, embedding_dim=dim,
                               embedding_provider_id=embedding_provider_id)
            self.db.initialize()
            self.vec = VecStore(self.db)
            self.raw_cache = RawCache(self.db, bot_display_name=bot_display_name)
            self.extractor = EventExtractor(context, self.db, extract_provider_id)
            self.writer = EpisodeWriter(provider, self.db, self.vec)
            self.scheduler = BatchScheduler(self.db, self.extractor, self.writer, sched_cfg)
            self.retriever = Retriever(provider, self.db, self.vec, retr_cfg)
            self.panel = MemoirPanel(self.db, self.vec, self.retriever, self.scheduler,
                                     self.writer, provider, embedding_provider_id, dim)
            register_panel_routes(context, self.panel)
            repaired = await self.writer.reindex_missing_vectors()
            logger.info("[Memoir] startup vector repair: %s", repaired)
            self.scheduler.start()
            logger.info("[Memoir] 插件加载完成 (embedding_provider=%s, dim=%d)",
                        embedding_provider_id, dim)
        except Exception as e:
            self.startup_error = str(e)
            logger.exception("[Memoir] 初始化失败: %s", e)
            if self.scheduler:
                await self.scheduler.stop()
            # Configuration errors stay visible in the Plugin Page status tab.
            if not hasattr(self, "panel"):
                async def h_status_error():
                    return {"status": "error", "message": self.startup_error}
                try:
                    context.register_web_api(
                        f"/{PLUGIN_NAME}/stats", h_status_error, methods=["GET"],
                        desc="Memoir startup error",
                    )
                except Exception:
                    logger.exception("[Memoir] 无法注册启动错误状态端点")

    async def _ready(self) -> bool:
        await self._startup_task
        return self.startup_error is None

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
        if await self._ready():
            await self.raw_cache.record_user_message(event)

    @filter.on_llm_response()
    async def on_astra_reply(self, event: AstrMessageEvent, resp: LLMResponse):
        """
        Astra 最终回复入 raw cache。
        speaker_id = event.get_self_id()（bot 自己的 QQ 号）
        trigger_raw_id = 反查触发本次 LLM 的用户 raw
        反查不到 → warning + skip
        """
        if await self._ready():
            await self.raw_cache.record_assistant_reply(event, resp)

    @filter.on_llm_request()
    async def on_before_llm(self, event: AstrMessageEvent, req):
        """
        LLM 请求前检索相关记忆并注入。
        用 req.extra_user_content_parts + mark_as_temp（v4 官方姿势），
        不动 system_prompt / prompt / contexts，保护 prompt cache。
        """
        if await self._ready():
            await self.retriever.inject(event, req)

    # ==========================================================
    # Lifecycle
    # ==========================================================

    async def terminate(self):
        """插件卸载：停调度，关 DB。"""
        await self._startup_task
        try:
            if self.scheduler:
                await self.scheduler.stop()
        except Exception:
            logger.exception("[Memoir] scheduler.stop 异常")
        try:
            if self.db:
                self.db.close()
        except Exception:
            logger.exception("[Memoir] db.close 异常")
        logger.info("[Memoir] 插件已卸载")
