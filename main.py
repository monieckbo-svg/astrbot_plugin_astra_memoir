"""
astrbot_plugin_astra_memoir — Astra 的自动记忆库

完整装配：消息缓存、分块事件提取、episode/FTS/向量存储、检索注入，
以及管理面板和缺失向量修复。
"""
from __future__ import annotations

import asyncio
import time
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
    MaintenanceManager,
)
from .pipeline.embedding import resolve_embedding_provider, embedding_dimension
from .pipeline.panel import MemoirPanel, register_panel_routes
from .pipeline.health import HealthState
from .pipeline.embedding_runtime import EmbeddingRuntime


PLUGIN_NAME = "astrbot_plugin_astra_memoir"
RECOVERY_INTERVAL_SECONDS = 30


@register(
    PLUGIN_NAME,
    "Astra & Celii",
    "星星的自动记忆库 - 私聊/群聊事件消化",
    "0.2.0",
    "https://github.com/monieckbo-svg/astrbot_plugin_astra_memoir",
)
class AstraMemoir(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.context = context
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
            nightly_maintenance_enabled=bool(self._cfg("nightly_maintenance_enabled", True)),
            nightly_maintenance_hour=max(0, min(int(self._cfg("nightly_maintenance_hour", 5)), 23)),
        )

        retr_cfg = RetrieverConfig(
            top_k=max(1, min(int(self._cfg("private_recall_top_k", 3)), 5)),
            private_recall_max=max(1, min(int(self._cfg("private_recall_max", 5)), 5)),
            group_group_quota=max(0, min(int(self._cfg("group_group_quota", 3)), 5)),
            group_private_quota=max(0, min(int(self._cfg("group_private_quota", 1)), 5)),
            group_total_max=max(1, min(int(self._cfg("group_total_max", 4)), 5)),
            max_cosine_distance=float(self._cfg("retrieval_max_cosine_distance", 0.9)),
            owner_qq_id=str(self._cfg("owner_qq_id", "") or "").strip(),
        )
        logger.info("[Memoir] unified recall enabled (owner_qq_id=%r is identity only)",
                    retr_cfg.owner_qq_id)

        # ---- 异步初始化：先确定 provider/dim，才能创建 vec 表 ----
        data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        db_path = data_dir / "memoir.db"
        logger.info("[Memoir] DB path: %s", db_path)
        self.db = None
        self.scheduler = None
        self.startup_error = None
        self.health = HealthState(data_dir / 'memoir-health.json')
        self._supervisor_task = None
        self._watchdog_task = None
        self._stopping = False
        self._startup_task = asyncio.create_task(self._initialize(
            context, db_path, embedding_provider_id, extract_provider_id,
            bot_display_name, sched_cfg, retr_cfg,
        ))

    async def _initialize(self, context, db_path, embedding_provider_id,
                          extract_provider_id, bot_display_name, sched_cfg, retr_cfg):
        try:
            self.db = MemoirDB(db_path, embedding_dim=0,
                               embedding_provider_id=embedding_provider_id)
            self.db.initialize(defer_vectors=True)
            self.vec = VecStore(self.db)
            self.raw_cache = RawCache(self.db, bot_display_name=bot_display_name, health=self.health)
            provider = EmbeddingRuntime(context, embedding_provider_id, self.db, self.health)
            self.embedding_runtime = provider
            self.extractor = EventExtractor(context, self.db, extract_provider_id)
            self.writer = EpisodeWriter(provider, self.db, self.vec)
            self.maintenance = MaintenanceManager(
                self.db, self.vec, self.writer, self.extractor,
                batch_size=int(self._cfg("maintenance_batch_size", 12)),
            )
            self.scheduler = BatchScheduler(
                self.db, self.extractor, self.writer, sched_cfg, self.maintenance)
            self.retriever = Retriever(provider, self.db, self.vec, retr_cfg)
            self.panel = MemoirPanel(self.db, self.vec, self.retriever, self.scheduler,
                                     self.writer, provider, embedding_provider_id, self.db.embedding_dim,
                                     self.maintenance)
            self.panel.health = self.health
            self.panel.context = context
            register_panel_routes(context, self.panel)
            self.scheduler.start()
            self._supervisor_task = asyncio.create_task(self._supervise(), name='memoir-recovery')
            self._watchdog_task = asyncio.create_task(self._watchdog(), name='memoir-watchdog')
            logger.info('[Memoir] raw采集和提取已启动；embedding在后台恢复')
        except Exception as e:
            self.startup_error = str(e)
            self.health.error('startup', e)
            logger.exception("[Memoir] 初始化失败: %s", e)
            if self.scheduler:
                await self.scheduler.stop()
            # Configuration errors stay visible in the Plugin Page status tab.
            if not hasattr(self, "panel"):
                async def h_status_error():
                    return {"status": "ok", "data": {'health': self.health.watchdog(),
                            'raw_ingestion_status': 'error', 'extractor_status': 'stopped',
                            'embedding_status': 'degraded', 'vector_index_status': 'unknown',
                            'scheduler_running': False}}
                try:
                    context.register_web_api(
                        f"/{PLUGIN_NAME}/stats", h_status_error, methods=["GET"],
                        desc="Memoir startup error",
                    )
                except Exception:
                    logger.exception("[Memoir] 无法注册启动错误状态端点")

    async def _ready(self) -> bool:
        await asyncio.shield(self._startup_task)
        return not self._stopping and hasattr(self, 'raw_cache')

    async def _supervise(self):
        while not self._stopping:
            try:
                self.health.watchdog()
                runtime = self.embedding_runtime
                try:
                    selected = resolve_embedding_provider(self.context, runtime.id)
                except Exception:
                    selected = None
                if runtime.provider is None or selected is not runtime.provider:
                    await runtime.recover()
                if runtime.provider is not None:
                    result = await self.writer.reindex_missing_vectors(max_episodes=20)
                    self.health.update(last_vector_repair=result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.health.error('recovery', exc)
                logger.exception('[Memoir] recovery/watchdog失败，下轮重试')
            await asyncio.sleep(RECOVERY_INTERVAL_SECONDS)

    async def _watchdog(self):
        while not self._stopping:
            self.health.watchdog()
            await asyncio.sleep(30)

    def _cfg(self, key: str, default):
        """兼容 dict / AstrBotConfig 两种配置形态。"""
        try:
            return self.config.get(key, default)
        except AttributeError:
            return getattr(self.config, key, default)

    # ==========================================================
    # Hooks
    # ==========================================================

    @filter.event_message_type(EventMessageType.ALL, priority=1000)
    async def on_user_message(self, event: AstrMessageEvent):
        """
        所有用户消息入 raw cache（包括群里非 @Astra 的普通消息）。
        speaker_id = event.get_sender_id()（QQ 号，稳定主键）
        """
        self.health.update(last_message_hook_seen_at=int(time.time()))
        try:
            if await self._ready():
                await self.raw_cache.record_user_message(event)
            elif not self._stopping:
                self.health.error('raw', self.startup_error or '原文数据库未就绪')
        except Exception as exc:
            self.health.error('raw', exc)
            logger.exception('[Memoir] user hook失败')

    @filter.on_llm_response()
    async def on_astra_reply(self, event: AstrMessageEvent, resp: LLMResponse):
        """
        Astra 最终回复入 raw cache。
        speaker_id = event.get_self_id()（bot 自己的 QQ 号）
        trigger_raw_id = 反查触发本次 LLM 的用户 raw
        反查不到 → warning + skip
        """
        if await self._ready():
            self.health.update(last_llm_activity_at=int(time.time()))
            await self.raw_cache.record_assistant_reply(event, resp)

    @filter.on_llm_request()
    async def on_before_llm(self, event: AstrMessageEvent, req):
        """
        LLM 请求前检索相关记忆并注入。
        用 req.extra_user_content_parts + mark_as_temp（v4 官方姿势），
        不动 system_prompt / prompt / contexts，保护 prompt cache。
        """
        self.health.update(last_llm_activity_at=int(time.time()))
        if await self._ready():
            await self.retriever.inject(event, req)

    # ==========================================================
    # Lifecycle
    # ==========================================================

    async def terminate(self):
        """插件卸载：停调度，关 DB。"""
        self._stopping = True
        await asyncio.shield(self._startup_task)
        if self._supervisor_task:
            self._supervisor_task.cancel()
            await asyncio.gather(self._supervisor_task, return_exceptions=True)
        if self._watchdog_task:
            self._watchdog_task.cancel()
            await asyncio.gather(self._watchdog_task, return_exceptions=True)
        try:
            if hasattr(self, "panel"):
                await self.panel.stop_background()
            if self.scheduler:
                await self.scheduler.stop()
        except Exception:
            logger.exception("[Memoir] scheduler.stop 异常")
        try:
            if self.db:
                await self.db.run(self.db.close)
        except Exception:
            logger.exception("[Memoir] db.close 异常")
        logger.info("[Memoir] 插件已卸载")
