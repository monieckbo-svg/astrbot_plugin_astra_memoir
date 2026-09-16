"""
astrbot_plugin_astra_memoir — Astra 的自动记忆库

Phase 1 骨架：只挂 hook，业务逻辑在后续 commit 里实现。
详见 DESIGN.md。
"""
from astrbot.api.star import Context, Star, register
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.core.provider.entities import LLMResponse
from astrbot.api import logger


@register(
    "astrbot_plugin_astra_memoir",
    "Astra & Celii",
    "星星的自动记忆库 - 私聊/群聊事件消化",
    "0.1.0",
    "https://github.com/monieckbo-svg/astrbot_plugin_astra_memoir",
)
class AstraMemoir(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config or {}
        logger.info("[Memoir] 插件加载中... (Phase 1 骨架)")
        # 后续 commit 会初始化：storage、scheduler、extractor、retriever

    @filter.event_message_type(EventMessageType.ALL)
    async def on_user_message(self, event: AstrMessageEvent):
        """所有用户消息入 raw cache（包括群里非@的普通消息）。"""
        # TODO: raw_cache.record_user_message(event)
        pass

    @filter.on_llm_response()
    async def on_astra_reply(self, event: AstrMessageEvent, resp: LLMResponse):
        """Astra 最终回复入 raw cache（不含工具调用中间步骤）。"""
        # TODO: raw_cache.record_assistant_reply(event, resp)
        pass

    @filter.on_llm_request()
    async def on_before_llm(self, event: AstrMessageEvent, req):
        """LLM 请求前，检索相关记忆并注入 system_prompt。"""
        # TODO: retriever.inject(event, req)
        pass

    async def terminate(self):
        """插件卸载时清理。"""
        logger.info("[Memoir] 插件已卸载")
