"""
pipeline/raw_cache.py — 消息入 recent_messages

用户消息入库（event_message_type ALL hook）：
    - 抓群聊里所有人的消息（包括未 @Astra 的）
    - speaker_id = event.get_sender_id()（QQ 号，稳定主键）

Astra 回复入库（on_llm_response hook）：
    - speaker_id = event.get_self_id()（bot 自己的 QQ 号，不用 'assistant'）
    - 反查触发它的用户 raw 拿 trigger_raw_id
    - 反查不到时 V1 warning + skip（不生成不稳定 fallback key）

见 DESIGN v4 §2.1-2.3。
"""
from __future__ import annotations

import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.core.provider.entities import LLMResponse

from ..storage import MemoirDB
from ..utils.dedupe import user_dedupe_key, assistant_dedupe_key


class RawCache:
    """
    职责：把 AstrMessageEvent / LLMResponse 转换成 recent_messages 行落库。

    所有 DB 操作通过 asyncio.to_thread 走线程池，不阻塞事件循环。
    """

    def __init__(self, db: MemoirDB, bot_display_name: str = "星星"):
        self.db = db
        self.bot_display_name = bot_display_name

    # ---------- Helpers ----------

    @staticmethod
    def _chat_type_of(event: AstrMessageEvent) -> str:
        """判断私聊 / 群聊。"""
        return "group" if event.get_group_id() else "private"

    @staticmethod
    def _platform_of(event: AstrMessageEvent) -> str:
        """v1 只支持 qq；预留 discord 等扩展位。"""
        name = (event.get_platform_name() or "").lower()
        if "cqhttp" in name or "qq" in name or "onebot" in name:
            return "qq"
        return name or "qq"

    @staticmethod
    def _get_platform_message_id(event: AstrMessageEvent) -> str | None:
        """获取平台原始消息 id。"""
        mid = getattr(event.message_obj, "message_id", None)
        return str(mid) if mid else None

    # ---------- User message ----------

    async def record_user_message(self, event: AstrMessageEvent) -> int | None:
        """
        用户/群友消息落 raw cache。
        返回：新行 id；已存在（dedupe 命中）返回 None；异常返回 None 并日志。
        """
        try:
            content = event.get_message_str()
            if not content or not content.strip():
                # 纯图片/表情/命令等 —— 也可以先记，但 V1 先跳过无字消息避免噪声
                return None

            session_id = event.unified_msg_origin
            platform_message_id = self._get_platform_message_id(event)
            if not platform_message_id:
                logger.debug("[Memoir] user msg has no platform_message_id, skip")
                return None

            speaker_id = event.get_sender_id() or ""
            speaker_name = event.get_sender_name() or speaker_id or "unknown"
            group_id = event.get_group_id() or None

            key = user_dedupe_key(session_id, platform_message_id)

            def _insert():
                return self.db.insert_raw_message(
                    dedupe_key=key,
                    platform=self._platform_of(event),
                    chat_type=self._chat_type_of(event),
                    session_id=session_id,
                    group_id=group_id,
                    platform_message_id=platform_message_id,
                    speaker_id=str(speaker_id),
                    speaker_name=speaker_name,
                    role="user",
                    content=content,
                    reply_to_id=None,     # V1 暂不解析引用回复
                    trigger_raw_id=None,
                    created_at=int(time.time()),
                )

            return await self.db.run(_insert)
        except Exception:
            logger.exception("[Memoir] record_user_message 失败")
            return None

    # ---------- Assistant reply ----------

    async def record_assistant_reply(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> int | None:
        """
        Astra 回复落 raw cache。

        dedupe_key = assistant:{session_id}:{trigger_raw_id}
        trigger_raw_id 通过反查触发本次 LLM 的用户 raw 得到。
        反查不到 → V1 warning + skip，不生成不稳定 fallback。
        """
        try:
            text = getattr(resp, "completion_text", None)
            if not text or not text.strip():
                return None

            session_id = event.unified_msg_origin
            trigger_platform_msg_id = self._get_platform_message_id(event)
            if not trigger_platform_msg_id:
                logger.warning(
                    "[Memoir] on_llm_response: event has no platform_message_id, "
                    "skip assistant raw write (session=%s)",
                    session_id,
                )
                return None

            # 反查触发本次 LLM 的用户 raw
            def _find_trigger():
                return self.db.find_raw_by_platform_msg(session_id, trigger_platform_msg_id)

            trigger_row = await self.db.run(_find_trigger)
            if trigger_row is None:
                logger.warning(
                    "[Memoir] on_llm_response: trigger user raw not found "
                    "(session=%s, msg_id=%s), skip assistant write",
                    session_id, trigger_platform_msg_id,
                )
                return None

            trigger_raw_id = int(trigger_row["id"])
            key = assistant_dedupe_key(session_id, trigger_raw_id)

            bot_qq = event.get_self_id() or "bot"
            group_id = event.get_group_id() or None

            def _insert():
                return self.db.insert_raw_message(
                    dedupe_key=key,
                    platform=self._platform_of(event),
                    chat_type=self._chat_type_of(event),
                    session_id=session_id,
                    group_id=group_id,
                    platform_message_id=None,      # 此时 Astra 还没发消息出去
                    speaker_id=str(bot_qq),
                    speaker_name=self.bot_display_name,
                    role="assistant",
                    content=text,
                    reply_to_id=trigger_platform_msg_id,
                    trigger_raw_id=trigger_raw_id,
                    created_at=int(time.time()),
                )

            return await self.db.run(_insert)
        except Exception:
            logger.exception("[Memoir] record_assistant_reply 失败")
            return None
