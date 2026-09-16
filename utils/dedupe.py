"""
utils/dedupe.py — dedupe_key 生成规则

集中管理 recent_messages.dedupe_key 的构造，保证幂等入库。

- 用户/群友消息:  user:{session_id}:{platform_message_id}
- Astra 回复:    assistant:{session_id}:{trigger_raw_id}

见 DESIGN v4 §3.1。
"""
from __future__ import annotations


def user_dedupe_key(session_id: str, platform_message_id: str) -> str:
    """用户/群友消息的 dedupe_key。"""
    return f"user:{session_id}:{platform_message_id}"


def assistant_dedupe_key(session_id: str, trigger_raw_id: int) -> str:
    """Astra 回复的 dedupe_key。trigger_raw_id 是触发它的用户消息在 recent_messages 里的自增 id。"""
    return f"assistant:{session_id}:{trigger_raw_id}"
