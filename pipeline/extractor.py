"""
pipeline/extractor.py — LLM 事件拆分

输入：本批 new_messages + overlap（context_only）
输出：list[Event]，events=[] 是合法结果

Prompt 要点（DESIGN §5）：
- 识别相互独立的事件，不同主题必须拆开
- 每个事件必须能脱离本批次独立理解
- 保留具体人/物/名称/数字/决定/结果
- 群聊 msg 必须带 [qq=xxx][name=xxx] 让模型知道 speaker，但 participants 由代码回查
- 硬规则：每个 event 必须至少引用一个 new_messages 的 raw_id
    （overlap 只帮理解，不允许单独据此造事件）
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass

from astrbot.api import logger
from astrbot.api.star import Context

from ..storage import MemoirDB


# ============================================================
# Data classes
# ============================================================

@dataclass
class ExtractedEvent:
    """LLM 拆分出的一条事件（已通过 hard rule 校验）。"""
    title: str
    content: str
    source_raw_ids: list[int]     # 引用的 recent_messages.id，至少含一个 new_messages id
    keywords: list[str]


# ============================================================
# Prompt 构造
# ============================================================

_SYSTEM_PROMPT_PRIVATE = """你是记忆系统的事件提取器，只输出严格 JSON。

你的任务不是总结整段聊天，而是识别其中相互独立、未来可能被重新提及的事件。

规则：
1. 一批可以包含 0 到 N 个事件。不同主题必须拆开；只有确实属于同一件事的发展过程才合并。
2. 不要为了减少数量把午饭、工作、宠物、画图等无关主题写进同一事件。
3. 每个事件必须能脱离本批次独立理解，保留具体人、物、名称、数字、决定和结果。
4. 不要生成"聊了生活"、"讨论了各种话题"这种空泛摘要。
5. 单纯的"哈哈哈"、表情、寒暄没有值得记忆的内容时，返回 events=[]。
6. 消息中的 [raw_id=N] 是内部标识，输出的 source_raw_ids 必须只包含 <new_messages> 标记内的 raw_id，不能只用 <context_only> 里的。
7. keywords 是 2~5 个中文关键词，用于后续检索命中。

输出格式（严格 JSON，不带 markdown fence 或额外说明）：
{"events":[
  {"title":"简短标题","content":"详细内容","source_raw_ids":[101,102],"keywords":["关键词1","关键词2"]}
]}
如没有值得记忆的事件，返回：{"events":[]}"""


_SYSTEM_PROMPT_GROUP = """你是记忆系统的事件提取器，只输出严格 JSON。

你的任务不是总结整段群聊，而是识别其中相互独立、未来可能被重新提及的事件。

规则：
1. 一批可以包含 0 到 N 个事件。不同主题必须拆开。
2. 群聊中即使 AI 助手完全没有参与，只要有值得记忆的信息（个人经历、观点、决定、事件），也应提取。
3. 每条消息前的 [qq=XXX][name=YYY] 表明发言者身份，不同人的言论必须严格区分。绝不允许把 A 说的话记成 B 说的。
4. 每个事件必须能脱离本批次独立理解，保留具体人、物、名称、数字、决定和结果。
5. 不要生成"群里聊了各种话题"这种空泛摘要。
6. 单纯的"哈哈哈"、表情、寒暄没有值得记忆的内容时，返回 events=[]。
7. 消息中的 [raw_id=N] 是内部标识，输出的 source_raw_ids 必须只包含 <new_messages> 标记内的 raw_id。
8. keywords 是 2~5 个中文关键词，用于后续检索命中。

输出格式（严格 JSON，不带 markdown fence 或额外说明）：
{"events":[
  {"title":"简短标题","content":"详细内容，含谁说了什么","source_raw_ids":[8871,8872],"keywords":["关键词1","关键词2"]}
]}
如没有值得记忆的事件，返回：{"events":[]}"""


def _format_msg_line(row: sqlite3.Row, include_speaker_meta: bool) -> str:
    """
    格式化一条消息给 LLM 看。
    - 私聊: [raw_id=N] 陆忱: 内容
    - 群聊: [raw_id=N][qq=XXX][name=YYY] 内容
    """
    rid = row["id"]
    content = row["content"].replace("\n", " ")  # 单行化，避免 prompt 结构混乱
    if include_speaker_meta:
        return f"[raw_id={rid}][qq={row['speaker_id']}][name={row['speaker_name']}] {content}"
    else:
        return f"[raw_id={rid}] {row['speaker_name']}: {content}"


def build_user_prompt(
    new_msgs: list[sqlite3.Row],
    overlap_msgs: list[sqlite3.Row],
    chat_type: str,
) -> str:
    """
    组装 user prompt。overlap 放 <context_only>，新消息放 <new_messages>。
    """
    is_group = chat_type == "group"
    lines: list[str] = []

    if overlap_msgs:
        lines.append("<context_only>")
        lines.append("(以下是上一批对话的尾部，仅帮助你理解上下文，不允许单独据此提取事件)")
        for row in overlap_msgs:
            lines.append(_format_msg_line(row, include_speaker_meta=is_group))
        lines.append("</context_only>")
        lines.append("")

    lines.append("<new_messages>")
    lines.append("(以下是本批新增消息，事件的 source_raw_ids 必须至少含其中一个 raw_id)")
    for row in new_msgs:
        lines.append(_format_msg_line(row, include_speaker_meta=is_group))
    lines.append("</new_messages>")

    return "\n".join(lines)


# ============================================================
# JSON 解析（健壮）
# ============================================================

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _strip_json_fence(text: str) -> str:
    """去掉 markdown code fence，模型偶尔会加上。"""
    return _JSON_FENCE_RE.sub("", text).strip()


def _extract_first_json_object(text: str) -> str | None:
    """
    从可能含解释性文本的字符串里抽出第一个 {} JSON 对象。
    简单大括号计数，够 v1 用。
    """
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def parse_llm_response(text: str, new_msg_ids: set[int]) -> list[ExtractedEvent]:
    """
    解析 LLM 返回的 JSON，应用硬规则校验。

    硬规则：
    - source_raw_ids 必须至少含一个 new_msg_ids 里的元素
    - source_raw_ids 里所有 id 必须在候选集合内（防止模型编 id）
    - title / content 非空
    """
    if not text:
        return []

    body = _strip_json_fence(text)
    obj_str = _extract_first_json_object(body)
    if obj_str is None:
        logger.warning("[Memoir] extractor: no JSON object in LLM response: %r", text[:200])
        return []

    try:
        obj = json.loads(obj_str)
    except json.JSONDecodeError as e:
        logger.warning("[Memoir] extractor: JSON decode fail (%s): %r", e, obj_str[:200])
        return []

    raw_events = obj.get("events", [])
    if not isinstance(raw_events, list):
        logger.warning("[Memoir] extractor: events not a list, got %r", type(raw_events))
        return []

    valid: list[ExtractedEvent] = []
    for i, ev in enumerate(raw_events):
        if not isinstance(ev, dict):
            continue

        title = str(ev.get("title", "")).strip()
        content = str(ev.get("content", "")).strip()
        raw_ids = ev.get("source_raw_ids", [])
        keywords = ev.get("keywords", [])

        if not title or not content:
            logger.debug("[Memoir] event #%d skipped: empty title/content", i)
            continue

        # 归一化 source_raw_ids
        try:
            ids = [int(x) for x in raw_ids]
        except (TypeError, ValueError):
            logger.debug("[Memoir] event #%d skipped: source_raw_ids not ints: %r", i, raw_ids)
            continue

        if not ids:
            continue

        # 硬规则 1: 至少一个 id 是本批 new_messages 的
        if not any(x in new_msg_ids for x in ids):
            logger.debug(
                "[Memoir] event #%d skipped: no new_messages raw_id referenced "
                "(context_only 不能单独造事件)", i,
            )
            continue

        # 归一化 keywords（V1 允许空，但不允许非字符串）
        kw = [str(k).strip() for k in keywords if str(k).strip()] if isinstance(keywords, list) else []

        valid.append(ExtractedEvent(
            title=title,
            content=content,
            source_raw_ids=ids,
            keywords=kw[:8],  # 最多 8 个
        ))

    return valid


# ============================================================
# Extractor
# ============================================================

class EventExtractor:
    """
    调用 LLM provider 拆分事件。

    provider 从 Context 里按 config.extract_provider_id 拿，避免走默认 provider 抢配额。
    """

    def __init__(
        self,
        context: Context,
        db: MemoirDB,
        extract_provider_id: str,
    ):
        self.context = context
        self.db = db
        self.extract_provider_id = extract_provider_id.strip()

    def _get_provider(self):
        """
        解析要用的 provider。
        - 有配 extract_provider_id → 精确按 id 拿
        - 没配 → warning 并用 using_provider 兜底（不推荐，会抢星星对话的配额）
        """
        if self.extract_provider_id:
            prov = self.context.get_provider_by_id(self.extract_provider_id)
            if prov is None:
                logger.warning(
                    "[Memoir] extract_provider_id=%r 找不到 provider，回退到 using_provider",
                    self.extract_provider_id,
                )
            return prov or self.context.get_using_provider()
        return self.context.get_using_provider()

    async def extract(
        self,
        new_msgs: list[sqlite3.Row],
        overlap_msgs: list[sqlite3.Row],
        chat_type: str,
    ) -> list[ExtractedEvent]:
        """
        主入口。返回通过校验的 ExtractedEvent 列表。
        任何异常 → 返回 []，raw 保持 unprocessed，下次调度重试。
        """
        if not new_msgs:
            return []

        provider = self._get_provider()
        if provider is None:
            logger.error("[Memoir] 没有可用的 LLM provider，跳过本批")
            return []

        system_prompt = _SYSTEM_PROMPT_GROUP if chat_type == "group" else _SYSTEM_PROMPT_PRIVATE
        user_prompt = build_user_prompt(new_msgs, overlap_msgs, chat_type)
        new_msg_ids = {int(m["id"]) for m in new_msgs}

        logger.debug(
            "[Memoir] extract: chat_type=%s, new=%d, overlap=%d",
            chat_type, len(new_msgs), len(overlap_msgs),
        )

        try:
            resp = await provider.text_chat(
                prompt=user_prompt,
                system_prompt=system_prompt,
                # session_id / contexts 故意不传：这个调用与"星星和陆忱的对话"完全无关
                # 是插件自己的 side channel，不需要历史上下文
            )
        except Exception:
            logger.exception("[Memoir] LLM 事件提取调用失败，跳过本批")
            return []

        text = getattr(resp, "completion_text", None) or ""
        events = parse_llm_response(text, new_msg_ids)
        logger.info(
            "[Memoir] extracted %d event(s) from %d new msg(s) (chat_type=%s)",
            len(events), len(new_msgs), chat_type,
        )
        return events
