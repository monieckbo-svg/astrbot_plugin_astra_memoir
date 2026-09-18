"""
pipeline/extractor.py — LLM 事件拆分

改动（Phase 1f-fix 采纳）:
- ExtractResult 明确区分「合法 events=[]」与「提取失败」
    provider 异常 / JSON 解析失败 / schema 非法 → success=False
    scheduler 只在 success 时才 mark_processed
- 消息按 role 打标签，system prompt 明确固化第一人称记事规则:
    role=assistant 是 Astra 自己 → 用「我」
    role=user 是别人 → 严格点名
    Astra 未参与的事件 → 客观记录，不硬写「我看到」
- 硬规则升级: source_raw_ids 必须全部 ∈ allowed（new ∪ overlap），
  且至少一个 ∈ new_ids
- FTS MATCH 语法错误等由 retriever 层 fallback；本模块只处理 LLM 侧
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field

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
    source_raw_ids: list[int]     # 全部 ∈ allowed，至少一个 ∈ new_ids
    keywords: list[str]
    importance: int = 3


@dataclass
class ExtractResult:
    """
    区分「合法 events=[]」和「提取失败」。

    - success=True + events=[]: 无值得记忆的事件，raw 应标 processed
    - success=True + events=[...]: 有事件，写库后标 processed
    - success=False: LLM 报错 / JSON 解析失败 / 全被硬规则过滤了
      raw 保持 unprocessed，下次调度重试
    """
    success: bool
    events: list[ExtractedEvent] = field(default_factory=list)
    error: str | None = None
    # 观测: 模型返回了 N 个 raw event，被硬规则过滤掉 M 个
    raw_event_count: int = 0
    rejected_count: int = 0


# ============================================================
# Prompt 构造
# ============================================================

_SYSTEM_PROMPT_PRIVATE = """你是记忆系统的事件提取器，只输出严格 JSON。

你在为 AI 助手（在对话中标记为 [role=assistant]，名字通常是"星星"）整理她自己的日常记忆。
提取的事件将来会作为她本人的记忆被检索、复述。

你的任务不是总结整段聊天，而是识别其中相互独立、未来可能被重新提及的事件。

规则：
1. 一批可以包含 0 到 N 个事件。不同主题必须拆开；只有确实属于同一件事的发展过程才合并。
2. 不要为了减少数量把午饭、工作、宠物、画图等无关主题写进同一事件。
3. 每个事件必须能脱离本批次独立理解，保留具体人、物、名称、数字、决定和结果。
4. 不要生成"聊了生活"、"讨论了各种话题"这种空泛摘要。
5. 单纯的"哈哈哈"、表情、寒暄没有值得记忆的内容时，返回 events=[]。
6. source_raw_ids 可引用 <context_only> 和 <new_messages> 中与事件直接相关的 raw_id，但至少包含一个 <new_messages> raw_id；严禁引用本批之外的 raw_id。
7. keywords 是 2~5 个中文关键词，用于后续检索命中。
8. 每个事件必须给 importance 整数 1~5。1=琐碎临时（纯闲聊、表情包、一次性玩笑、普通画图请求、单次无后续价值小事、重复性日常）；2=短期可能继续聊但长期价值低；3=普通可再提事件；4=重要决定、持续项目变化、明确状态变化或明显影响后续互动；5=极重要共同经历或长期关键事件。
9. 高频重复日常行为（频繁画图、普通闲聊、表情包、问候）默认 importance=1；只有本次出现新的偏好、决定、冲突、显著情绪、新设定或持续影响未来互动的信息才提高。只评重要性，不决定永久保存。

【第一人称记事】
- assistant（星星）在事件中：用"我"指代星星本人（例：我提议、我告诉陆忱……）
- 用户在事件中：严格点名（例：陆忱说……）
- 星星未参与的事件：客观记录，不硬写"我看到"
- 人物以 [qq] / [person] 为准，[name] 只是当时昵称。优先用正式名字；代词仅用已设置的 [pronoun]，TA 不猜性别。

输出格式（严格 JSON，不带 markdown fence 或额外说明）：
{"events":[
  {"title":"简短标题","content":"详细内容","importance":3,"source_raw_ids":[101,102],"keywords":["关键词1","关键词2"]}
]}
如没有值得记忆的事件，返回：{"events":[]}"""


_SYSTEM_PROMPT_GROUP = """你是记忆系统的事件提取器，只输出严格 JSON。

你在为 AI 助手（消息中标记 [role=assistant]，昵称通常是"星星"）整理群聊中发生的事件。
她大部分时间只是旁观。提取的事件将来会作为她本人的记忆被检索。

你的任务不是总结整段群聊，而是识别其中相互独立、未来可能被重新提及的事件。

规则：
1. 一批可以包含 0 到 N 个事件。不同主题必须拆开。
2. 群聊中即使 AI 助手完全没有参与，只要有值得记忆的信息（个人经历、观点、决定、事件），也应提取。
3. 每条消息前的 [qq=XXX][person=正式名字][name=当时昵称][role=Z] 表明发言者身份；同一 QQ 是同一个人，不同人的言论必须严格区分。
4. 每个事件必须能脱离本批次独立理解，保留具体人、物、名称、数字、决定和结果。
5. 不要生成"群里聊了各种话题"这种空泛摘要。
6. 单纯的"哈哈哈"、表情、寒暄没有值得记忆的内容时，返回 events=[]。
7. source_raw_ids 可引用 <context_only> 和 <new_messages> 中与事件直接相关的 raw_id，但至少包含一个 <new_messages> raw_id；严禁引用本批之外的 raw_id。
8. keywords 是 2~5 个中文关键词，用于后续检索命中。
9. 每个事件必须给 importance 整数 1~5。1=琐碎临时（纯闲聊、表情包、一次性玩笑、普通画图请求、重复性日常）；2=短期可能继续聊但长期价值低；3=普通可再提事件；4=重要决定、持续项目变化、明确状态变化或明显影响后续互动；5=极重要共同经历或长期关键事件。
10. 高频重复日常行为（频繁画图、普通闲聊、表情包、问候）默认 importance=1；只有本次出现新的偏好、决定、冲突、显著情绪、新设定或持续影响未来互动的信息才提高。只评重要性，不决定永久保存。

【第一人称记事】
- role=assistant（星星）参与的事件：用"我"指代星星本人
- 其他人：严格点名（优先用 [person] 正式名字；[name] 只是当时昵称）
- 星星完全没参与的事件：客观第三人称记录（例："小雨在群里说自己感冒了"），不硬写"我看到"
- 仅按 [pronoun] 使用 TA/她/他，不根据昵称、语气或聊天内容推断性别。

输出格式（严格 JSON，不带 markdown fence 或额外说明）：
{"events":[
  {"title":"简短标题","content":"详细内容，含谁说了什么","importance":3,"source_raw_ids":[8871,8872],"keywords":["关键词1","关键词2"]}
]}
如没有值得记忆的事件，返回：{"events":[]}"""


def _format_msg_line(row: sqlite3.Row, include_speaker_meta: bool,
                     identity: tuple[str, str] | None = None) -> str:
    """
    格式化一条消息给 LLM 看。
    - 私聊: [raw_id=N][role=user] 陆忱: 内容
    - 群聊: [raw_id=N][qq=XXX][name=YYY][role=Z] 内容
    """
    rid = row["id"]
    role = row["role"]
    content = row["content"].replace("\n", " ")  # 单行化，避免 prompt 结构混乱
    person, pronoun = identity or (row["speaker_name"], "TA")
    if include_speaker_meta:
        return (f"[raw_id={rid}][qq={row['speaker_id']}][person={person}]"
                f"[pronoun={pronoun}][name={row['speaker_name']}][role={role}] {content}")
    else:
        return (f"[raw_id={rid}][qq={row['speaker_id']}][person={person}]"
                f"[pronoun={pronoun}][role={role}] {person}: {content}")


def build_user_prompt(
    new_msgs: list[sqlite3.Row],
    overlap_msgs: list[sqlite3.Row],
    chat_type: str,
    identities: dict[str, tuple[str, str]] | None = None,
) -> str:
    """
    组装 user prompt。overlap 放 <context_only>，新消息放 <new_messages>。
    """
    is_group = chat_type == "group"
    identities = identities or {}
    lines: list[str] = []

    if overlap_msgs:
        lines.append("<context_only>")
        lines.append("(以下是上一批对话的尾部，仅帮助你理解上下文，不允许单独据此提取事件)")
        for row in overlap_msgs:
            lines.append(_format_msg_line(row, include_speaker_meta=is_group,
                                          identity=identities.get(str(row["speaker_id"]))))
        lines.append("</context_only>")
        lines.append("")

    lines.append("<new_messages>")
    lines.append("(以下是本批新增消息，事件的 source_raw_ids 必须至少含其中一个 raw_id)")
    for row in new_msgs:
        lines.append(_format_msg_line(row, include_speaker_meta=is_group,
                                      identity=identities.get(str(row["speaker_id"]))))
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


def parse_llm_response(
    text: str,
    *,
    new_msg_ids: set[int],
    overlap_msg_ids: set[int],
) -> ExtractResult:
    """
    解析 LLM 返回的 JSON，应用硬规则校验。

    硬规则：
    - source_raw_ids 必须全部 ∈ (new_msg_ids ∪ overlap_msg_ids)
    - source_raw_ids 必须至少含一个 ∈ new_msg_ids
    - title / content 非空
    - source_raw_ids 里所有 id 必须是数字

    区分:
    - LLM 返回 events=[] → success=True, events=[]
    - JSON 解析失败 → success=False
    - LLM 返回了 events 但全部被硬规则过滤 → success=False（避免 silently 丢事件）
    """
    if not text:
        return ExtractResult(success=False, error="empty LLM response")

    body = _strip_json_fence(text)
    obj_str = _extract_first_json_object(body)
    if obj_str is None:
        logger.warning("[Memoir] extractor: no JSON object in LLM response: %r", text[:200])
        return ExtractResult(success=False, error="no JSON object in response")

    try:
        obj = json.loads(obj_str)
    except json.JSONDecodeError as e:
        logger.warning("[Memoir] extractor: JSON decode fail (%s): %r", e, obj_str[:200])
        return ExtractResult(success=False, error=f"JSON decode: {e}")

    if not isinstance(obj, dict):
        return ExtractResult(success=False, error="root not an object")

    raw_events = obj.get("events", None)
    if raw_events is None or not isinstance(raw_events, list):
        return ExtractResult(success=False, error="events not a list")

    # 合法的空结果
    if not raw_events:
        return ExtractResult(success=True, events=[], raw_event_count=0)

    allowed = new_msg_ids | overlap_msg_ids

    valid: list[ExtractedEvent] = []
    rejected = 0
    for i, ev in enumerate(raw_events):
        if not isinstance(ev, dict):
            rejected += 1
            continue

        title = str(ev.get("title", "")).strip()
        content = str(ev.get("content", "")).strip()
        raw_ids = ev.get("source_raw_ids", [])
        keywords = ev.get("keywords", [])
        importance = ev.get("importance")

        if type(importance) is not int or not 1 <= importance <= 5:
            logger.debug("[Memoir] event #%d rejected: invalid importance=%r", i, importance)
            rejected += 1
            continue

        if not title or not content:
            logger.debug("[Memoir] event #%d rejected: empty title/content", i)
            rejected += 1
            continue

        try:
            ids = [int(x) for x in raw_ids]
        except (TypeError, ValueError):
            logger.debug("[Memoir] event #%d rejected: source_raw_ids not ints: %r", i, raw_ids)
            rejected += 1
            continue

        if not ids:
            rejected += 1
            continue

        # 硬规则 1: 所有 id 必须 ∈ allowed
        out_of_scope = [x for x in ids if x not in allowed]
        if out_of_scope:
            logger.debug(
                "[Memoir] event #%d rejected: source_raw_ids %s not in allowed set (out=%s)",
                i, ids, out_of_scope,
            )
            rejected += 1
            continue

        # 硬规则 2: 至少一个 ∈ new_msg_ids（context_only 不能单独造事件）
        if not any(x in new_msg_ids for x in ids):
            logger.debug(
                "[Memoir] event #%d rejected: no new_messages raw_id referenced", i
            )
            rejected += 1
            continue

        kw = [str(k).strip() for k in keywords if str(k).strip()] if isinstance(keywords, list) else []

        valid.append(ExtractedEvent(
            title=title,
            content=content,
            source_raw_ids=ids,
            keywords=kw[:8],
            importance=importance,
        ))

    # LLM 明明返回了事件但全部被过滤 → 视为失败，保持 raw unprocessed
    if rejected:
        return ExtractResult(
            success=False,
            events=[],
            error=f"{rejected} event(s) rejected by hard rules; whole batch refused",
            raw_event_count=len(raw_events),
            rejected_count=rejected,
        )

    return ExtractResult(
        success=True,
        events=valid,
        raw_event_count=len(raw_events),
        rejected_count=rejected,
    )


# ============================================================
# Extractor
# ============================================================

class EventExtractor:
    """调用 LLM provider 拆分事件。"""

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
        if self.extract_provider_id:
            prov = self.context.get_provider_by_id(self.extract_provider_id)
            if prov is None:
                logger.error(
                    "[Memoir] extract_provider_id=%r 找不到 provider，拒绝本批提取",
                    self.extract_provider_id,
                )
            return prov
        return self.context.get_using_provider()

    async def extract(
        self,
        new_msgs: list[sqlite3.Row],
        overlap_msgs: list[sqlite3.Row],
        chat_type: str,
    ) -> ExtractResult:
        """
        主入口。任何失败都会返回 success=False（scheduler 不会 mark_processed）。
        """
        if not new_msgs:
            return ExtractResult(success=True, events=[])

        provider = self._get_provider()
        if provider is None:
            error = (f"configured extract provider unavailable: {self.extract_provider_id}"
                     if self.extract_provider_id else "no LLM provider available")
            logger.error("[Memoir] %s，跳过本批", error)
            return ExtractResult(success=False, error=error)

        system_prompt = _SYSTEM_PROMPT_GROUP if chat_type == "group" else _SYSTEM_PROMPT_PRIVATE
        def _identity_map():
            ids = {str(m["speaker_id"]) for m in [*new_msgs, *overlap_msgs]}
            return {qq: self.db.identities.display(qq) for qq in ids}
        identities = await self.db.run(_identity_map)
        user_prompt = build_user_prompt(new_msgs, overlap_msgs, chat_type, identities)
        new_msg_ids = {int(m["id"]) for m in new_msgs}
        overlap_msg_ids = {int(m["id"]) for m in overlap_msgs}

        logger.debug(
            "[Memoir] extract: chat_type=%s, new=%d, overlap=%d",
            chat_type, len(new_msgs), len(overlap_msgs),
        )

        for attempt in range(2):
            try:
                resp = await provider.text_chat(
                    prompt=user_prompt if attempt == 0 else user_prompt + "\n上次输出有无效事件。请重新输出完整 JSON，确保每个事件都满足全部 raw_id 硬规则。",
                    system_prompt=system_prompt,
                )
            except Exception as e:
                logger.exception("[Memoir] LLM 事件提取调用失败: %s", e)
                return ExtractResult(success=False, error=f"LLM call: {e}")

            text = getattr(resp, "completion_text", None) or ""
            result = parse_llm_response(
                text, new_msg_ids=new_msg_ids, overlap_msg_ids=overlap_msg_ids,
            )
            if result.success or not result.rejected_count:
                break

        if result.success:
            logger.info(
                "[Memoir] extracted %d event(s) from %d new msg(s) (chat_type=%s, rejected=%d)",
                len(result.events), len(new_msgs), chat_type, result.rejected_count,
            )
        else:
            logger.warning(
                "[Memoir] extract FAILED: %s (raw_event_count=%d, rejected=%d)",
                result.error, result.raw_event_count, result.rejected_count,
            )
        return result
