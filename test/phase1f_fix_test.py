"""
phase1f_fix_test.py — Phase 1f-fix 采纳后新增的 8 个场景

1. LLM provider 报错 → raw 必须仍 unprocessed
2. malformed JSON → raw 必须仍 unprocessed
3. 单 event writer 失败（embedding None）→ raw 不得 silently processed
4. 10 completed + 第 11 条未回复 → 第 11 条不得进 batch
5. 群里一次积压 50 条 → 第一批只处理设定的 20 条
6. source_raw_ids 幻觉其它 session id → 整个 event 被拒绝
7. group_id 缺失 → 召回必须为空
8. 两 session 并行 process → SQLite transaction 不冲突
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import time
import types
from pathlib import Path


# ---- mock astrbot ----
def _mock_astrbot():
    api = types.ModuleType("astrbot")
    api_api = types.ModuleType("astrbot.api")
    api_api.logger = logging.getLogger("memoir-test")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    api_event = types.ModuleType("astrbot.api.event")
    class _Filter:
        def event_message_type(self, *a, **k):
            def deco(fn): return fn
            return deco
        def on_llm_response(self, *a, **k):
            def deco(fn): return fn
            return deco
        def on_llm_request(self, *a, **k):
            def deco(fn): return fn
            return deco
    api_event.filter = _Filter()
    class AstrMessageEvent: pass
    api_event.AstrMessageEvent = AstrMessageEvent

    api_event_filter = types.ModuleType("astrbot.api.event.filter")
    class EventMessageType: ALL = "ALL"
    api_event_filter.EventMessageType = EventMessageType

    api_star = types.ModuleType("astrbot.api.star")
    class Context: pass
    class Star:
        def __init__(self, *a, **k): pass
    class StarTools:
        @staticmethod
        def get_data_dir(name): return Path(tempfile.gettempdir()) / name
    def register(*a, **k):
        def deco(cls): return cls
        return deco
    api_star.Context = Context
    api_star.Star = Star
    api_star.StarTools = StarTools
    api_star.register = register

    core = types.ModuleType("astrbot.core")
    core_provider = types.ModuleType("astrbot.core.provider")
    core_provider_entities = types.ModuleType("astrbot.core.provider.entities")
    class LLMResponse: pass
    core_provider_entities.LLMResponse = LLMResponse

    core_agent = types.ModuleType("astrbot.core.agent")
    core_agent_message = types.ModuleType("astrbot.core.agent.message")
    class TextPart:
        def __init__(self, text: str):
            self.text = text
            self._no_save = False
        def mark_as_temp(self):
            self._no_save = True
            return self
    core_agent_message.TextPart = TextPart

    sys.modules["astrbot"] = api
    sys.modules["astrbot.api"] = api_api
    sys.modules["astrbot.api.event"] = api_event
    sys.modules["astrbot.api.event.filter"] = api_event_filter
    sys.modules["astrbot.api.star"] = api_star
    sys.modules["astrbot.core"] = core
    sys.modules["astrbot.core.provider"] = core_provider
    sys.modules["astrbot.core.provider.entities"] = core_provider_entities
    sys.modules["astrbot.core.agent"] = core_agent
    sys.modules["astrbot.core.agent.message"] = core_agent_message


_mock_astrbot()

_PKG_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_PKG_DIR.parent))
_PKG = _PKG_DIR.name

import importlib
storage_mod = importlib.import_module(f"{_PKG}.storage")
pipeline_mod = importlib.import_module(f"{_PKG}.pipeline")
MemoirDB = storage_mod.MemoirDB
VecStore = storage_mod.VecStore
RawCache = pipeline_mod.RawCache
EventExtractor = pipeline_mod.EventExtractor
EpisodeWriter = pipeline_mod.EpisodeWriter
BatchScheduler = pipeline_mod.BatchScheduler
SchedulerConfig = pipeline_mod.SchedulerConfig
Retriever = pipeline_mod.Retriever
RetrieverConfig = pipeline_mod.RetrieverConfig


# ============================================================
# Mocks
# ============================================================

class FakeLLMResponse:
    def __init__(self, text: str):
        self.completion_text = text


class ExplodingProvider:
    """text_chat 直接抛异常。"""
    async def text_chat(self, prompt, system_prompt="", **kw):
        raise RuntimeError("simulated provider outage")


class MalformedJsonProvider:
    """返回一段无法解析的文本。"""
    async def text_chat(self, prompt, system_prompt="", **kw):
        return FakeLLMResponse("this is not JSON at all, {broken")


class HallucinatingProvider:
    """返回带 out-of-scope raw_id 的事件。"""
    def __init__(self, out_of_scope_id: int):
        self.out_of_scope_id = out_of_scope_id
    async def text_chat(self, prompt, system_prompt="", **kw):
        return FakeLLMResponse(json.dumps({
            "events": [
                {
                    "title": "幻觉事件",
                    "content": "假装引用了不属于本 session 的 raw",
                    "source_raw_ids": [self.out_of_scope_id],
                    "keywords": ["假"],
                }
            ]
        }, ensure_ascii=False))


class OneGoodOneBadProvider:
    """一批里第一个 event 走 new_ids，第二个用 overlap_only（无 new 引用）应被拒。"""
    def __init__(self, good_new_ids: list[int], bad_overlap_id: int):
        self.good = good_new_ids
        self.bad = bad_overlap_id
    async def text_chat(self, prompt, system_prompt="", **kw):
        return FakeLLMResponse(json.dumps({
            "events": [
                {"title": "好事件", "content": "aaa",
                 "source_raw_ids": self.good, "keywords": []},
                {"title": "只用overlap", "content": "bbb",
                 "source_raw_ids": [self.bad], "keywords": []},
            ]
        }, ensure_ascii=False))


class SimpleGoodProvider:
    """针对 prompt 里出现的 raw_id 返回一个合法事件。"""
    async def text_chat(self, prompt, system_prompt="", **kw):
        import re
        raw_ids = [int(x) for x in re.findall(r"\[raw_id=(\d+)\]", prompt)]
        # 只用 new_messages 里的（我们简单取所有 raw_id 的前 3 个）
        if not raw_ids:
            return FakeLLMResponse('{"events":[]}')
        return FakeLLMResponse(json.dumps({
            "events": [{
                "title": "事件",
                "content": "内容 " + " ".join(str(r) for r in raw_ids[:3]),
                "source_raw_ids": raw_ids[:3],
                "keywords": ["k"],
            }]
        }, ensure_ascii=False))


class FailingEmbeddingProvider:
    """始终抛异常 → 让 write_batch 失败。"""
    id = "fail-embed"
    async def get_embedding(self, text):
        raise RuntimeError("simulated embedding outage")


class OkEmbeddingProvider:
    id = "ok"
    def __init__(self, dim=8):
        self.dim = dim
    async def get_embedding(self, text):
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        raw = [b / 255.0 for b in h[: self.dim]]
        n = sum(x*x for x in raw) ** 0.5 or 1.0
        return [x / n for x in raw]


class FakeContext:
    def __init__(self, llm, embed):
        self._llm = llm
        self._embed = embed
    def get_provider_by_id(self, pid): return self._llm
    def get_using_provider(self, umo=None): return self._llm
    def get_all_embedding_providers(self): return [self._embed]


# ============================================================
# Helpers
# ============================================================

async def build_stack(llm, embed=None):
    tmp = tempfile.mkdtemp()
    db = MemoirDB(Path(tmp) / "t.db", embedding_dim=8)
    db.initialize()
    vec = VecStore(db)
    if embed is None:
        embed = OkEmbeddingProvider()
    context = FakeContext(llm, embed)
    extractor = EventExtractor(context, db, extract_provider_id="")
    writer = EpisodeWriter(context, db, vec, embedding_provider_id="")
    cfg = SchedulerConfig(
        interval_seconds=1,
        private_batch_user_turns=5,
        group_batch_inbound=20,
    )
    scheduler = BatchScheduler(db, extractor, writer, cfg)
    return db, vec, extractor, writer, scheduler, context


async def insert_priv_pair(db, session, uid_prefix, i, ts):
    """插入一对 user + assistant，返回 (user_raw_id, assistant_raw_id)。"""
    u = db.insert_raw_message(
        dedupe_key=f"user:{session}:{uid_prefix}u{i}", platform="qq",
        chat_type="private", session_id=session, group_id=None,
        platform_message_id=f"{uid_prefix}u{i}",
        speaker_id="111", speaker_name="陆忱", role="user",
        content=f"user msg {i}", reply_to_id=None,
        trigger_raw_id=None, created_at=ts,
    )
    a = db.insert_raw_message(
        dedupe_key=f"assistant:{session}:{u}", platform="qq",
        chat_type="private", session_id=session, group_id=None,
        platform_message_id=None,
        speaker_id="999", speaker_name="星星", role="assistant",
        content=f"assistant reply {i}", reply_to_id=f"{uid_prefix}u{i}",
        trigger_raw_id=u, created_at=ts + 1,
    )
    return u, a


def count_unprocessed(db, session):
    return db.fetchone(
        "SELECT COUNT(*) AS n FROM recent_messages "
        "WHERE session_id = ? AND processed_at IS NULL",
        (session,)
    )["n"]


# ============================================================
# Tests
# ============================================================

async def test_1_llm_error_keeps_raw_unprocessed():
    """LLM 报错 → raw 必须仍 unprocessed"""
    db, vec, extractor, writer, scheduler, _ = await build_stack(ExplodingProvider())
    session = "s1"
    now = int(time.time())
    for i in range(3):
        await insert_priv_pair(db, session, "p", i, now + i * 10)

    assert count_unprocessed(db, session) == 6
    await scheduler.process_batch(session, mode="threshold")
    assert count_unprocessed(db, session) == 6, "LLM 报错时 raw 应保持 unprocessed"
    # 也不应有 episode
    assert db.fetchone("SELECT COUNT(*) AS n FROM episodes")["n"] == 0
    db.close()
    print("✓ 1. LLM provider 报错 → raw 仍 unprocessed，无 episode 生成")


async def test_2_malformed_json_keeps_raw_unprocessed():
    """malformed JSON → raw 必须仍 unprocessed"""
    db, vec, extractor, writer, scheduler, _ = await build_stack(MalformedJsonProvider())
    session = "s2"
    now = int(time.time())
    for i in range(3):
        await insert_priv_pair(db, session, "p", i, now + i * 10)

    await scheduler.process_batch(session, mode="threshold")
    assert count_unprocessed(db, session) == 6, "JSON 解析失败时 raw 应保持 unprocessed"
    assert db.fetchone("SELECT COUNT(*) AS n FROM episodes")["n"] == 0
    db.close()
    print("✓ 2. malformed JSON → raw 仍 unprocessed，无 episode 生成")


async def test_3_embedding_failure_blocks_processed():
    """embedding 失败 → 整批不能 silently processed"""
    db, vec, extractor, writer, scheduler, _ = await build_stack(
        SimpleGoodProvider(), embed=FailingEmbeddingProvider(),
    )
    session = "s3"
    now = int(time.time())
    for i in range(3):
        await insert_priv_pair(db, session, "p", i, now + i * 10)

    await scheduler.process_batch(session, mode="threshold")
    assert count_unprocessed(db, session) == 6, "writer 失败时 raw 不得 silently processed"
    assert db.fetchone("SELECT COUNT(*) AS n FROM episodes")["n"] == 0
    db.close()
    print("✓ 3. embedding 失败 → raw 保持 unprocessed，episode 未落库")


async def test_4_bounded_batch_private_completed():
    """10 completed + 第 11 条未回复 → 第 11 条不进 batch"""
    db, vec, extractor, writer, scheduler, _ = await build_stack(SimpleGoodProvider())
    session = "s4"
    now = int(time.time())
    # 10 个完整的 user+assistant 对
    for i in range(10):
        await insert_priv_pair(db, session, "p", i, now + i * 10)
    # 第 11 条 user，没有 assistant reply
    orphan_uid = db.insert_raw_message(
        dedupe_key=f"user:{session}:orphan", platform="qq",
        chat_type="private", session_id=session, group_id=None,
        platform_message_id="orphan",
        speaker_id="111", speaker_name="陆忱", role="user",
        content="第 11 条用户消息未回复", reply_to_id=None,
        trigger_raw_id=None, created_at=now + 200,
    )
    assert count_unprocessed(db, session) == 21  # 10 pair + 1 孤

    # threshold 只应吃 10 completed（scheduler 的 batch_size 是 config
    # 私聊阈值，这里默认 private_batch_user_turns=5）
    scheduler.config.private_batch_user_turns = 10
    await scheduler.process_batch(session, mode="threshold")

    # 应该吃掉 10 pair = 20 条，剩下孤 user 1 条
    remaining = db.fetchall(
        "SELECT id FROM recent_messages WHERE session_id = ? AND processed_at IS NULL",
        (session,)
    )
    remaining_ids = {r["id"] for r in remaining}
    assert remaining_ids == {orphan_uid}, \
        f"孤 user #{orphan_uid} 应保留，其他 processed；实际 remaining={remaining_ids}"
    db.close()
    print(f"✓ 4. 10 completed + 1 孤 user → threshold batch 只吃 10 pair，孤 user #{orphan_uid} 留下")


async def test_5_group_bounded_batch():
    """群里 50 条 → 第一批只处理设定的 20 条"""
    db, vec, extractor, writer, scheduler, _ = await build_stack(SimpleGoodProvider())
    session = "gp1"
    now = int(time.time())
    for i in range(50):
        db.insert_raw_message(
            dedupe_key=f"user:{session}:g{i}", platform="qq",
            chat_type="group", session_id=session, group_id="g_main",
            platform_message_id=f"g{i}",
            speaker_id=str(200 + (i % 5)), speaker_name=f"人{i%5}",
            role="user", content=f"群消息 {i}", reply_to_id=None,
            trigger_raw_id=None, created_at=now + i,
        )

    assert count_unprocessed(db, session) == 50
    await scheduler.process_batch(session, mode="threshold")

    remaining = count_unprocessed(db, session)
    assert remaining == 30, f"第一批只应吃 20 条，剩 30；实际剩 {remaining}"
    db.close()
    print("✓ 5. 群 50 条积压 → 第一批只吃 20，剩 30")


async def test_6_hallucinated_raw_id_rejected():
    """source_raw_ids 幻觉其它 session → 整个 event 被拒绝"""
    db, vec, extractor, writer, scheduler, _ = await build_stack(
        HallucinatingProvider(out_of_scope_id=999999),
    )
    session = "s6"
    now = int(time.time())
    for i in range(3):
        await insert_priv_pair(db, session, "p", i, now + i * 10)

    await scheduler.process_batch(session, mode="threshold")
    # LLM 返回的 event 全被硬规则过滤 → success=False → raw 保持 unprocessed
    assert count_unprocessed(db, session) == 6, \
        "所有 event 被拒时 raw 应保持 unprocessed"
    assert db.fetchone("SELECT COUNT(*) AS n FROM episodes")["n"] == 0
    db.close()
    print("✓ 6. 幻觉 source_raw_ids → event 被拒 → raw 未标 processed")


async def test_7_missing_group_id_returns_empty():
    """群聊 group_id 缺失 → recall 返回空"""
    db, vec, extractor, writer, scheduler, _ = await build_stack(SimpleGoodProvider())
    embed = OkEmbeddingProvider()
    context = FakeContext(SimpleGoodProvider(), embed)
    cfg = RetrieverConfig(top_k=4, max_cosine_distance=1.5)
    retr = Retriever(context, db, vec, cfg, embedding_provider_id="")

    # 塞一条群 episode
    now = int(time.time())
    rid = db.insert_raw_message(
        dedupe_key="user:g:msg", platform="qq", chat_type="group",
        session_id="g_sess", group_id="g_main", platform_message_id="msg",
        speaker_id="111", speaker_name="陆忱", role="user",
        content="X", reply_to_id=None, trigger_raw_id=None, created_at=now,
    )
    with db.transaction():
        eid = db.insert_episode(
            platform="qq", chat_type="group", session_id="g_sess",
            group_id="g_main", title="X", content="X",
            source_raw_ids=[rid], event_start_at=now,
            event_end_at=now, extracted_at=now,
        )
        db.insert_participants(eid, [{"speaker_id":"111","speaker_name":"陆忱","role":"user"}])
        db.insert_fts(eid, "X", "X", [])
    v = await embed.get_embedding("X\nX")
    vec.upsert(eid, v, chat_type="group", session_id="g_sess", group_id="g_main")

    # 群聊场景传 group_id=None → 应返回空（不能全库可见）
    r = await retr.recall(
        "X", session_id="g_other", chat_type="group",
        group_id=None, current_speaker_id="111",
    )
    assert r == [], f"group_id=None 时召回必须为空，实际 got {len(r)}"

    # 传空字符串也应视作缺失
    r = await retr.recall(
        "X", session_id="g_other", chat_type="group",
        group_id="", current_speaker_id="111",
    )
    assert r == [], f"group_id='' 时召回也必须为空"

    # 正常传 group_id 才能召回
    r = await retr.recall(
        "X", session_id="g_sess", chat_type="group",
        group_id="g_main", current_speaker_id="111",
    )
    assert len(r) == 1

    db.close()
    print("✓ 7. group_id 缺失 → recall 返回空；正常传值可召回")


async def test_8_parallel_sessions_no_conflict():
    """两 session 并行 process → transaction 不冲突"""
    db, vec, extractor, writer, scheduler, _ = await build_stack(SimpleGoodProvider())

    session_a = "sa"
    session_b = "sb"
    now = int(time.time())
    for i in range(3):
        await insert_priv_pair(db, session_a, "a", i, now + i * 10)
    for i in range(3):
        await insert_priv_pair(db, session_b, "b", i, now + i * 10)

    # 并行触发
    scheduler.config.private_batch_user_turns = 3
    await asyncio.gather(
        scheduler.process_batch(session_a, mode="threshold"),
        scheduler.process_batch(session_b, mode="threshold"),
    )

    remaining_a = count_unprocessed(db, session_a)
    remaining_b = count_unprocessed(db, session_b)
    assert remaining_a == 0, f"session_a 应全 processed，剩 {remaining_a}"
    assert remaining_b == 0, f"session_b 应全 processed，剩 {remaining_b}"

    eps = db.fetchall("SELECT DISTINCT session_id FROM episodes")
    session_ids = {e["session_id"] for e in eps}
    assert session_a in session_ids and session_b in session_ids

    db.close()
    print("✓ 8. 两 session 并行 process_batch → 无冲突，各自落库")


# ============================================================
# Main
# ============================================================

async def main():
    await test_1_llm_error_keeps_raw_unprocessed()
    await test_2_malformed_json_keeps_raw_unprocessed()
    await test_3_embedding_failure_blocks_processed()
    await test_4_bounded_batch_private_completed()
    await test_5_group_bounded_batch()
    await test_6_hallucinated_raw_id_rejected()
    await test_7_missing_group_id_returns_empty()
    await test_8_parallel_sessions_no_conflict()

    print()
    print("========== ALL 8 PHASE 1F-FIX TESTS PASSED ==========")


if __name__ == "__main__":
    asyncio.run(main())
