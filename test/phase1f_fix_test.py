"""
phase1f_fix_test.py — Phase 1f-fix 采纳后新增的 8 个场景

1. LLM provider 报错 → raw 必须仍 unprocessed
2. malformed JSON → raw 必须仍 unprocessed
3. 单 event writer 失败（embedding None）→ raw 不得 silently processed
4. 10 completed + 第 11 条未回复 → 第 11 条不得进 batch
5. 群里一次积压 50 条 → 连续处理两个完整的 20 条批次
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
    api_web = types.ModuleType("astrbot.api.web")
    api_web.request = types.SimpleNamespace(query={})

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
    sys.modules["astrbot.api.web"] = api_web
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
                    "importance": 3,
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
                {"title": "好事件", "content": "aaa", "importance": 3,
                 "source_raw_ids": self.good, "keywords": []},
                {"title": "只用overlap", "content": "bbb", "importance": 3,
                 "source_raw_ids": [self.bad], "keywords": []},
            ]
        }, ensure_ascii=False))


class SimpleGoodProvider:
    """针对 prompt 里出现的 raw_id 返回一个合法事件。"""
    async def text_chat(self, prompt, system_prompt="", **kw):
        import re
        new_block = prompt.split("<new_messages>", 1)[-1].split("</new_messages>", 1)[0]
        raw_ids = [int(x) for x in re.findall(r"\[raw_id=(\d+)\]", new_block)]
        if not raw_ids:
            return FakeLLMResponse('{"events":[]}')
        return FakeLLMResponse(json.dumps({
            "events": [{
                "title": "事件",
                "importance": 3,
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
    writer = EpisodeWriter(embed, db, vec)
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
    assert remaining == 10, f"应连续处理两个完整的 20 条批次，剩 10；实际剩 {remaining}"
    db.close()
    print("✓ 5. 群 50 条积压 → 连续处理两个 20 条批次，剩 10")


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
    retr = Retriever(embed, db, vec, cfg)

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


async def test_9_group_same_second_and_late_reply():
    db, vec, extractor, writer, scheduler, _ = await build_stack(SimpleGoodProvider())
    now = int(time.time())
    inbound = []
    for i in range(22):
        rid = db.insert_raw_message(
            dedupe_key=f"same-second:{i}", platform="qq", chat_type="group",
            session_id="same-second", group_id="g", platform_message_id=str(i),
            speaker_id="111", speaker_name="甲", role="user", content=f"消息{i}",
            reply_to_id=None, trigger_raw_id=None, created_at=now,
        )
        inbound.append(rid)
    reply = db.insert_raw_message(
        dedupe_key="late-reply", platform="qq", chat_type="group",
        session_id="same-second", group_id="g", platform_message_id=None,
        speaker_id="999", speaker_name="星星", role="assistant", content="回复",
        reply_to_id=None, trigger_raw_id=inbound[19], created_at=now + 5,
    )
    batch = db.get_bounded_batch("same-second", "group", mode="threshold", batch_size=20)
    ids = {r["id"] for r in batch}
    assert ids == set(inbound[:20]) | {reply}, ids
    db.close()
    print("✓ 9. 同秒第 21/22 条不越界，晚时间戳的回复仍归第 20 条")


async def test_10_idle_chunks_and_repair():
    db, vec, extractor, writer, scheduler, _ = await build_stack(SimpleGoodProvider())
    scheduler.config.group_batch_inbound = 20
    now = int(time.time())
    for i in range(45):
        db.insert_raw_message(
            dedupe_key=f"idle:{i}", platform="qq", chat_type="group",
            session_id="idle", group_id="g", platform_message_id=str(i),
            speaker_id="111", speaker_name="甲", role="user", content=f"消息{i}",
            reply_to_id=None, trigger_raw_id=None, created_at=now + i,
        )
    assert len(db.get_bounded_batch("idle", "group", mode="idle", batch_size=20)) == 20
    await scheduler.process_batch("idle", mode="idle")
    assert count_unprocessed(db, "idle") == 0
    episode_count = db.fetchone("SELECT COUNT(*) AS n FROM episodes")["n"]
    assert episode_count == 3, episode_count
    missing_id = db.fetchone("SELECT MIN(id) AS id FROM episodes")["id"]
    vec.delete(missing_id)
    assert missing_id in vec.missing_episode_ids()
    result = await writer.reindex_missing_vectors()
    assert result == {"repaired": 1, "failed": 0}, result
    assert vec.missing_episode_ids() == []
    db.close()
    print("✓ 10. idle 45 条分为 20/20/5，缺失向量可补齐")


async def test_11_partial_invalid_is_atomic():
    from importlib import import_module
    parse = import_module(f"{_PKG}.pipeline.extractor").parse_llm_response
    result = parse(json.dumps({"events": [
        {"title": "好", "content": "事", "importance": 3, "source_raw_ids": [1]},
        {"title": "坏", "content": "事", "importance": 3, "source_raw_ids": [999]},
    ]}), new_msg_ids={1}, overlap_msg_ids=set())
    assert not result.success and result.events == [] and result.rejected_count == 1
    print("✓ 11. 混合合法/非法事件整批拒绝")


async def test_12_provider_switch_rebuilds_only_vectors():
    db, vec, extractor, writer, scheduler, _ = await build_stack(SimpleGoodProvider())
    await insert_priv_pair(db, "switch", "p", 0, int(time.time()))
    await scheduler.process_batch("switch", mode="idle")
    assert vec.count() == 1
    path = db.db_path
    db.close()
    switched = MemoirDB(path, embedding_dim=8, embedding_provider_id="new-provider")
    switched.initialize()
    assert switched.fetchone("SELECT COUNT(*) AS n FROM episodes")["n"] == 1
    assert VecStore(switched).count() == 0
    assert VecStore(switched).missing_episode_ids() == [1]
    switched.close()
    print("✓ 12. provider 变化只重建 vec，保留 episode")


async def test_13_semantic_duplicate_skipped():
    from importlib import import_module
    ExtractedEvent = import_module(f"{_PKG}.pipeline.extractor").ExtractedEvent
    db, vec, extractor, writer, scheduler, _ = await build_stack(SimpleGoodProvider())
    now = int(time.time())
    raw_ids = []
    for i in range(2):
        raw_ids.append(db.insert_raw_message(
            dedupe_key=f"duplicate:{i}", platform="qq", chat_type="private",
            session_id="duplicate", group_id=None, platform_message_id=str(i),
            speaker_id="111", speaker_name="甲", role="user", content="插件故障",
            reply_to_id=None, trigger_raw_id=None, created_at=now + i,
        ))
    for rid in raw_ids:
        await writer.write_batch(
            [ExtractedEvent(title="插件故障", content="插件启动失败", source_raw_ids=[rid], keywords=[])],
            [rid], session_id="duplicate", chat_type="private", group_id=None, platform="qq",
        )
    assert db.fetchone("SELECT COUNT(*) AS n FROM episodes")["n"] == 1
    assert count_unprocessed(db, "duplicate") == 0
    db.close()
    print("✓ 13. 高度重复 episode 跳过，同时 raw 正常 processed")


async def test_14_embedding_selection_is_strict():
    from importlib import import_module
    embedding = import_module(f"{_PKG}.pipeline.embedding")
    provider = OkEmbeddingProvider(dim=8)
    context = FakeContext(SimpleGoodProvider(), provider)
    assert embedding.resolve_embedding_provider(context, "ok") is provider

    class MetaOnlyEmbeddingProvider:
        def meta(self):
            return types.SimpleNamespace(id="meta-only", name="测试向量模型")
        def get_dim(self):
            return 8
        async def get_embedding(self, text):
            return await provider.get_embedding(text)

    meta_only = MetaOnlyEmbeddingProvider()
    meta_context = FakeContext(SimpleGoodProvider(), meta_only)
    assert not hasattr(meta_only, "id")
    assert embedding.resolve_embedding_provider(meta_context, "meta-only") is meta_only
    assert embedding.provider_display_name(meta_only, "meta-only") == "测试向量模型"
    assert await embedding.embedding_dimension(meta_only) == 8
    for invalid in ("", "missing"):
        try:
            embedding.resolve_embedding_provider(context, invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"provider {invalid!r} must not fallback")
    assert await embedding.embedding_dimension(provider) == 8
    print("✓ 14. 兼容 meta().id、无效 ID 不回退、维度自动探测")


async def test_15_late_assistant_idle_flush():
    db, vec, extractor, writer, scheduler, _ = await build_stack(SimpleGoodProvider())
    now = int(time.time())
    user_id = db.insert_raw_message(
        dedupe_key="late:u", platform="qq", chat_type="group", session_id="late",
        group_id="g", platform_message_id="u", speaker_id="111", speaker_name="甲",
        role="user", content="问题", reply_to_id=None, trigger_raw_id=None,
        created_at=now,
    )
    db.mark_processed([user_id], now)
    reply_id = db.insert_raw_message(
        dedupe_key="late:a", platform="qq", chat_type="group", session_id="late",
        group_id="g", platform_message_id=None, speaker_id="999", speaker_name="星星",
        role="assistant", content="答案", reply_to_id=None, trigger_raw_id=user_id,
        created_at=now + 10,
    )
    batch = db.get_bounded_batch("late", "group", mode="idle", batch_size=20)
    assert [r["id"] for r in batch] == [reply_id]
    await scheduler.process_batch("late", mode="idle")
    assert count_unprocessed(db, "late") == 0
    db.close()
    print("✓ 15. 已处理 inbound 的迟到回复仍能 idle 消化")


async def test_16_invalid_extract_provider_never_falls_back():
    class CountingLLM:
        calls = 0
        async def text_chat(self, **kwargs):
            self.calls += 1
            return FakeLLMResponse('{"events":[]}')

    llm = CountingLLM()
    db, vec, extractor, writer, scheduler, _ = await build_stack(llm)
    class MissingSelectedProvider(FakeContext):
        def get_provider_by_id(self, pid): return None
    extractor.context = MissingSelectedProvider(llm, OkEmbeddingProvider())
    extractor.extract_provider_id = "missing-extract-provider"
    rid = db.insert_raw_message(
        dedupe_key="invalid-extract", platform="qq", chat_type="private",
        session_id="extract", group_id=None, platform_message_id="1",
        speaker_id="111", speaker_name="甲", role="user", content="测试",
        reply_to_id=None, trigger_raw_id=None, created_at=int(time.time()),
    )
    await scheduler.process_batch("extract", mode="idle")
    assert llm.calls == 0
    assert count_unprocessed(db, "extract") == 1
    db.close()
    print("✓ 16. 无效 extract provider 不调用主 provider，raw 保持未处理")


async def test_17_panel_invalid_fts_query_falls_back():
    from importlib import import_module
    from astrbot.api.web import request
    panel_mod = import_module(f"{_PKG}.pipeline.panel")
    db, vec, extractor, writer, scheduler, _ = await build_stack(SimpleGoodProvider())
    now = int(time.time())
    eid = db.insert_episode(
        platform="qq", chat_type="private", session_id="search", group_id=None,
        title="插件/故障", content="插件/故障处理好了", source_raw_ids=[],
        event_start_at=now, event_end_at=now, extracted_at=now,
    )
    db.insert_participants(eid, [{"speaker_id": "111", "speaker_name": "甲", "role": "user"}])
    db.insert_fts(eid, "插件/故障", "插件/故障处理好了", [])
    request.query = {"q": "插件/故障", "participant": "111", "chat_type": "private"}
    panel = panel_mod.MemoirPanel(db, vec, None, scheduler, writer,
                                  writer.embedding_provider, "ok", 8)
    result = await panel.list_episodes()
    assert result["status"] == "ok", result
    assert [e["id"] for e in result["data"]] == [eid], result
    request.query = {}
    db.close()
    print("✓ 17. Panel 非法 FTS 语法退回字面搜索，仍保留参与者筛选")


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
    await test_9_group_same_second_and_late_reply()
    await test_10_idle_chunks_and_repair()
    await test_11_partial_invalid_is_atomic()
    await test_12_provider_switch_rebuilds_only_vectors()
    await test_13_semantic_duplicate_skipped()
    await test_14_embedding_selection_is_strict()
    await test_15_late_assistant_idle_flush()
    await test_16_invalid_extract_provider_never_falls_back()
    await test_17_panel_invalid_fts_query_falls_back()

    print()
    print("========== ALL 17 PHASE 1F-FIX TESTS PASSED ==========")


if __name__ == "__main__":
    asyncio.run(main())
