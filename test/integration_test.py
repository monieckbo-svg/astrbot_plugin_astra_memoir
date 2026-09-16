"""
integration_test.py — 不依赖 AstrBot 的端到端 mock 测试

用假 provider 模拟 LLM 和 embedding，跑通：
  1. 直接 insert raw （不经过 hook，模拟已经落库的消息）
  2. scheduler.process_batch(session_id) 拆事件 + 落库 + 标 processed
  3. 断言 episode / participants / FTS / vec 都对
  4. 断言 raw 被标 processed

跑法: python3 test/integration_test.py
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
from unittest.mock import MagicMock

# ---- mock astrbot 依赖（本地测试环境没有装 astrbot） ----
def _mock_astrbot():
    api = types.ModuleType("astrbot")
    api_api = types.ModuleType("astrbot.api")
    api_api.logger = logging.getLogger("memoir-test")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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
    class EventMessageType:
        ALL = "ALL"
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

    sys.modules["astrbot"] = api
    sys.modules["astrbot.api"] = api_api
    sys.modules["astrbot.api.event"] = api_event
    sys.modules["astrbot.api.event.filter"] = api_event_filter
    sys.modules["astrbot.api.star"] = api_star
    sys.modules["astrbot.core"] = core
    sys.modules["astrbot.core.provider"] = core_provider
    sys.modules["astrbot.core.provider.entities"] = core_provider_entities

_mock_astrbot()

# 把 memoir 的父目录加到 sys.path，用绝对包名 import
_PKG_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_PKG_DIR.parent))
_PKG = _PKG_DIR.name  # "astrbot_plugin_astra_memoir"

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


# ============================================================
# Mock provider
# ============================================================

class FakeLLMResponse:
    def __init__(self, text: str):
        self.completion_text = text


class FakeLLMProvider:
    """按 prompt 内容返回预设 JSON；source_raw_ids 从 prompt 里出现的 raw_id 自动填。"""

    def __init__(self, event_specs: list[dict]):
        """
        event_specs 每项形如：
          {"match": "石锅拌饭", "title": "...", "content": "...", "keywords": [...], "raw_hits_needed": 1}
        raw_hits_needed 表示这个事件用多少条 raw（依 prompt 里出现顺序）
        """
        self.specs = event_specs
        self.call_count = 0

    async def text_chat(self, prompt: str, system_prompt: str = "", **kwargs):
        self.call_count += 1
        import re
        # 抽 prompt 里 new_messages 块里的 raw_id
        # 简单起见抽所有 [raw_id=N] 并按出现顺序
        raw_ids = [int(x) for x in re.findall(r"\[raw_id=(\d+)\]", prompt)]

        events = []
        for spec in self.specs:
            if spec["match"] in prompt:
                n = spec.get("raw_hits_needed", 1)
                events.append({
                    "title": spec["title"],
                    "content": spec["content"],
                    "source_raw_ids": raw_ids[:n],
                    "keywords": spec.get("keywords", []),
                })
        return FakeLLMResponse(json.dumps({"events": events}, ensure_ascii=False))


class FakeEmbeddingProvider:
    """返回 hash-based 稳定 embedding，同文本同向量。"""

    def __init__(self, dim: int = 8):
        self.id = "fake-embed"
        self.dim = dim
        self.call_count = 0

    async def get_embedding(self, text: str) -> list[float]:
        self.call_count += 1
        # 简单 hash → dim 维向量
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        raw = [b / 255.0 for b in h[: self.dim]]
        # normalize（模拟 cosine 需要）
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]


class FakeContext:
    def __init__(self, llm: FakeLLMProvider, embed: FakeEmbeddingProvider):
        self._llm = llm
        self._embed = embed

    def get_provider_by_id(self, pid: str):
        return self._llm

    def get_using_provider(self, umo=None):
        return self._llm

    def get_all_embedding_providers(self):
        return [self._embed]


# ============================================================
# Test main
# ============================================================

async def main():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memoir.db"
        db = MemoirDB(db_path, embedding_dim=8)
        db.initialize()
        vec = VecStore(db)

        # -------- 场景 1: 私聊，一批里聊了两件事 --------

        # 用 event specs 而非 canned JSON，raw_ids 自动从 prompt 抽
        llm = FakeLLMProvider([
            # 场景 1: 私聊里 4 条 raw，希望拆成 2 个事件
            #   为简化 mock，遇到 "石锅拌饭" 时同时产 2 个事件，各占一半 raw
            {
                "match": "石锅拌饭",
                "title": "午饭吃了石锅拌饭",
                "content": "陆忱中午吃了石锅拌饭和奶茶",
                "keywords": ["石锅拌饭", "午饭"],
                "raw_hits_needed": 2,  # 用前 2 个 raw_id
            },
            # 但同一 prompt 里 mock 只能 return 一次，所以第二个事件需要独立 spec
            # 用不同 match 关键词（这批消息里应包含）
            {
                "match": "插件",
                "title": "插件报错询问",
                "content": "陆忱问那个插件后来怎么样了",
                "keywords": ["插件"],
                "raw_hits_needed": 4,  # 占后 2 个 raw（会被去重），实际全占 4
            },
            {
                "match": "感冒",
                "title": "小雨感冒了",
                "content": "小雨在群里说自己感冒了在家休息",
                "keywords": ["小雨", "感冒"],
                "raw_hits_needed": 1,
            },
        ])
        embed = FakeEmbeddingProvider(dim=8)
        context = FakeContext(llm, embed)

        raw = RawCache(db, bot_display_name="星星")
        extractor = EventExtractor(context, db, extract_provider_id="")
        writer = EpisodeWriter(context, db, vec, embedding_provider_id="")
        cfg = SchedulerConfig(interval_seconds=1, private_batch_user_turns=2)
        scheduler = BatchScheduler(db, extractor, writer, cfg)

        # -------- 私聊数据 --------
        priv_sess = "aiocqhttp:FriendMessage:111"
        raws = [
            ("qq:priv:p1", "user", "111", "陆忱", "p1", "中午吃了石锅拌饭", None),
            ("qq:priv:p2", "assistant", "999", "星星", None, "你吃啥了呀", None),
            ("qq:priv:p3", "user", "111", "陆忱", "p2", "那个插件后来怎么样了", None),
            ("qq:priv:p4", "assistant", "999", "星星", None, "好像还没修", None),
        ]
        now = int(time.time())
        for i, (key, role, sid, sname, pmid, content, trg) in enumerate(raws):
            db.insert_raw_message(
                dedupe_key=key, platform="qq", chat_type="private",
                session_id=priv_sess, group_id=None,
                platform_message_id=pmid,
                speaker_id=sid, speaker_name=sname, role=role,
                content=content, reply_to_id=None,
                trigger_raw_id=trg, created_at=now + i,
            )

        # -------- 触发 process_batch --------
        await scheduler.process_batch(priv_sess)

        # 断言
        print()
        print("=== 私聊场景验收 ===")
        eps = db.fetchall("SELECT * FROM episodes ORDER BY id")
        print(f"episodes: {len(eps)} 条")
        for e in eps:
            print(f"  #{e['id']}: {e['title']} | source_raws={e['source_raw_ids']}")
        assert len(eps) == 2, f"预期 2 条 episode，实际 {len(eps)}"

        # participants: 每条应含陆忱和星星
        for e in eps:
            parts = db.fetchall(
                "SELECT speaker_id, speaker_name, role FROM episode_participants WHERE episode_id = ?",
                (e["id"],)
            )
            sids = {p["speaker_id"] for p in parts}
            print(f"  #{e['id']} participants: {[(p['speaker_id'], p['speaker_name']) for p in parts]}")
            assert "111" in sids, f"陆忱应在 participants，got {sids}"
            assert "999" in sids, f"星星应在 participants，got {sids}"

        # FTS 检索
        r = db.fts_search("石锅拌饭", include_session_id=priv_sess, include_all_groups=True)
        print(f"FTS 石锅拌饭: {r}")
        assert len(r) == 1

        r = db.fts_search("插件", include_session_id=priv_sess, include_all_groups=True)
        print(f"FTS 插件: {r}")
        assert len(r) == 1

        # vec 检索（用相同 embedding provider 拿 query 向量）
        q_vec = await embed.get_embedding("午饭吃了石锅拌饭\n陆忱中午吃了石锅拌饭和奶茶")
        hits = vec.knn(q_vec, k=5)
        print(f"KNN top: {hits[:3]}")
        assert hits[0][1] < 0.001, "同文本 embedding 应该距离≈0"

        # 检查 raw 全被 processed
        unproc = db.fetchall(
            "SELECT * FROM recent_messages WHERE session_id = ? AND processed_at IS NULL",
            (priv_sess,)
        )
        assert not unproc, f"raw 应全被 processed，还有 {len(unproc)} 未处理"
        print(f"✓ 私聊 4 条 raw 全部 processed")

        # -------- 场景 2: 全是寒暄，events=[] 也应标 processed --------
        priv2 = "aiocqhttp:FriendMessage:222"
        db.insert_raw_message(
            dedupe_key="qq:priv2:x1", platform="qq", chat_type="private",
            session_id=priv2, group_id=None, platform_message_id="x1",
            speaker_id="222", speaker_name="B", role="user",
            content="哈哈哈哈", reply_to_id=None, trigger_raw_id=None,
            created_at=now + 100,
        )
        await scheduler.process_batch(priv2)
        unproc2 = db.fetchall(
            "SELECT * FROM recent_messages WHERE session_id = ? AND processed_at IS NULL",
            (priv2,)
        )
        assert not unproc2, "events=[] 时也应把 raw 标 processed"
        print("✓ 场景 2: events=[] 时 raw 仍被 processed（不死循环）")

        # -------- 场景 3: 群聊 speaker 严格区分 --------
        print()
        print("=== 群聊场景验收 ===")
        grp_sess = "aiocqhttp:GroupMessage:g_main"
        db.insert_raw_message(
            dedupe_key="qq:grp:g1", platform="qq", chat_type="group",
            session_id=grp_sess, group_id="g_main", platform_message_id="g1",
            speaker_id="222", speaker_name="小雨", role="user",
            content="我感冒了在家休息", reply_to_id=None, trigger_raw_id=None,
            created_at=now + 200,
        )
        await scheduler.process_batch(grp_sess)
        eps_g = db.fetchall("SELECT * FROM episodes WHERE session_id = ?", (grp_sess,))
        print(f"群聊 episodes: {len(eps_g)}")
        for e in eps_g:
            parts = db.fetchall(
                "SELECT speaker_id, speaker_name FROM episode_participants WHERE episode_id = ?",
                (e["id"],)
            )
            print(f"  #{e['id']}: {e['title']} participants: {[(p['speaker_id'], p['speaker_name']) for p in parts]}")
            sids = {p["speaker_id"] for p in parts}
            assert "222" in sids, "小雨(222) 应在 participants"
            assert "999" not in sids, "星星不应出现（她没参与这条群聊）"
        print("✓ 群聊 speaker 严格区分：星星未参与不进 participants")

        # -------- 场景 4: 私聊召回群聊事件 --------
        r = db.fts_search("感冒", include_session_id=priv_sess, include_all_groups=True)
        assert len(r) == 1, "私聊应能召回群聊'小雨感冒'事件"
        print("✓ 私聊场景可召回群聊事件")

        # 群聊召回私聊：不能
        r = db.fts_search("石锅拌饭", include_group_id="g_main")
        assert r == [], "群聊不应召回私聊事件"
        print("✓ 群聊场景严格隔离私聊事件")

        print()
        print("========== ALL INTEGRATION TESTS PASSED ==========")
        print(f"LLM 调用总数: {llm.call_count} (3 批: 私聊×1 + 寒暄×1 + 群聊×1)")
        print(f"Embedding 调用总数: {embed.call_count}")
        assert llm.call_count == 3
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
