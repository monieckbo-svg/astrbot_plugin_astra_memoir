"""
retriever_test.py — Phase 1e 8 个必测场景

覆盖 GPT 反馈第 11 条：
  1. 私聊能召回同一 private session 的旧事件
  2. 私聊能召回群聊事件
  3. 私聊不能召回别人的 private
  4. 群聊不能召回任何 private
  5. participant 不在事件里但语义相关仍能召回
  6. 完全无关 query 返回空
  7. 同一 episode 同时被 vec 和 FTS 命中只出现一次
  8. 低相似度 KNN 候选被 relevance gate 拦掉
  9. 最终注入内容不含技术 metadata（本 test 补的）
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


# ---- mock astrbot（同 integration_test）----
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

    # TextPart mock 需要跟真实一致：有 text 属性 + mark_as_temp() 方法
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
Retriever = pipeline_mod.Retriever
RetrieverConfig = pipeline_mod.RetrieverConfig


# ---- 简单 embedding：每个"主题"返回固定向量，同主题相似，跨主题正交 ----
class ThemeEmbedding:
    """按主题词表做 embedding。同主题 cosine≈0，跨主题≈1."""
    THEMES = {
        "石锅拌饭": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "午饭": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "吃": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "插件": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "报错": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "感冒": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "小雨": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "生病": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "天气": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "宇宙": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 完全无关
    }
    id = "theme"

    async def get_embedding(self, text: str):
        # 找 text 里出现的主题词，把它们的向量相加然后 normalize
        v = [0.0] * 8
        matched = False
        for keyword, vec in self.THEMES.items():
            if keyword in text:
                v = [a + b for a, b in zip(v, vec)]
                matched = True
        if not matched:
            # 完全匹配不到主题 → 放到一个独立正交轴 dim=7，
            # 跟所有 episode 的 cosine distance = 1.0（>阈值 0.9）
            v = [0.0] * 8
            v[7] = 1.0
        norm = sum(x*x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]


class FakeContext:
    def __init__(self, embed):
        self._embed = embed
    def get_all_embedding_providers(self):
        return [self._embed]


# ============================================================
# 数据准备
# ============================================================

async def prepare_data(db: MemoirDB, vec: VecStore, embed: ThemeEmbedding):
    """
    准备 5 条 episode:
      E1: 陆忱私聊 - 午饭石锅拌饭             (private, session=priv_A, participant=陆忱111,星星999)
      E2: 群聊 - 插件报错讨论                  (group, session=grp, group_id=g_main, participant=陆忱111,小雨222)
      E3: 群聊 - 小雨感冒                       (group, session=grp, group_id=g_main, participant=小雨222)
                                                （陆忱和星星都没参与，测 participant 不过滤）
      E4: 别人的私聊 - 天气                    (private, session=priv_B, participant=其他人333)
                                                （测私聊隔离）
      E5: 群聊 - 宇宙话题                       (group, session=grp, group_id=g_main, participant=陆忱111)
                                                （测无关 query 被 relevance gate 拦掉）
    """
    now = int(time.time())

    async def _mk_episode(
        *, title, content, chat_type, session_id, group_id, participants, keywords,
    ):
        rid = db.insert_raw_message(
            dedupe_key=f"user:{session_id}:{title}",
            platform="qq", chat_type=chat_type,
            session_id=session_id, group_id=group_id,
            platform_message_id=title,
            speaker_id=participants[0]["speaker_id"],
            speaker_name=participants[0]["speaker_name"],
            role="user",
            content=content, reply_to_id=None, trigger_raw_id=None,
            created_at=now,
        )
        with db.transaction():
            eid = db.insert_episode(
                platform="qq", chat_type=chat_type, session_id=session_id,
                group_id=group_id, title=title, content=content,
                source_raw_ids=[rid], event_start_at=now, event_end_at=now,
                extracted_at=now,
            )
            db.insert_participants(eid, participants)
            db.insert_keywords(eid, keywords)
            db.insert_fts(eid, title, content, keywords)
        v = await embed.get_embedding(f"{title}\n{content}")
        vec.upsert(eid, v, chat_type=chat_type, session_id=session_id, group_id=group_id)
        return eid

    e1 = await _mk_episode(
        title="午饭吃了石锅拌饭",
        content="陆忱中午吃了石锅拌饭和奶茶",
        chat_type="private", session_id="priv_A", group_id=None,
        participants=[
            {"speaker_id":"111","speaker_name":"陆忱","role":"user"},
            {"speaker_id":"999","speaker_name":"星星","role":"assistant"},
        ],
        keywords=["石锅拌饭","午饭"],
    )
    e2 = await _mk_episode(
        title="插件报错讨论",
        content="陆忱在群里问插件报错，小雨提出重装",
        chat_type="group", session_id="grp", group_id="g_main",
        participants=[
            {"speaker_id":"111","speaker_name":"陆忱","role":"user"},
            {"speaker_id":"222","speaker_name":"小雨","role":"user"},
        ],
        keywords=["插件","报错"],
    )
    e3 = await _mk_episode(
        title="小雨感冒了",
        content="小雨在群里说自己感冒了在家休息",
        chat_type="group", session_id="grp", group_id="g_main",
        participants=[{"speaker_id":"222","speaker_name":"小雨","role":"user"}],
        keywords=["小雨","感冒"],
    )
    e4 = await _mk_episode(
        title="别人的私聊天气",
        content="别人在私聊里说今天天气不错",
        chat_type="private", session_id="priv_B", group_id=None,
        participants=[{"speaker_id":"333","speaker_name":"别人","role":"user"}],
        keywords=["天气"],
    )
    e5 = await _mk_episode(
        title="宇宙话题",
        content="群里聊了一会儿宇宙起源",
        chat_type="group", session_id="grp", group_id="g_main",
        participants=[{"speaker_id":"111","speaker_name":"陆忱","role":"user"}],
        keywords=["宇宙"],
    )
    return e1, e2, e3, e4, e5


# ============================================================
# Main test
# ============================================================

async def main():
    with tempfile.TemporaryDirectory() as tmp:
        db = MemoirDB(Path(tmp)/"t.db", embedding_dim=8)
        db.initialize()
        vec = VecStore(db)
        embed = ThemeEmbedding()
        context = FakeContext(embed)
        cfg = RetrieverConfig(top_k=5, max_cosine_distance=0.9)
        retr = Retriever(context, db, vec, cfg, embedding_provider_id="")

        e1, e2, e3, e4, e5 = await prepare_data(db, vec, embed)
        print(f"prepared episodes: e1={e1} e2={e2} e3={e3} e4={e4} e5={e5}")

        # === 1. 私聊能召回同一 private session 的旧事件 ===
        r = await retr.recall(
            "我今天吃了什么", session_id="priv_A", chat_type="private",
            group_id=None, current_speaker_id="111",
        )
        ids = [x.episode_id for x in r]
        assert e1 in ids, f"私聊应召回本会话午饭事件，got {ids}"
        print(f"✓ 场景 1: 私聊召回本会话旧事件 ({ids})")

        # === 2. 私聊能召回群聊事件 ===
        r = await retr.recall(
            "小雨怎么了", session_id="priv_A", chat_type="private",
            group_id=None, current_speaker_id="111",
        )
        ids = [x.episode_id for x in r]
        assert e3 in ids, f"私聊应召回群聊'小雨感冒'，got {ids}"
        print(f"✓ 场景 2: 私聊召回群聊事件 ({ids})")

        # === 3. 私聊不能召回别人的 private ===
        r = await retr.recall(
            "天气", session_id="priv_A", chat_type="private",
            group_id=None, current_speaker_id="111",
        )
        ids = [x.episode_id for x in r]
        assert e4 not in ids, f"私聊不应召回别人的 private，got {ids}"
        print(f"✓ 场景 3: 私聊隔离别人的 private (query='天气', got {ids})")

        # === 4. 群聊不能召回任何 private ===
        r = await retr.recall(
            "石锅拌饭 天气", session_id="grp", chat_type="group",
            group_id="g_main", current_speaker_id="222",
        )
        ids = [x.episode_id for x in r]
        assert e1 not in ids, f"群聊不应召回 e1，got {ids}"
        assert e4 not in ids, f"群聊不应召回 e4，got {ids}"
        print(f"✓ 场景 4: 群聊隔离所有 private ({ids})")

        # === 5. participant 不在事件里但语义相关仍能召回 ===
        #   陆忱(111) 没参与 e3（小雨感冒），但语义相关应能召回
        r = await retr.recall(
            "感冒", session_id="priv_A", chat_type="private",
            group_id=None, current_speaker_id="111",
        )
        ids = [x.episode_id for x in r]
        assert e3 in ids, f"participant 不该硬过滤，got {ids}"
        print(f"✓ 场景 5: participant 不在事件仍能召回 (陆忱查'感冒' → e3={e3})")

        # === 6. 完全无关 query 返回空（relevance gate） ===
        r = await retr.recall(
            "亲亲抱抱",  # 无主题词命中 → embedding 远离所有 episode，FTS 也不匹配
            session_id="priv_A", chat_type="private",
            group_id=None, current_speaker_id="111",
        )
        assert r == [], f"无关 query 应返回空，got {[(x.episode_id, x.title) for x in r]}"
        print(f"✓ 场景 6: 无关 query 被 relevance gate 拦掉，返回空")

        # === 7. 同一 episode 同时被 vec 和 FTS 命中只出现一次 ===
        r = await retr.recall(
            "石锅拌饭",  # 主题词命中 e1 + FTS 命中 e1
            session_id="priv_A", chat_type="private",
            group_id=None, current_speaker_id="111",
        )
        ids = [x.episode_id for x in r]
        assert ids.count(e1) == 1, f"e1 应只出现一次，got {ids}"
        # 验证 debug 字段确实两路都命中了
        for x in r:
            if x.episode_id == e1:
                assert x.debug_fts_hit, "e1 应有 FTS 命中"
                assert x.debug_vec_distance is not None and x.debug_vec_distance < 0.5, \
                    "e1 应有向量命中"
        print(f"✓ 场景 7: vec + FTS 双命中的 e1 只出现一次")

        # === 8. 低相似度 KNN 候选被 relevance gate 拦掉 ===
        #   即使 vec KNN 返回了远距离候选，也不能进入最终结果
        r = await retr.recall(
            "亲亲抱抱", session_id="priv_A", chat_type="private",
            group_id=None, current_speaker_id="111",
        )
        # 已经在场景 6 里断言过返回空。这里再显式检查候选池被 gate 拦掉。
        # 直接调 vec.knn 看候选池确实存在
        q_vec = await embed.get_embedding("亲亲抱抱")
        raw_vec_hits = vec.knn(q_vec, k=10,
                               include_session_id="priv_A", include_all_groups=True)
        assert raw_vec_hits, "KNN 底层应能返回候选（即使距离远）"
        min_dist = min(d for _, d in raw_vec_hits)
        print(f"✓ 场景 8: KNN 返回候选(最近距离={min_dist:.3f})但被 gate 拦掉 (阈值={cfg.max_cosine_distance})")

        # === 9. 注入内容不含技术 metadata ===
        r = await retr.recall(
            "小雨怎么了", session_id="priv_A", chat_type="private",
            group_id=None, current_speaker_id="111",
        )
        text = retr.format_injection(r)
        print()
        print("--- 注入文本预览 ---")
        print(text)
        print("--- END ---")
        print()

        # 硬检查：注入文本里不能出现任何技术字段
        forbidden = ["raw_id", "session_id", "episode_id", "rrf", "cosine",
                     "distance", "embedding", "speaker_id", "qq=", "111", "222", "333",
                     "priv_A", "grp"]
        # 允许 QQ 号在 participants 名字里出现吗？不允许——我们只输出昵称
        # 但注意 "111", "222" 这些数字可能误伤日期，先排除数字位串
        # 只检查真正的字段名
        text_lower = text.lower()
        for kw in ["raw_id", "rrf", "cosine", "distance", "embedding",
                   "speaker_id", "session_id"]:
            assert kw.lower() not in text_lower, f"注入文本不该含 {kw!r}: {text}"
        # QQ 号也不该出现
        for qq in ["111", "222", "333", "999"]:
            assert qq not in text, f"注入文本不该含 QQ 号 {qq!r}"
        print(f"✓ 场景 9: 注入文本不含 raw_id / QQ 号 / 内部字段")

        # === 10. inject 路径 e2e：无命中不塞空 block ===
        class FakeReq:
            def __init__(self):
                self.extra_user_content_parts = []
        class FakeEvent:
            def __init__(self, message_str, sender_id, session, group_id=None):
                self._msg = message_str
                self._sid = sender_id
                self._sess = session
                self._gid = group_id or ""
            def get_message_str(self): return self._msg
            def get_sender_id(self): return self._sid
            def get_group_id(self): return self._gid
            @property
            def unified_msg_origin(self): return self._sess

        # 有命中
        req = FakeReq()
        await retr.inject(
            FakeEvent("小雨怎么了", "111", "priv_A"), req,
        )
        assert len(req.extra_user_content_parts) == 1
        assert req.extra_user_content_parts[0]._no_save, "mark_as_temp 未生效"
        assert "小雨感冒" in req.extra_user_content_parts[0].text
        print(f"✓ 场景 10a: inject 命中 → append TextPart + mark_as_temp")

        # 无命中
        req2 = FakeReq()
        await retr.inject(
            FakeEvent("亲亲抱抱", "111", "priv_A"), req2,
        )
        assert req2.extra_user_content_parts == [], \
            f"无命中不该 append，got {req2.extra_user_content_parts}"
        print(f"✓ 场景 10b: inject 无命中 → 不 append，不塞空 block")

        # 空 query 也不 append
        req3 = FakeReq()
        await retr.inject(FakeEvent("", "111", "priv_A"), req3)
        assert req3.extra_user_content_parts == []
        print(f"✓ 场景 10c: 空 query → 不 inject")

        print()
        print("========== ALL 10 RETRIEVER TESTS PASSED ==========")
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
