"""importance、衰减、归档、强化与旧库迁移回归测试。"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import struct
import tempfile
import time
import types
from datetime import datetime, timezone
from pathlib import Path
import sqlite_vec

from retriever_test import MemoirDB, VecStore, Retriever, RetrieverConfig, ThemeEmbedding
from astrbot_plugin_astra_memoir.pipeline.extractor import ExtractedEvent, parse_llm_response
from astrbot_plugin_astra_memoir.pipeline.writer import EpisodeWriter
from astrbot_plugin_astra_memoir.pipeline import panel as panel_mod
from unittest.mock import patch


async def main():
    now = int(time.time())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.db"
        old = sqlite3.connect(path)
        old.execute(
            "CREATE TABLE episodes (id INTEGER PRIMARY KEY, platform TEXT, chat_type TEXT, "
            "session_id TEXT, group_id TEXT, title TEXT, content TEXT, source_raw_ids TEXT, "
            "event_start_at INTEGER, event_end_at INTEGER, extracted_at INTEGER, last_accessed_at INTEGER)"
        )
        old.execute(
            "INSERT INTO episodes VALUES (1, 'qq', 'private', 'old', NULL, '旧记忆', '内容', "
            "'[]', ?, ?, ?, NULL)", (now, now, now),
        )
        old.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        old.executemany("INSERT INTO meta VALUES (?, ?)",
                        [("embedding_dim", "8"), ("embedding_provider_id", "")])
        old.enable_load_extension(True)
        sqlite_vec.load(old)
        old.enable_load_extension(False)
        old.execute(
            "CREATE VIRTUAL TABLE episode_vec USING vec0(episode_id INTEGER PRIMARY KEY, "
            "chat_type TEXT, session_id TEXT, group_id TEXT, embedding FLOAT[8] distance_metric=cosine)"
        )
        old.execute("INSERT INTO episode_vec VALUES (?, ?, ?, ?, ?)",
                    (1, "private", "old", "", struct.pack("8f", 1, 0, 0, 0, 0, 0, 0, 0)))
        old.commit()
        old.close()

        db = MemoirDB(path, embedding_dim=8)
        db.initialize()
        try:
            old_ep = db.fetchone("SELECT * FROM episodes WHERE id = 1")
            assert old_ep["importance"] == 2 and old_ep["is_archived"] == 0
            assert old_ep["reinforcement_count"] == 0
            assert "is_archived" in {row["name"] for row in db.fetchall("PRAGMA table_info(episode_vec)")}
            assert db.fetchone("SELECT COUNT(*) FROM episode_vec")[0] == 0

            vec = VecStore(db)
            embed = ThemeEmbedding()
            writer = EpisodeWriter(embed, db, vec)
            retr = Retriever(embed, db, vec, RetrieverConfig(top_k=99))
            assert (await writer.reindex_missing_vectors())["repaired"] == 1

            def raw(message_id: str, content: str, at: int = now) -> int:
                return db.insert_raw_message(
                    dedupe_key=message_id, platform="qq", chat_type="private",
                    session_id="priv", group_id=None, platform_message_id=message_id,
                    speaker_id="111", speaker_name="陆忱", role="user",
                    content=content, reply_to_id=None, trigger_raw_id=None, created_at=at,
                )

            rid1 = raw("trivial", "发了个表情包")
            skipped = await writer.write_batch(
                [ExtractedEvent("表情包", "发了表情包", [rid1], [], importance=1)],
                [rid1], session_id="priv", chat_type="private", group_id=None, platform="qq",
            )
            assert skipped == []
            assert db.fetchone("SELECT processed_at FROM recent_messages WHERE id = ?", (rid1,))[0]
            assert db.fetchone("SELECT COUNT(*) FROM episodes")[0] == 1

            parsed = parse_llm_response(
                json.dumps({"events": [{"title": "日常画图", "content": "普通请求", "importance": 1,
                    "source_raw_ids": [rid1], "keywords": []}]}),
                new_msg_ids={rid1}, overlap_msg_ids=set(),
            )
            assert parsed.success and parsed.events[0].importance == 1
            invalid = parse_llm_response(
                json.dumps({"events": [{"title": "坏", "content": "事", "importance": 8,
                    "source_raw_ids": [rid1]}]}),
                new_msg_ids={rid1}, overlap_msg_ids=set(),
            )
            assert not invalid.success

            rid2 = raw("short", "短期插件问题", now - 31 * 86400)
            short = await writer.write_batch(
                [ExtractedEvent("插件报错讨论", "插件报错了", [rid2], [], importance=2)],
                [rid2], session_id="priv", chat_type="private", group_id=None, platform="qq",
            )
            assert len(short) == 1
            eid = short[0]
            assert db.archive_expired_short_term(now) == 1
            assert db.fetchone("SELECT is_archived FROM episodes WHERE id = ?", (eid,))[0] == 1
            assert db.fts_search("插件", include_session_id="priv") == []
            qvec = await embed.get_embedding("插件")
            assert eid not in {i for i, _ in vec.knn(qvec, include_session_id="priv")}
            assert db.fetchone("SELECT id FROM episodes WHERE id = ?", (eid,)) is not None
            panel = object.__new__(panel_mod.MemoirPanel)
            panel.db = db
            with patch.object(panel_mod, "astr_request", types.SimpleNamespace(query={"archived": "1"})):
                archived_list = await panel.list_episodes()
            assert eid in {ep["id"] for ep in archived_list["data"]}

            vec.delete(eid)
            repair = await writer.reindex_missing_vectors()
            assert repair["repaired"] >= 1
            assert eid not in {i for i, _ in vec.knn(qvec, include_session_id="priv")}

            rid3 = raw("fresh", "插件报错讨论再次提到")
            duplicate = await writer.write_batch(
                [ExtractedEvent("插件报错讨论", "插件报错了", [rid3], [], importance=3)],
                [rid3], session_id="priv", chat_type="private", group_id=None, platform="qq",
            )
            assert duplicate == []
            revived = db.fetchone("SELECT * FROM episodes WHERE id = ?", (eid,))
            assert revived["is_archived"] == 0 and revived["reinforcement_count"] == 1
            assert revived["importance"] == 3 and revived["last_reinforced_at"]
            assert eid in {i for i, _ in vec.knn(qvec, include_session_id="priv")}

            results = await retr.recall("插件报错讨论", session_id="priv", chat_type="private",
                                        group_id=None, current_speaker_id="111")
            assert eid in {r.episode_id for r in results}
            assert len(results) <= 5
            assert db.fetchone("SELECT reinforcement_count FROM episodes WHERE id = ?", (eid,))[0] == 1
            await retr.reinforce_explicit_mentions("插件报错讨论", results)
            assert db.fetchone("SELECT reinforcement_count FROM episodes WHERE id = ?", (eid,))[0] == 2
            await retr.reinforce_explicit_mentions("这个怎么样", results)
            assert db.fetchone("SELECT reinforcement_count FROM episodes WHERE id = ?", (eid,))[0] == 2

            for importance, expected in ((2, 0.0), (3, 0.2), (4, 0.6), (5, 1.0)):
                factor = retr.decay_factor({"importance": importance,
                    "last_reinforced_at": None, "event_end_at": now - 200 * 86400}, now)
                assert factor == expected, (importance, factor)
            fresh = retr.decay_factor({"importance": 2,
                "last_reinforced_at": datetime.now(timezone.utc).isoformat(),
                "event_end_at": now - 200 * 86400}, now)
            assert fresh > 0.99

            batch_ids = []
            for index in range(2):
                other = db.insert_episode(
                    platform="qq", chat_type="group", session_id="grp",
                    group_id="g1", title=f"旧短期事件{index}", content="旧内容",
                    source_raw_ids=[], event_start_at=now - 31 * 86400,
                    event_end_at=now - 31 * 86400, extracted_at=now,
                    importance=2,
                )
                vec.upsert(other, qvec, chat_type="group", session_id="grp", group_id="g1")
                batch_ids.append(other)
            assert db.archive_expired_short_term(now) == 2
            assert all(db.fetchone("SELECT is_archived FROM episode_vec WHERE episode_id = ?", (i,))[0] == 1
                       for i in batch_ids)
        finally:
            db.close()
        # 再次启动不能把新版评分过的记忆从 3 重评为 2。
        reopened = MemoirDB(path, embedding_dim=8)
        reopened.initialize()
        try:
            assert reopened.fetchone("SELECT importance FROM episodes WHERE id = 1")[0] == 2
            assert reopened.fetchone("SELECT importance FROM episodes WHERE id = ?", (eid,))[0] == 3
        finally:
            reopened.close()
    print("Memory lifecycle test passed")


if __name__ == "__main__":
    asyncio.run(main())
