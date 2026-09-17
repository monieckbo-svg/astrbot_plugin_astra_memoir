"""
pipeline/writer.py — 事件落库

改动（Phase 1f-fix，恢复 DESIGN §5.4）:

一次 write_batch 的流程:
1. 从 source_raw_ids 回查 raws，二次校验（session_id 匹配 + row 数 = ids 数）
2. 批量生成整批 events 的 embedding
3. 单个 SQLite transaction 里写：
       episodes → participants → keywords → FTS → mark_processed(raws)
4. transaction commit 后，逐条写 vec:
       vec 写失败**不删除 episode** —— SQLite 是真库，vec 是可重建索引
       后续 reindex 可以补齐（VecStore.missing_episode_ids）

也就是说，events 落库和 raw processed 是一体事务。events=[] 也应该原子标 processed，
这一步由 scheduler 处理（本模块不管；本模块的 mark_processed 只处理有事件的情况）。
"""
from __future__ import annotations

import time
import math
import struct
from datetime import datetime, timezone

from astrbot.api import logger

from ..storage import MemoirDB, VecStore
from .extractor import ExtractedEvent


class WriteBatchError(Exception):
    """整批写入失败，raw 保持 unprocessed。"""
    pass


class EpisodeWriter:
    """事件写入器 + embedding 生成。"""

    def __init__(
        self,
        embedding_provider,
        db: MemoirDB,
        vec: VecStore,
    ):
        self.embedding_provider = embedding_provider
        self.db = db
        self.vec = vec

    # ---------- Embedding ----------

    async def _embed(self, text: str) -> list[float] | None:
        try:
            embedding = await self.embedding_provider.get_embedding(text)
            if len(embedding) != self.db.embedding_dim:
                raise ValueError(f"embedding dimension {len(embedding)} != {self.db.embedding_dim}")
            return embedding
        except Exception:
            logger.exception("[Memoir] embedding 生成失败")
            return None

    async def reindex_missing_vectors(self, batch_size: int = 50) -> dict:
        """Repair missing vectors in bounded pages; preserve episodes on failures."""
        repaired = failed = 0
        last_id = 0
        while True:
            def _load():
                return self.db.fetchall(
                    "SELECT e.id, e.title, e.content, e.chat_type, e.session_id, e.group_id, e.is_archived "
                    "FROM episodes e LEFT JOIN episode_vec v ON v.episode_id = e.id "
                    "WHERE v.episode_id IS NULL AND e.id > ? ORDER BY e.id LIMIT ?",
                    (last_id, batch_size),
                )
            rows = await self.db.run(_load)
            if not rows:
                break
            for row in rows:
                last_id = row["id"]
                embedding = await self._embed(f"{row['title']}\n{row['content']}")
                if embedding is None:
                    failed += 1
                    continue
                try:
                    await self.db.run(lambda row=row, embedding=embedding: self.vec.upsert(
                        row["id"], embedding, chat_type=row["chat_type"],
                        session_id=row["session_id"], group_id=row["group_id"],
                        is_archived=row["is_archived"],
                    ))
                    repaired += 1
                except Exception:
                    logger.exception("[Memoir] vector repair failed for episode %d", row["id"])
                    failed += 1
        return {"repaired": repaired, "failed": failed}

    # ---------- write_batch ----------

    async def write_batch(
        self,
        events: list[ExtractedEvent],
        raw_ids_to_mark: list[int],
        *,
        session_id: str,
        chat_type: str,
        group_id: str | None,
        platform: str,
    ) -> list[int]:
        """
        原子写入一整批事件 + 标 raw processed。

        返回新写入的 episode_id 列表。
        任一步失败 → 抛 WriteBatchError，scheduler 不 mark_processed 走重试。

        events=[] 时**不调用本方法**（scheduler 直接标 processed），
        本方法专门处理"有事件"的情况。
        """
        if not events:
            raise WriteBatchError("write_batch called with empty events; scheduler should handle events=[] directly")

        # 1. 从 source_raw_ids 回查 raws（一次拉齐），二次校验
        all_source_ids = sorted({rid for ev in events for rid in ev.source_raw_ids})

        def _load_raws():
            placeholders = ",".join("?" * len(all_source_ids))
            return self.db.fetchall(
                f"SELECT id, session_id, speaker_id, speaker_name, role, created_at "
                f"FROM recent_messages WHERE id IN ({placeholders})",
                all_source_ids,
            )
        raws = await self.db.run(_load_raws)

        if len(raws) != len(all_source_ids):
            found = {r["id"] for r in raws}
            missing = [i for i in all_source_ids if i not in found]
            raise WriteBatchError(
                f"source_raw_ids not found in recent_messages: {missing}"
            )

        # 二次校验：所有 source raw 必须属于本 session（防幻觉跨 session）
        wrong_session = [
            r["id"] for r in raws if r["session_id"] != session_id
        ]
        if wrong_session:
            raise WriteBatchError(
                f"source_raw_ids belong to other session, refused: {wrong_session}"
            )

        raws_by_id = {r["id"]: r for r in raws}

        # importance=1 不进长期 episode，但整批 raw 仍在事务里标 processed。
        events = [ev for ev in events if ev.importance > 1]
        if not events:
            await self.db.run(lambda: self.db.mark_processed(raw_ids_to_mark, int(time.time())))
            return []

        # 2. 批量生成 embedding（并发）
        import asyncio
        embed_tasks = [
            self._embed(f"{ev.title}\n{ev.content}") for ev in events
        ]
        embeddings = await asyncio.gather(*embed_tasks, return_exceptions=False)

        # 至少一个失败 → 不落库（避免半推半就的批次）
        # V1 保守：全部成功才落
        if any(e is None for e in embeddings):
            failed_idx = [i for i, e in enumerate(embeddings) if e is None]
            raise WriteBatchError(
                f"embedding failed for event indices: {failed_idx}"
            )

        # 3. 单 transaction: episodes + participants + keywords + FTS + mark_processed
        extracted_at = int(time.time())
        prepared: list[dict] = []  # 每个 event 的 participants + times
        for ev in events:
            ev_raws = [raws_by_id[i] for i in ev.source_raw_ids]
            parts_map: dict[str, dict] = {}
            for r in ev_raws:
                sid = str(r["speaker_id"])
                parts_map[sid] = {
                    "speaker_id": sid,
                    "speaker_name": r["speaker_name"],
                    "role": r["role"],
                }
            participants = list(parts_map.values())
            times = [int(r["created_at"]) for r in ev_raws]
            prepared.append({
                "participants": participants,
                "event_start_at": min(times),
                "event_end_at": max(times),
            })

        def _tx_write():
            written: list[tuple[int, int]] = []
            accepted_embeddings: list[list[float]] = []
            with self.db.transaction():
                for index, (ev, pp, emb) in enumerate(zip(events, prepared, embeddings)):
                    packed = struct.pack(f"{len(emb)}f", *emb)
                    duplicate = self.db.fetchone(
                        "SELECT e.id FROM episodes e JOIN episode_vec v ON v.episode_id = e.id "
                        "WHERE e.session_id = ? "
                        "AND vec_distance_cosine(v.embedding, ?) <= 0.025 "
                        "ORDER BY e.id DESC LIMIT 1",
                        (session_id, packed),
                    )
                    if duplicate:
                        # 新事件和旧事件高度相同，视为再次提及；不是因误召回而强化。
                        self.db.execute(
                            "UPDATE episodes SET reinforcement_count = reinforcement_count + 1, "
                            "last_reinforced_at = ?, is_archived = 0, importance = MAX(importance, ?) "
                            "WHERE id = ?",
                            (datetime.now(timezone.utc).isoformat(), ev.importance, duplicate["id"]),
                        )
                        self.db.execute(
                            "UPDATE episode_vec SET is_archived = 0 WHERE episode_id = ?",
                            (duplicate["id"],),
                        )
                        logger.info("[Memoir] repeated episode reinforced: %s", ev.title)
                        continue
                    if any(
                        1 - sum(a*b for a, b in zip(emb, old)) /
                        (math.sqrt(sum(a*a for a in emb) * sum(b*b for b in old)) or 1) <= 0.025
                        for old in accepted_embeddings
                    ):
                        logger.info("[Memoir] duplicate episode skipped: %s", ev.title)
                        continue
                    eid = self.db.insert_episode(
                        platform=platform,
                        chat_type=chat_type,
                        session_id=session_id,
                        group_id=group_id,
                        title=ev.title,
                        content=ev.content,
                        source_raw_ids=ev.source_raw_ids,
                        event_start_at=pp["event_start_at"],
                        event_end_at=pp["event_end_at"],
                        extracted_at=extracted_at,
                        importance=ev.importance,
                    )
                    self.db.insert_participants(eid, pp["participants"])
                    self.db.insert_keywords(eid, ev.keywords)
                    self.db.insert_fts(eid, ev.title, ev.content, ev.keywords)
                    written.append((eid, index))
                    accepted_embeddings.append(emb)
                # 同一 transaction 里标 raw processed
                self.db.mark_processed(raw_ids_to_mark, extracted_at)
            return written

        written = await self.db.run(_tx_write)
        episode_ids = [eid for eid, _ in written]

        # 4. transaction 外写 vec —— 失败不删 episode
        for eid, index in written:
            emb = embeddings[index]
            def _vec_upsert(eid=eid, emb=emb):
                self.vec.upsert(
                    eid, emb,
                    chat_type=chat_type, session_id=session_id, group_id=group_id,
                )
            try:
                await self.db.run(_vec_upsert)
            except Exception:
                logger.exception(
                    "[Memoir] vec upsert failed for episode %d — episode 保留，可用 reindex 后补",
                    eid,
                )

        logger.info(
            "[Memoir] wrote %d episode(s), marked %d raw(s) processed",
            len(episode_ids), len(raw_ids_to_mark),
        )
        return episode_ids
