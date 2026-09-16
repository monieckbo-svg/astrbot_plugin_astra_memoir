"""
pipeline/writer.py — 事件落库

对每个通过校验的 ExtractedEvent：
1. 从 source_raw_ids 回查参与者（participants 由代码回查，不听 LLM）
2. 生成 embedding
3. 单个 transaction 里写 episode + participants + keywords + FTS
4. transaction 外写 vec（vec0 不支持事务嵌套）
5. 全部成功 → 由 scheduler 统一 mark_processed；任一失败抛异常 → raw 保持 unprocessed

见 DESIGN §5.4。
"""
from __future__ import annotations

import time

from astrbot.api import logger
from astrbot.api.star import Context

from ..storage import MemoirDB, VecStore
from .extractor import ExtractedEvent


class EpisodeWriter:
    """事件写入器 + embedding 生成。"""

    def __init__(
        self,
        context: Context,
        db: MemoirDB,
        vec: VecStore,
        embedding_provider_id: str = "",
    ):
        self.context = context
        self.db = db
        self.vec = vec
        self.embedding_provider_id = embedding_provider_id.strip()

    # ---------- Embedding ----------

    def _get_embedding_provider(self):
        """
        - 有配 embedding_provider_id 且能匹配 → 用它
        - 否则用第一个可用的 embedding provider
        """
        if self.embedding_provider_id:
            for prov in self.context.get_all_embedding_providers():
                if getattr(prov, "id", None) == self.embedding_provider_id:
                    return prov
            logger.warning(
                "[Memoir] embedding_provider_id=%r 找不到，回退到第一个可用",
                self.embedding_provider_id,
            )
        providers = self.context.get_all_embedding_providers()
        return providers[0] if providers else None

    async def _embed(self, text: str) -> list[float] | None:
        """生成 embedding，失败返回 None。"""
        prov = self._get_embedding_provider()
        if prov is None:
            logger.error("[Memoir] 没有可用的 embedding provider")
            return None
        try:
            return await prov.get_embedding(text)
        except Exception:
            logger.exception("[Memoir] embedding 生成失败")
            return None

    # ---------- Main write ----------

    async def write_events(
        self,
        events: list[ExtractedEvent],
        *,
        session_id: str,
        chat_type: str,
        group_id: str | None,
        platform: str,
    ) -> tuple[int, int]:
        """
        写入所有 events。返回 (成功条数, 失败条数)。

        events=[] 是合法的（无值得记忆的事件）—— 直接返回 (0, 0)。
        单条失败不影响其他条；由 scheduler 决定是否整批回滚。
        """
        if not events:
            return (0, 0)

        ok = 0
        fail = 0
        for ev in events:
            try:
                await self._write_one(
                    ev,
                    session_id=session_id,
                    chat_type=chat_type,
                    group_id=group_id,
                    platform=platform,
                )
                ok += 1
            except Exception:
                logger.exception(
                    "[Memoir] 写入 episode 失败，跳过: title=%r", ev.title,
                )
                fail += 1
        return (ok, fail)

    async def _write_one(
        self,
        ev: ExtractedEvent,
        *,
        session_id: str,
        chat_type: str,
        group_id: str | None,
        platform: str,
    ) -> int:
        """
        写一条 event。返回 episode_id。

        流程:
        1. 回查 participants + 时间边界（在 SQL 层）
        2. 生成 embedding（异步；vec 插入需要 embedding）
        3. transaction 里: episode + participants + keywords + FTS
        4. transaction 外: vec upsert
        """
        # --- 1. 回查 source raw 的元信息（participants + 时间）---
        def _load_raws():
            placeholders = ",".join("?" * len(ev.source_raw_ids))
            rows = self.db.fetchall(
                f"SELECT id, speaker_id, speaker_name, role, created_at "
                f"FROM recent_messages WHERE id IN ({placeholders})",
                ev.source_raw_ids,
            )
            return rows

        raws = await self.db.run(_load_raws)
        if not raws:
            raise RuntimeError(
                f"source_raw_ids 全部找不到 raw row: {ev.source_raw_ids}"
            )

        # participants: 用 speaker_id 去重，保留最后一次出现的 name
        parts_map: dict[str, dict] = {}
        for r in raws:
            sid = str(r["speaker_id"])
            parts_map[sid] = {
                "speaker_id": sid,
                "speaker_name": r["speaker_name"],
                "role": r["role"],
            }
        participants = list(parts_map.values())

        # event_start / event_end
        times = [int(r["created_at"]) for r in raws]
        event_start_at = min(times)
        event_end_at = max(times)
        extracted_at = int(time.time())

        # --- 2. embedding（用 title + content 拼接做 embedding 输入）---
        embed_text = f"{ev.title}\n{ev.content}"
        embedding = await self._embed(embed_text)
        if embedding is None:
            raise RuntimeError("embedding 生成失败，事件不落库")

        # --- 3. transaction: episode + participants + keywords + FTS ---
        def _tx_write():
            with self.db.transaction():
                eid = self.db.insert_episode(
                    platform=platform,
                    chat_type=chat_type,
                    session_id=session_id,
                    group_id=group_id,
                    title=ev.title,
                    content=ev.content,
                    source_raw_ids=ev.source_raw_ids,
                    event_start_at=event_start_at,
                    event_end_at=event_end_at,
                    extracted_at=extracted_at,
                )
                self.db.insert_participants(eid, participants)
                self.db.insert_keywords(eid, ev.keywords)
                self.db.insert_fts(eid, ev.title, ev.content, ev.keywords)
                return eid

        episode_id = await self.db.run(_tx_write)

        # --- 4. vec upsert（vec0 不在事务里）---
        def _vec_upsert():
            self.vec.upsert(episode_id, embedding)

        try:
            await self.db.run(_vec_upsert)
        except Exception:
            # vec 写失败 → 把刚写的 episode 也回滚
            logger.exception(
                "[Memoir] vec 写入失败，回滚 episode %d", episode_id
            )

            def _rollback():
                with self.db.transaction():
                    self.db.execute(
                        "DELETE FROM episodes WHERE id = ?", (episode_id,)
                    )
                    self.db.execute(
                        "DELETE FROM episodes_fts WHERE rowid = ?", (episode_id,)
                    )
                    # participants/keywords 有 FK CASCADE 会跟着删

            await self.db.run(_rollback)
            raise

        logger.debug(
            "[Memoir] episode %d written: %r (raws=%s, parts=%d)",
            episode_id, ev.title, ev.source_raw_ids, len(participants),
        )
        return episode_id
