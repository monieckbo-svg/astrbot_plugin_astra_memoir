"""
storage/vec_store.py — sqlite-vec 封装

vec0 虚拟表 episode_vec 已在 db.py 的 initialize() 里创建（含 metadata 列）。

cosine_distance 范围 [0, 2]（0=完全相同,1=正交,2=完全相反）
sqlite-vec 里 metric=cosine 时返回值就是 1 - cosine_similarity，
所以 similarity = 1 - distance。
代码里统一使用 distance 概念，避免混用。
"""
from __future__ import annotations

import struct

from .db import MemoirDB


def _pack_f32(vec: list[float]) -> bytes:
    """把 python list[float] 打包成 sqlite-vec 需要的 float32 bytes。"""
    return struct.pack(f"{len(vec)}f", *vec)


class VecStore:
    """
    对 episode_vec 虚拟表的操作封装。
    """

    def __init__(self, db: MemoirDB):
        self.db = db

    def upsert(
        self, episode_id: int, embedding: list[float],
        *, chat_type: str, session_id: str, group_id: str | None,
    ) -> None:
        """
        插入或更新一条 episode 的向量，携带 metadata 用于 KNN 阶段过滤。
        vec0 metadata 列不接 NULL，None → ""。
        """
        packed = _pack_f32(embedding)
        self.db.execute(
            "DELETE FROM episode_vec WHERE episode_id = ?", (episode_id,)
        )
        self.db.execute(
            "INSERT INTO episode_vec(episode_id, chat_type, session_id, group_id, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode_id, chat_type, session_id, group_id or "", packed),
        )

    def delete(self, episode_id: int) -> None:
        self.db.execute(
            "DELETE FROM episode_vec WHERE episode_id = ?", (episode_id,)
        )

    def knn(
        self,
        query_embedding: list[float],
        k: int = 20,
        *,
        include_session_id: str | None = None,
        include_group_id: str | None = None,
        include_all_groups: bool = False,
    ) -> list[tuple[int, float]]:
        """
        KNN 检索 top-k，返回 [(episode_id, cosine_distance), ...]，距离升序。

        可见性用 vec0 metadata 列在 KNN 阶段做，不是后过滤。

        典型场景:
        - 私聊场景: include_session_id=当前私聊, include_all_groups=True
          → 两路 KNN 各取 k，再 merge（vec0 一次 query 不支持 OR，需要拆两次）
        - 群聊场景: include_group_id=当前群
          → 单路 KNN with WHERE
        """
        # 无可见性 → 全库 KNN
        no_scope = (
            include_session_id is None
            and include_group_id is None
            and not include_all_groups
        )
        if no_scope:
            return self._knn_where(query_embedding, k, extra_where="", params=[])

        results: dict[int, float] = {}

        # 私聊 own session（含匿名 session 完全等于此 id 的）
        if include_session_id is not None:
            for eid, dist in self._knn_where(
                query_embedding, k,
                extra_where="AND chat_type = 'private' AND session_id = ?",
                params=[include_session_id],
            ):
                if eid not in results or dist < results[eid]:
                    results[eid] = dist

        # 所有群
        if include_all_groups:
            for eid, dist in self._knn_where(
                query_embedding, k,
                extra_where="AND chat_type = 'group'",
                params=[],
            ):
                if eid not in results or dist < results[eid]:
                    results[eid] = dist
        # 单群
        elif include_group_id is not None:
            for eid, dist in self._knn_where(
                query_embedding, k,
                extra_where="AND chat_type = 'group' AND group_id = ?",
                params=[include_group_id],
            ):
                if eid not in results or dist < results[eid]:
                    results[eid] = dist

        # 距离升序取 top k
        sorted_r = sorted(results.items(), key=lambda x: x[1])
        return sorted_r[:k]

    def _knn_where(
        self, query_embedding: list[float], k: int,
        *, extra_where: str, params: list,
    ) -> list[tuple[int, float]]:
        """单次 vec0 KNN with metadata filter。"""
        packed = _pack_f32(query_embedding)
        sql = (
            "SELECT episode_id, distance FROM episode_vec "
            f"WHERE embedding MATCH ? AND k = ? {extra_where} "
            "ORDER BY distance"
        )
        rows = self.db.fetchall(sql, tuple([packed, k, *params]))
        return [(r["episode_id"], r["distance"]) for r in rows]

    def count(self) -> int:
        row = self.db.fetchone("SELECT COUNT(*) AS n FROM episode_vec")
        return row["n"] if row else 0

    def missing_episode_ids(self, limit: int = 100) -> list[int]:
        """
        找 episodes 表里存在但 vec 表里缺失的 episode id。
        用于 vec 写失败后的 reindex。
        """
        rows = self.db.fetchall(
            "SELECT e.id FROM episodes e "
            "LEFT JOIN episode_vec v ON v.episode_id = e.id "
            "WHERE v.episode_id IS NULL "
            "ORDER BY e.id LIMIT ?",
            (limit,),
        )
        return [r["id"] for r in rows]
