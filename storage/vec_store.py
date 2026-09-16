"""
storage/vec_store.py — sqlite-vec 封装

vec0 虚拟表 episode_vec 已在 db.py 的 initialize() 里创建。
本模块只负责：
- 向量序列化（float32 → bytes）
- 插入 / 删除
- KNN 查询（返回 [(episode_id, cosine_distance), ...]）

cosine_distance 范围 [0, 2]，越小越近；similarity = 1 - distance / 2。
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

    def upsert(self, episode_id: int, embedding: list[float]) -> None:
        """
        插入或更新一条 episode 的向量。
        vec0 不支持 INSERT OR REPLACE，先删后插。
        """
        packed = _pack_f32(embedding)
        # 尝试删除已有
        self.db.execute(
            "DELETE FROM episode_vec WHERE episode_id = ?", (episode_id,)
        )
        self.db.execute(
            "INSERT INTO episode_vec(episode_id, embedding) VALUES (?, ?)",
            (episode_id, packed),
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

        可见性参数与 MemoirDB.fts_search 一致（三个允许集，或关系；都不传则不限制）。

        实现：先取 k*3 候选，再按可见性在 python 侧过滤，取前 k。
        （vec0 对 JOIN 支持有限，规模小时这么做完全够用。）
        """
        packed = _pack_f32(query_embedding)
        overshot_k = k * 3

        rows = self.db.fetchall(
            """
            SELECT episode_id, distance
            FROM episode_vec
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (packed, overshot_k),
        )
        if not rows:
            return []

        candidate_ids = [r["episode_id"] for r in rows]
        distances = {r["episode_id"]: r["distance"] for r in rows}

        placeholders = ",".join("?" * len(candidate_ids))
        meta_rows = self.db.fetchall(
            f"SELECT id, chat_type, session_id, group_id FROM episodes "
            f"WHERE id IN ({placeholders})",
            candidate_ids,
        )
        meta = {r["id"]: r for r in meta_rows}

        no_scope = (
            include_session_id is None
            and include_group_id is None
            and not include_all_groups
        )

        filtered: list[tuple[int, float]] = []
        for eid in candidate_ids:
            m = meta.get(eid)
            if m is None:
                continue

            if no_scope:
                visible = True
            else:
                visible = False
                if (
                    include_session_id is not None
                    and m["chat_type"] == "private"
                    and m["session_id"] == include_session_id
                ):
                    visible = True
                elif m["chat_type"] == "group":
                    if include_all_groups:
                        visible = True
                    elif include_group_id is not None and m["group_id"] == include_group_id:
                        visible = True

            if not visible:
                continue

            filtered.append((eid, distances[eid]))
            if len(filtered) >= k:
                break

        return filtered

    def count(self) -> int:
        row = self.db.fetchone("SELECT COUNT(*) AS n FROM episode_vec")
        return row["n"] if row else 0
