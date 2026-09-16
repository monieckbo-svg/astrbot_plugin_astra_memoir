"""
storage/db.py — SQLite 连接管理与初始化

用同步 sqlite3 + asyncio.to_thread() 的方式，避免引入 aiosqlite 依赖。
sqlite-vec 通过 load_extension 加载，在 vec_store.py 中完成。

DB 路径由 AstrBot v4 官方 API 提供：
    StarTools.get_data_dir()  →  data/plugin_data/astrbot_plugin_astra_memoir/
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import sqlite_vec

# schema.sql 相对于本文件的位置
_SCHEMA_SQL_PATH = Path(__file__).parent / "schema.sql"


class MemoirDB:
    """
    单例数据库连接封装。

    - `initialize()`: 建库、加载 sqlite-vec、执行 schema.sql、建 vec 表
    - `execute(sql, params)`: 同步 execute（内部使用；异步调用请用 `run(...)`）
    - `run(func, *args)`: 用 asyncio.to_thread 包装同步操作，返回 awaitable
    - `transaction()`: 事务上下文管理器（同步）
    - `close()`: 关闭连接
    """

    def __init__(self, db_path: Path, embedding_dim: int = 1024):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()  # 用于并发写入的粗粒度保护

    # ---------- 初始化 ----------

    def initialize(self) -> None:
        """
        建库、加载 sqlite-vec 扩展、执行 schema.sql、创建 vec 虚拟表。
        幂等：重复调用不会破坏已有数据。
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,  # 允许多线程共用（我们用 asyncio.to_thread）
            isolation_level=None,      # 手动管理事务
        )
        conn.row_factory = sqlite3.Row

        # 基础 pragmas
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")

        # 加载 sqlite-vec 扩展
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        # 执行 schema.sql（普通表 + FTS5 + meta）
        schema_sql = _SCHEMA_SQL_PATH.read_text(encoding="utf-8")
        conn.executescript(schema_sql)

        # 创建 vec0 虚拟表（在 python 里创建，因为需要动态填入 embedding_dim）
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS episode_vec USING vec0(
                episode_id INTEGER PRIMARY KEY,
                embedding FLOAT[{self.embedding_dim}] distance_metric=cosine
            )
            """
        )

        # 存 embedding_dim 到 meta，便于以后校验
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("embedding_dim", str(self.embedding_dim)),
        )

        self._conn = conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ---------- 同步操作（在 asyncio.to_thread 里调用） ----------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("MemoirDB not initialized. Call initialize() first.")
        return self._conn

    def execute(self, sql: str, params: tuple | list | dict = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, seq_params: list) -> sqlite3.Cursor:
        return self.conn.executemany(sql, seq_params)

    def fetchone(self, sql: str, params: tuple | list | dict = ()) -> sqlite3.Row | None:
        return self.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple | list | dict = ()) -> list[sqlite3.Row]:
        return self.execute(sql, params).fetchall()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """
        显式事务上下文管理器。用法：
            with db.transaction() as conn:
                conn.execute("INSERT ...")
                conn.execute("UPDATE ...")
        任一异常自动 ROLLBACK。
        """
        conn = self.conn
        conn.execute("BEGIN")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # ---------- 异步入口 ----------

    async def run(self, func, *args, **kwargs) -> Any:
        """
        用 asyncio.to_thread 包装一次同步操作。
        典型用法：
            await db.run(lambda: db.fetchall("SELECT ..."))
        """
        return await asyncio.to_thread(func, *args, **kwargs)

    # ---------- 常用查询封装 ----------

    def insert_raw_message(
        self,
        *,
        dedupe_key: str,
        platform: str,
        chat_type: str,
        session_id: str,
        group_id: str | None,
        platform_message_id: str | None,
        speaker_id: str,
        speaker_name: str,
        role: str,
        content: str,
        reply_to_id: str | None,
        trigger_raw_id: int | None,
        created_at: int,
    ) -> int | None:
        """
        插入一条 raw message，若 dedupe_key 冲突返回 None（幂等）。
        成功返回新行 id。

        trigger_raw_id: Astra 回复的触发消息 raw_id；用户消息传 None。
        """
        cur = self.execute(
            """
            INSERT OR IGNORE INTO recent_messages(
                dedupe_key, platform, chat_type, session_id, group_id,
                platform_message_id, speaker_id, speaker_name, role,
                content, reply_to_id, trigger_raw_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dedupe_key, platform, chat_type, session_id, group_id,
                platform_message_id, speaker_id, speaker_name, role,
                content, reply_to_id, trigger_raw_id, created_at,
            ),
        )
        return cur.lastrowid if cur.rowcount > 0 else None

    def find_raw_by_platform_msg(
        self, session_id: str, platform_message_id: str,
    ) -> sqlite3.Row | None:
        """
        供 on_llm_response 反查触发它的用户 raw：
        输入 session_id + QQ 原始 msg_id，返回对应的 recent_messages 行。
        找不到返回 None（V1 保守：跳过该 assistant 写入并 warning）。
        """
        return self.fetchone(
            "SELECT * FROM recent_messages "
            "WHERE session_id = ? AND platform_message_id = ? AND role = 'user' "
            "ORDER BY created_at DESC LIMIT 1",
            (session_id, platform_message_id),
        )

    def get_unprocessed_by_session(
        self, session_id: str, limit: int | None = None
    ) -> list[sqlite3.Row]:
        """按时间升序拉取指定会话所有未处理消息。"""
        sql = (
            "SELECT * FROM recent_messages "
            "WHERE session_id = ? AND processed_at IS NULL "
            "ORDER BY created_at ASC"
        )
        params: list = [session_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self.fetchall(sql, tuple(params))

    def get_active_sessions_with_unprocessed(self) -> list[sqlite3.Row]:
        """
        获取所有"存在未处理消息"的活跃会话，
        返回每个会话的 session_id、chat_type、最后未处理消息 created_at、未处理消息数。
        供 scheduler 判断 idle flush 用。
        """
        return self.fetchall(
            """
            SELECT
                session_id,
                MAX(chat_type) AS chat_type,
                MAX(created_at) AS last_unprocessed_at,
                COUNT(*) AS unprocessed_count,
                SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS user_turn_count,
                SUM(CASE WHEN role = 'user' AND chat_type = 'group' THEN 1 ELSE 0 END) AS group_inbound_count
            FROM recent_messages
            WHERE processed_at IS NULL
            GROUP BY session_id
            """
        )

    def mark_processed(self, raw_ids: list[int], processed_at: int) -> None:
        """把一批 raw messages 标为已处理。"""
        if not raw_ids:
            return
        placeholders = ",".join("?" * len(raw_ids))
        self.execute(
            f"UPDATE recent_messages SET processed_at = ? WHERE id IN ({placeholders})",
            [processed_at, *raw_ids],
        )

    def delete_expired_raw(self, cutoff_ts: int) -> int:
        """删除已处理且早于 cutoff_ts 的 raw messages，返回删除条数。"""
        cur = self.execute(
            "DELETE FROM recent_messages WHERE processed_at IS NOT NULL AND created_at < ?",
            (cutoff_ts,),
        )
        return cur.rowcount

    # ---------- Episode 相关 ----------

    def insert_episode(
        self,
        *,
        platform: str,
        chat_type: str,
        session_id: str,
        group_id: str | None,
        title: str,
        content: str,
        source_raw_ids: list[int],
        event_start_at: int,
        event_end_at: int,
        extracted_at: int,
    ) -> int:
        """插入一条 episode，返回新 id。source_raw_ids 存为 JSON string。"""
        cur = self.execute(
            """
            INSERT INTO episodes(
                platform, chat_type, session_id, group_id,
                title, content, source_raw_ids,
                event_start_at, event_end_at, extracted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                platform, chat_type, session_id, group_id,
                title, content, json.dumps(source_raw_ids),
                event_start_at, event_end_at, extracted_at,
            ),
        )
        return cur.lastrowid

    def insert_participants(
        self,
        episode_id: int,
        participants: list[dict],
    ) -> None:
        """
        批量插入参与者。participants 每项形如：
            {"speaker_id": "111", "speaker_name": "小雨", "role": "user"}
        """
        if not participants:
            return
        self.executemany(
            "INSERT OR IGNORE INTO episode_participants("
            "episode_id, speaker_id, speaker_name, role) VALUES (?, ?, ?, ?)",
            [
                (episode_id, p["speaker_id"], p["speaker_name"], p["role"])
                for p in participants
            ],
        )

    def insert_keywords(self, episode_id: int, keywords: list[str]) -> None:
        if not keywords:
            return
        self.executemany(
            "INSERT OR IGNORE INTO episode_keywords(episode_id, keyword) VALUES (?, ?)",
            [(episode_id, kw) for kw in keywords],
        )

    def insert_fts(
        self, episode_id: int, title: str, content: str, keywords: list[str]
    ) -> None:
        """FTS5 手工同步：写 episode 时同步插入。"""
        self.execute(
            "INSERT INTO episodes_fts(rowid, title, content, keywords) VALUES (?, ?, ?, ?)",
            (episode_id, title, content, " ".join(keywords)),
        )

    def get_participant_speakers(self, episode_id: int) -> list[str]:
        rows = self.fetchall(
            "SELECT speaker_id FROM episode_participants WHERE episode_id = ?",
            (episode_id,),
        )
        return [r["speaker_id"] for r in rows]

    def get_episodes_by_ids(self, episode_ids: list[int]) -> list[sqlite3.Row]:
        if not episode_ids:
            return []
        placeholders = ",".join("?" * len(episode_ids))
        return self.fetchall(
            f"SELECT * FROM episodes WHERE id IN ({placeholders})",
            episode_ids,
        )

    def fts_search(
        self, query: str, limit: int = 20,
        *,
        include_session_id: str | None = None,
        include_group_id: str | None = None,
        include_all_groups: bool = False,
    ) -> list[int]:
        """
        FTS5 搜索，返回 episode_id 列表（按 BM25 排名升序）。

        可见性参数（三个允许集，或关系；三个都不传则不限制）：
        - include_session_id: 允许该私聊 session 的 episode
        - include_group_id:   允许该群 的 episode
        - include_all_groups: 允许所有群的 episode

        典型场景（对应 DESIGN 6.3）：
        - 私聊时: include_session_id=当前私聊, include_all_groups=True
        - 群聊时: include_group_id=当前群
        """
        cleaned = query.strip()
        if not cleaned:
            return []

        base_sql = (
            "SELECT e.id FROM episodes_fts "
            "JOIN episodes e ON e.id = episodes_fts.rowid "
            "WHERE episodes_fts MATCH ? "
        )
        params: list = [cleaned]

        # 拼可见性 OR 条件
        or_clauses: list[str] = []
        if include_session_id is not None:
            or_clauses.append("(e.chat_type = 'private' AND e.session_id = ?)")
            params.append(include_session_id)
        if include_all_groups:
            or_clauses.append("e.chat_type = 'group'")
        elif include_group_id is not None:
            or_clauses.append("(e.chat_type = 'group' AND e.group_id = ?)")
            params.append(include_group_id)

        if or_clauses:
            base_sql += "AND (" + " OR ".join(or_clauses) + ") "

        base_sql += "ORDER BY bm25(episodes_fts) LIMIT ?"
        params.append(limit)

        rows = self.fetchall(base_sql, tuple(params))
        return [r["id"] for r in rows]

    def get_meta(self, key: str) -> str | None:
        row = self.fetchone("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value)
        )
