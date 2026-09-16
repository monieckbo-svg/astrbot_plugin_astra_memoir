-- astrbot_plugin_astra_memoir Phase 1 schema
-- 参见 DESIGN.md 第 3 节
--
-- 注意：sqlite-vec 的 episode_vec 虚拟表在 vec_store.py 里创建
-- （需要先 load_extension 才能 CREATE VIRTUAL TABLE ... USING vec0）

-- ============================================================
-- 1. recent_messages: 近期原文缓存（幂等）
-- ============================================================
CREATE TABLE IF NOT EXISTS recent_messages (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    dedupe_key          TEXT NOT NULL UNIQUE,           -- 幂等键
    platform            TEXT NOT NULL DEFAULT 'qq',     -- qq / discord（预留）
    chat_type           TEXT NOT NULL,                  -- private / group
    session_id          TEXT NOT NULL,                  -- unified_message_origin
    group_id            TEXT,                           -- 群号，私聊为 NULL
    platform_message_id TEXT,                           -- QQ 原始 msg_id（assistant 可为 NULL）
    speaker_id          TEXT NOT NULL,                  -- QQ 号（bot 自己也用 QQ 号）
    speaker_name        TEXT NOT NULL,                  -- 当时昵称快照
    role                TEXT NOT NULL,                  -- user / assistant
    content             TEXT NOT NULL,
    reply_to_id         TEXT,                           -- 引用回复的 platform_message_id
    created_at          INTEGER NOT NULL,               -- unix ts
    processed_at        INTEGER                         -- NULL = 未消化
);

CREATE INDEX IF NOT EXISTS idx_rm_session_time
    ON recent_messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_rm_unprocessed
    ON recent_messages(processed_at, session_id)
    WHERE processed_at IS NULL;

-- ============================================================
-- 2. episodes: 事件本体
-- ============================================================
CREATE TABLE IF NOT EXISTS episodes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    platform            TEXT NOT NULL DEFAULT 'qq',
    chat_type           TEXT NOT NULL,                  -- private / group
    session_id          TEXT NOT NULL,
    group_id            TEXT,
    title               TEXT NOT NULL,
    content             TEXT NOT NULL,                  -- 纯正文，无 <MNEMO_META>
    source_raw_ids      TEXT NOT NULL,                  -- JSON array of recent_messages.id
    event_start_at      INTEGER NOT NULL,               -- source raw 最早 created_at
    event_end_at        INTEGER NOT NULL,               -- source raw 最晚 created_at
    extracted_at        INTEGER NOT NULL,               -- 模型完成提取的时间
    last_accessed_at    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_ep_session_time
    ON episodes(session_id, event_end_at);

CREATE INDEX IF NOT EXISTS idx_ep_chat_type
    ON episodes(chat_type);

CREATE INDEX IF NOT EXISTS idx_ep_group
    ON episodes(group_id);

-- ============================================================
-- 3. episode_participants: 参与者（由代码回查生成，非 LLM 输出）
-- ============================================================
CREATE TABLE IF NOT EXISTS episode_participants (
    episode_id      INTEGER NOT NULL,
    speaker_id      TEXT NOT NULL,                      -- QQ 号，身份主键
    speaker_name    TEXT NOT NULL,                      -- 当时昵称快照
    role            TEXT NOT NULL,                      -- user / assistant
    PRIMARY KEY (episode_id, speaker_id),
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ep_part_speaker
    ON episode_participants(speaker_id);

-- ============================================================
-- 4. episode_keywords: 关键词
-- ============================================================
CREATE TABLE IF NOT EXISTS episode_keywords (
    episode_id      INTEGER NOT NULL,
    keyword         TEXT NOT NULL,
    PRIMARY KEY (episode_id, keyword),
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
);

-- ============================================================
-- 5. episodes_fts: FTS5 全文索引（普通表，手工同步；避免 external-content 的 schema 依赖问题）
-- ============================================================
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    title,
    content,
    keywords,                                           -- 空格拼接的关键词字符串
    tokenize='unicode61'
);

-- ============================================================
-- 6. 元信息表（记录 schema 版本、embedding 维度等，便于以后升级）
-- ============================================================
CREATE TABLE IF NOT EXISTS meta (
    key     TEXT PRIMARY KEY,
    value   TEXT
);

-- 初始化元信息（用 INSERT OR IGNORE 保证只写一次）
INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '1');
