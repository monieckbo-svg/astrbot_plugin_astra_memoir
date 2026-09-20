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
    trigger_raw_id      INTEGER,                        -- Astra 回复的触发消息 raw_id；用户消息为 NULL
    created_at          INTEGER NOT NULL,               -- unix ts
    processed_at        INTEGER                         -- NULL = 未消化
);

CREATE INDEX IF NOT EXISTS idx_rm_session_time
    ON recent_messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_rm_unprocessed
    ON recent_messages(processed_at, session_id)
    WHERE processed_at IS NULL;

-- 供 on_llm_response 反查触发消息用
CREATE INDEX IF NOT EXISTS idx_rm_platform_msg
    ON recent_messages(session_id, platform_message_id)
    WHERE platform_message_id IS NOT NULL;

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
    last_accessed_at    INTEGER,
    importance          INTEGER NOT NULL DEFAULT 3,
    last_reinforced_at  TEXT,
    reinforcement_count INTEGER NOT NULL DEFAULT 0,
    is_archived         INTEGER NOT NULL DEFAULT 0,
    status              TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','archived','trashed')),
    archive_reason      TEXT,
    merged_into         INTEGER,
    edited_by_user      INTEGER NOT NULL DEFAULT 0,
    restored_by_user    INTEGER NOT NULL DEFAULT 0,
    protected_until     INTEGER,
    created_by_run_id   INTEGER
);

-- 可逆生命周期。is_archived 为 vec0 兼容镜像：active=0，其余=1。
-- 新字段由 db.py 对旧库做 ALTER TABLE；这里供新库直接创建。

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

-- 人物身份：QQ 号为观测主键；合并只建立映射，不改写历史 raw。
CREATE TABLE IF NOT EXISTS identities (
    qq_id           TEXT PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    pronoun         TEXT NOT NULL DEFAULT 'TA' CHECK(pronoun IN ('TA','她','他')),
    person_type     TEXT NOT NULL DEFAULT 'human' CHECK(person_type IN ('human','AI'))
);
CREATE TABLE IF NOT EXISTS identity_aliases (
    qq_id TEXT NOT NULL REFERENCES identities(qq_id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    PRIMARY KEY (qq_id, alias)
);
CREATE TABLE IF NOT EXISTS identity_redirects (
    source_qq_id TEXT PRIMARY KEY,
    target_qq_id TEXT NOT NULL REFERENCES identities(qq_id)
);

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

CREATE TABLE IF NOT EXISTS episode_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL REFERENCES episodes(id),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    importance INTEGER NOT NULL,
    saved_at INTEGER NOT NULL,
    reason TEXT NOT NULL DEFAULT 'user_edit'
);

CREATE TABLE IF NOT EXISTS episode_merge_sources (
    merged_episode_id INTEGER NOT NULL REFERENCES episodes(id),
    source_episode_id INTEGER NOT NULL,
    PRIMARY KEY (merged_episode_id, source_episode_id)
);

CREATE TABLE IF NOT EXISTS maintenance_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL CHECK(run_type IN ('history','nightly')),
    target_start INTEGER NOT NULL,
    target_end INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('generating','preview','applied','undone','failed')),
    backup_path TEXT,
    source_count INTEGER NOT NULL DEFAULT 0,
    keep_count INTEGER NOT NULL DEFAULT 0,
    archive_count INTEGER NOT NULL DEFAULT 0,
    merge_source_count INTEGER NOT NULL DEFAULT 0,
    merge_result_count INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    applied_at INTEGER,
    undone_at INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS maintenance_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES maintenance_runs(id) ON DELETE CASCADE,
    action TEXT NOT NULL CHECK(action IN ('keep','archive','merge')),
    source_episode_id INTEGER NOT NULL REFERENCES episodes(id),
    merge_group TEXT,
    proposed_title TEXT,
    proposed_content TEXT,
    proposed_importance INTEGER,
    reason TEXT,
    result_episode_id INTEGER,
    before_status TEXT,
    before_archive_reason TEXT,
    before_merged_into INTEGER,
    before_importance INTEGER
);
CREATE INDEX IF NOT EXISTS idx_maintenance_actions_run ON maintenance_actions(run_id);

CREATE TABLE IF NOT EXISTS maintenance_days (
    target_date TEXT PRIMARY KEY,
    run_id INTEGER REFERENCES maintenance_runs(id),
    status TEXT NOT NULL CHECK(status IN ('running','preview','applied','failed')),
    updated_at INTEGER NOT NULL
);
