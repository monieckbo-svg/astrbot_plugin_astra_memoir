# DESIGN — astrbot_plugin_astra_memoir

Phase 1 设计文档。开工前给 Celii 和 GPT 审阅用。

---

## 1. 数据流

```
QQ 消息（私聊 + 群聊）
        │
        ├─ 用户消息 → event_message_type(ALL) hook
        └─ Astra 回复 → on_llm_response hook
                │
                ▼
        SQLite: recent_messages
                │
        ┌───────┴───────┐
        │               │
    达到阈值         idle timeout
    私聊 10 轮       私聊 30 分钟
    群聊 20 条       群聊 15 分钟
        │               │
        └───────┬───────┘
                ▼
        process_batch(session_id)
                │
                ▼
        LLM 事件拆分（DeepSeek Flash / 用户配的 provider）
        JSON 输出，events=[] 也是合法结果
                │
                ▼
        SQLite: episodes + episode_participants + episode_keywords
                │
                ▼
        Embedding (BAAI/bge-large-zh-v1.5 / 用户配的 provider)
                │
                ▼
        sqlite-vec: episode_embeddings
                │
                │  ── 新消息到来 ──
                ▼
        向量 + FTS5 混合召回 → top 3~5 → 临时注入 Astra
```

---

## 2. Hook 挂载点

已读 AstrBot 3.5.24 源码确认。

### 2.1 用户消息入库
```python
@filter.event_message_type(EventMessageType.ALL)
async def on_user_message(self, event: AstrMessageEvent):
    # 落 raw cache
```
- **能抓群聊里所有人的消息**（包括未 @Astra 的），这是 Celii 明确要求
- 群聊 speaker_id 用 `event.get_sender_id()`（QQ 号，稳定主键）
- speaker_name 用 `event.get_sender_name()`（当时昵称快照）

### 2.2 Astra 回复入库

**用 `on_llm_response` 而不是 `on_decorating_result` / `after_message_sent`**：

| Hook | 时机 | 问题 |
|------|------|------|
| `on_llm_response` | LLM 生成后，工具调用完毕后的最终 completion | ✅ 干净、一次触发、拿到完整原文 |
| `on_decorating_result` | 结果装饰阶段 | ❌ splitter 插件在这里切碎 chain，可能拿到碎片 |
| `after_message_sent` | 发送完成后 | ❌ chain 可能已被切分；且如果发送失败也会导致漏抓 |

源码确认（`tool_loop_agent.py:123-131`）：
```python
if not llm_resp.tools_call_name:
    # 没有工具调用时才触发 OnLLMResponseEvent —— 这就是最终响应
    await self.pipeline_ctx.call_event_hook(
        self.event, EventType.OnLLMResponseEvent, llm_resp
    )
```
中间的 tool_use 步骤**不会**触发，正是我们要的"一次对话一次落库"。

```python
@filter.on_llm_response()
async def on_astra_reply(self, event: AstrMessageEvent, resp: LLMResponse):
    text = resp.completion_text
    # 落 raw cache，role=assistant，speaker_id=BOT_QQ_ID
```

### 2.3 assistant 的 speaker_id
- **不用 "assistant" 字符串冒充**
- 从 config 读一个 `bot_qq_id`（Celii 配置里填），作为 Astra 在 QQ 里的身份主键
- speaker_name 用 config 里的 `bot_display_name`（默认 "星星"）

---

## 3. SQLite Schema

主库 `memoir.db`，三张核心表 + FTS5 + sqlite-vec。

### 3.1 recent_messages（近期原文缓存）
```sql
CREATE TABLE recent_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    platform        TEXT NOT NULL DEFAULT 'qq',      -- 预留：qq / discord
    chat_type       TEXT NOT NULL,                    -- private / group
    session_id      TEXT NOT NULL,                    -- unified_message_origin
    group_id        TEXT,                             -- 群号，私聊为 NULL
    message_id      TEXT,                             -- QQ 原始 msg_id
    speaker_id      TEXT NOT NULL,                    -- QQ 号（bot 也用 QQ 号，不用 'assistant'）
    speaker_name    TEXT NOT NULL,                    -- 当时昵称快照
    role            TEXT NOT NULL,                    -- user / assistant
    content         TEXT NOT NULL,
    reply_to_id     TEXT,                             -- 引用回复的 msg_id
    created_at      INTEGER NOT NULL,                 -- unix ts
    processed_at    INTEGER                           -- NULL 表示未消化
);
CREATE INDEX idx_rm_session_time ON recent_messages(session_id, created_at);
CREATE INDEX idx_rm_unprocessed ON recent_messages(processed_at) WHERE processed_at IS NULL;
```

### 3.2 episodes（事件本体）
```sql
CREATE TABLE episodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    platform        TEXT NOT NULL DEFAULT 'qq',
    chat_type       TEXT NOT NULL,                    -- private / group
    session_id      TEXT NOT NULL,
    group_id        TEXT,
    title           TEXT NOT NULL,
    content         TEXT NOT NULL,                    -- 纯正文，没有 <MNEMO_META>
    source_message_ids TEXT NOT NULL,                 -- JSON array
    created_at      INTEGER NOT NULL,
    last_accessed_at INTEGER
);
CREATE INDEX idx_ep_session_time ON episodes(session_id, created_at);
CREATE INDEX idx_ep_chat_type ON episodes(chat_type);
```

### 3.3 episode_participants（参与者，由代码回查而非 LLM 生成）
```sql
CREATE TABLE episode_participants (
    episode_id      INTEGER NOT NULL,
    speaker_id      TEXT NOT NULL,                    -- QQ 号，身份主键
    speaker_name    TEXT NOT NULL,                    -- 当时昵称快照
    role            TEXT NOT NULL,                    -- user / assistant
    PRIMARY KEY (episode_id, speaker_id),
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
);
CREATE INDEX idx_ep_part_speaker ON episode_participants(speaker_id);
```

**防串号关键**：participants 由 `source_message_ids` → `recent_messages` 回查算出，**LLM 输出的 participants 一律忽略**。QQ 号是主键，昵称改了不会变成另一个人。Astra 参与时她的 QQ 号（config 里的 bot_qq_id）也进 participants。

### 3.4 episode_keywords
```sql
CREATE TABLE episode_keywords (
    episode_id      INTEGER NOT NULL,
    keyword         TEXT NOT NULL,
    PRIMARY KEY (episode_id, keyword),
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
);
```

### 3.5 FTS5 全文索引
```sql
CREATE VIRTUAL TABLE episodes_fts USING fts5(
    title, content, keywords,
    content='episodes',
    content_rowid='id',
    tokenize='unicode61'
);
```

### 3.6 sqlite-vec 向量索引
```sql
CREATE VIRTUAL TABLE episode_vec USING vec0(
    episode_id INTEGER PRIMARY KEY,
    embedding FLOAT[1024]              -- bge-large-zh-v1.5 是 1024 维
);
```

**为什么 sqlite-vec 而不是 Milvus**：
- 一个 `.db` 文件搞定，不需要单独跑 Docker 容器
- Celii 服务器 2G 内存紧张，Milvus 光启动就吃几百 MB
- 群十几个人 + 私聊的规模，一年最多几万条 episode，sqlite-vec 完全够用
- 换 embedding 模型时只需 DROP TABLE + 重建，比 Milvus 简单

---

## 4. 触发时机

统一 scheduler，每 60 秒扫一次"有未处理消息的活跃会话"（查 `recent_messages` 的 `processed_at IS NULL` 视图）。

| 场景 | 数量阈值 | idle 阈值 |
|------|----------|-----------|
| 私聊 | 10 轮（20 条 user+assistant） | 30 分钟无新消息 |
| 群聊 | 20 条 | 15 分钟无新消息 |

任一先满足即触发 `process_batch(session_id)`。**两个触发走同一入口**，禁止两套代码。

---

## 5. 事件拆分 Prompt

调用 `context.get_provider_by_id(config.extract_provider_id).text_chat(...)`，让用户在 config 里指定用哪个 provider。

### 5.1 私聊 prompt 骨架
```
你不是在总结整段聊天，而是在识别其中相互独立、未来可能被重新提及的事件。
一批可以包含 0 到 N 个事件。不同主题必须拆开；只有确实属于同一件事的发展过程才合并。
不要为了减少数量把午饭、工作、宠物、画图等无关主题写进同一事件。
每个事件必须能脱离本批次独立理解，保留具体人、物、名称、数字、决定和结果。

【本批对话】
[msg_id=101] 陆忱: 中午吃了石锅拌饭
[msg_id=102] 星星: ...
...

输出严格 JSON，格式：
{"events":[
  {
    "title": "简短标题",
    "content": "详细内容，能独立理解",
    "source_message_ids": ["101","102"],
    "keywords": ["石锅拌饭","午饭"]
  }
]}
如没有值得记忆的事件，返回 {"events":[]}
```

### 5.2 群聊 prompt 骨架
相同结构，但输入格式强调 speaker：
```
[msg_id=8871][qq=111111][name=小雨] 我昨天已经把那个插件重装了
[msg_id=8872][qq=222222][name=温局] 我这边重装以后还是报错
```

**speaker_id 由代码从 event 对象取，绝不让 LLM 从昵称或正文猜。**

### 5.3 Overlap（跨批次上下文）
群聊每批附带**上一批最后 4 条**作为 `<context_only>`，私聊附带 **2 条**。硬规则：
> 每个输出 episode 至少必须引用一个 new_messages 的 msg_id。

代码在事件写入前校验 `source_message_ids` 至少含一个本批新增 ID，否则丢弃该事件。这样旧消息不会被重复消化。

---

## 6. 检索与注入

### 6.1 触发时机
每次收到用户消息、进入 LLM 请求前（`on_llm_request` hook），做一次检索并注入 system_prompt。

### 6.2 混合召回
- **向量召回**: sqlite-vec `MATCH` top 10
- **关键词召回**: FTS5 `MATCH` top 10
- **合并 + 重排**:
  - 语义相似度 (0-1)
  - FTS5 命中加 +0.2
  - **当前 speaker 参与过 → +0.15**（不做硬过滤，只加分）
  - 时间衰减：`exp(-days/30) * 0.1`
- 取 top 3~5 注入

### 6.3 私聊 / 群聊边界（信息可见性）
- **陆忱 ↔ Astra 私聊时**：召回范围 = 当前私聊 episode ∪ **所有 QQ 群聊 episode**
  - 所以陆忱能问"上午群里那个插件后来怎么了"
- **Astra 在群里时**：召回范围 = **仅当前群 episode**
  - 不自动带出任何私聊内容或其他群内容
  - 防止群里泄漏私聊隐私

### 6.4 注入格式
干净、无 metadata：
```
[相关记忆]
1. [2026-09-14 · 群聊 · QQ空间插件更新]
   陆忱完成图片上传功能，小雨测试通过。
   参与者：陆忱、小雨、星星

2. [2026-09-15 · 私聊 · 虾仁炒饭]
   陆忱中午吃了石锅拌饭和奶茶，觉得晚上可能不想再吃。
```
不带 embedding score、内部 ID、JSON、关系图。

---

## 7. TTL 与清理

后台任务每天扫一次 `recent_messages`：
- 已 `processed_at` 且 `created_at < now - retention_days` → 删除
- 未处理的消息永不删（保护未消化数据）

默认 `retention_days = 30`。

episode 永久保留（Phase 1 不做衰减）。

---

## 8. 重复检测（V1 简化版）

新事件写入前，用 embedding 余弦相似度查最近 3 天内的 episode：
- 相似度 > 0.96
- 且 participants 高度重叠
- 才判定重复，跳过写入

**不调用 LLM 做重复判断**。误合并比多存一条更难修，V1 宁可多存。

---

## 9. Config Schema（用户填什么）

```jsonc
{
  "extract_provider_id": {
    "description": "事件拆分用哪个 LLM provider（在 AstrBot 里配好，填 provider id）",
    "type": "string",
    "default": ""
  },
  "embedding_provider_id": {
    "description": "Embedding 用哪个 provider（在 AstrBot 里配好，留空则用第一个可用的 embedding provider）",
    "type": "string",
    "default": ""
  },
  "bot_qq_id": {
    "description": "Astra 的 QQ 号（作为 participants 里的稳定身份主键）",
    "type": "string",
    "default": ""
  },
  "bot_display_name": {
    "description": "Astra 在记忆里的显示名",
    "type": "string",
    "default": "星星"
  },
  "private_batch_pairs": { "type": "int", "default": 10 },
  "private_idle_minutes": { "type": "int", "default": 30 },
  "group_batch_messages": { "type": "int", "default": 20 },
  "group_idle_minutes": { "type": "int", "default": 15 },
  "overlap_group_messages": { "type": "int", "default": 4 },
  "overlap_private_messages": { "type": "int", "default": 2 },
  "raw_retention_days": { "type": "int", "default": 30 },
  "retrieval_top_k": { "type": "int", "default": 4 },
  "enable_group_recall_in_private": { "type": "bool", "default": true },
  "embedding_dim": { "type": "int", "default": 1024 }
}
```

---

## 10. Phase 1 验收清单

跑一段时间必须能做到：

1. ✅ 中午说"今天吃了虾仁炒饭"，晚上问星星"我今天吃了啥"，能召回
2. ✅ 上午聊某插件 bug，下午说"那插件后来怎样了"，能召回对应 episode
3. ✅ 群里小雨说自己感冒，第二天陆忱私聊问"小雨怎么了"，能从群聊 episode 回答
4. ✅ 同一批群聊同时讨论"插件"和"吃饭"，拆成至少两个独立 episode，不揉成一坨
5. ✅ 一批只有"哈哈哈哈"、表情、无意义寒暄时，events=[]，不新增 episode
6. ✅ 群聊跨批次同一件事，overlap 能帮理解，每个 episode 必须引用至少一个新批次 msg_id
7. ✅ 群友 A 的事实绝不归到群友 B 名下（QQ ID 由代码回查决定）
8. ✅ Astra 群里参与时，Astra 出现在 participants
9. ✅ 私聊自动召回群聊事件；群聊自动召回不带出任何私聊 episode
10. ✅ raw message 过期删除后，episode 与检索仍正常

---

## 11. 目录结构

```
astrbot_plugin_astra_memoir/
├── main.py                          # 插件入口，挂 hook
├── metadata.yaml
├── _conf_schema.json
├── requirements.txt
├── README.md
├── DESIGN.md
├── storage/
│   ├── __init__.py
│   ├── schema.sql                   # 初始 SQL
│   ├── db.py                        # SQLite 连接与查询
│   └── vec_store.py                 # sqlite-vec 封装
├── pipeline/
│   ├── __init__.py
│   ├── raw_cache.py                 # 消息入库
│   ├── scheduler.py                 # idle flush 调度
│   ├── extractor.py                 # LLM 事件拆分
│   ├── writer.py                    # episode 落库
│   └── retriever.py                 # 检索 + 注入
└── utils/
    ├── __init__.py
    └── logging.py
```

---

## 12. 依赖

```txt
sqlite-vec>=0.1.6
```

其他全部用 AstrBot 已有的（httpx、pydantic、asyncio 等）。

---

## 13. 开工顺序

1. ✅ 骨架 + DESIGN.md（本 commit）
2. schema.sql + storage/db.py + storage/vec_store.py
3. main.py 挂 hook + pipeline/raw_cache.py（用户消息 & Astra 回复入库）
4. pipeline/scheduler.py + pipeline/extractor.py + pipeline/writer.py（消化 + 落库）
5. pipeline/retriever.py（检索 + 注入）
6. 跑通端到端 + 10 个验收 case

每步一个 commit，Celii 可随时看进度。
