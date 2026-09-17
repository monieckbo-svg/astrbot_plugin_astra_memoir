# DESIGN — astrbot_plugin_astra_memoir

Phase 1 设计文档。v2（采纳 GPT 全部 13 条反馈）。

---

## 变更历史

- **v4 (2026-09-16)**: Astra 回复 dedupe_key 改成内部 raw_id 绑定
  - `dedupe_key` 格式：`assistant:{session_id}:{trigger_raw_id}`（原：`llm:{session_id}:{trigger_user_message_id}:assistant`）
  - `recent_messages` 新增 `trigger_raw_id INTEGER NULL` 字段
  - 用户消息 trigger_raw_id=NULL；Astra 回复 trigger_raw_id=触发它的用户 raw.id
  - on_llm_response 里通过 `find_raw_by_platform_msg(session_id, platform_message_id)` 反查
  - 反查不到时 V1 warning + skip，不生成不稳定 fallback key
  - 平台 message_id 只负责"定位原消息"，内部关系统一走 raw_id
- **v3 (2026-09-16)**: 版本号纠正 + 注入方式改回 v4 官方 API
  - AstrBot 分支纠正为 **v4.27.5**（Celii 实际运行版本，v3.5.x 是并行老分支）
  - 注入方式改回 GPT 推荐的 `req.extra_user_content_parts.append(TextPart(text=...).mark_as_temp())`
    —— v4 官方 `group_chat_context` 内置插件就是这个用法，比 prompt 前置更干净：
    `mark_as_temp()` 明确告诉 AstrBot"只给 provider 看、不进对话历史"
- **v2 (2026-09-16)**: 采纳 GPT 反馈 13 条（版本号一开始误写为 v3.5.24）
  - 记忆注入改用 `req.prompt` 前置，禁止污染 `system_prompt`（保护 prompt cache）
  - FTS5 改普通表 + 手工同步（去掉 external-content 的 schema bug）
  - sqlite-vec 显式 `distance_metric=cosine`
  - `source_message_ids` → `source_raw_ids`（用 `recent_messages.id` 做证据主键）
  - `recent_messages` 加 `dedupe_key UNIQUE` 保证幂等
  - 触发阈值语义化：私聊 10 个 user turn，群聊 20 条 inbound
  - `bot_qq_id` 删掉，改用 `event.get_self_id()`
  - DB 固定 `data/plugin_data/{plugin_name}/`
  - episode 时间字段拆成 `event_start_at / event_end_at / extracted_at`
  - 混合召回改 RRF（Reciprocal Rank Fusion）
  - 检索 query 用 `event.message_str`（不用被改写的 `req.prompt`）
  - `process_batch` 加 per-session lock；events=[] 也标 processed
- v1: 初版

---

## 1. 数据流

```
QQ 消息（私聊 + 群聊）
        │
        ├─ 用户消息 → event_message_type(ALL) hook
        └─ Astra 回复 → on_llm_response hook
                │
                ▼
        SQLite: recent_messages（幂等，dedupe_key UNIQUE）
                │
        ┌───────┴───────┐
        │               │
    数量阈值         idle timeout
    私聊 10 user     私聊 30 分钟
    群聊 20 inbound  群聊 15 分钟
        │               │
        └───────┬───────┘
                ▼
    process_batch(session_id)  [per-session lock]
                │
                ▼
    LLM 事件拆分（用户配的 provider）
    JSON 输出，events=[] 合法
                │
                ▼
    SQLite: episodes + episode_participants + episode_keywords + episodes_fts
                │
                ▼
    Embedding（用户配的 embedding provider）
                │
                ▼
    sqlite-vec: episode_vec (distance_metric=cosine)
                │
                ▼
    全部成功 → recent_messages 批量标 processed
    events=[] → 同样标 processed（防止死循环重跑）
                │
                │  ── 新消息到来 ──
                ▼
    on_llm_request hook: 用 event.message_str 做检索 query
    向量 KNN + FTS5 → RRF 融合 + participant/recency 加分 → top 3~5
                │
                ▼
    req.extra_user_content_parts.append(
        TextPart(text=f"[相关记忆]\n{memory}\n[/相关记忆]").mark_as_temp()
    )
    system_prompt / prompt / contexts 全部不动，只在本轮 provider 请求追加
    mark_as_temp() 保证不进对话历史，prompt cache 完好
```

---

## 2. Hook 挂载点

已读 AstrBot **v4.27.5** 源码确认（Celii 服务器实际运行版本；v3.5.x 是并行的老分支）。

### 2.1 用户消息入库
```python
@filter.event_message_type(EventMessageType.ALL)
async def on_user_message(self, event: AstrMessageEvent):
    # 落 raw cache
```
- 抓群聊里所有人的消息（包括未 @Astra 的），Celii 明确要求
- speaker_id = `event.get_sender_id()`（QQ 号，稳定身份主键）
- speaker_name = `event.get_sender_name()`（当时昵称快照）
- platform_message_id = 从 event 里取

### 2.2 Astra 回复入库

**用 `on_llm_response`**（v4 里也可选 `on_agent_done`，两者在 `astr_agent_hooks.on_agent_done` 里连续触发，语义相近；`on_llm_response` 是历史 API，`on_agent_done` 是 v4 新增更明确的"agent 一轮完成"信号——V1 用 `on_llm_response` 兼容性更好）：

| Hook | 时机 | 评估 |
|------|------|------|
| `on_llm_response` | agent 完成（tool loop 结束、拿到最终 completion）时触发 | ✅ 拿到完整原文 |
| `on_agent_done` | 同上，v4 新增，语义更明确 | ✅ 也可用 |
| `on_decorating_result` | 结果装饰阶段 | ❌ splitter 在这里切碎 chain |
| `after_message_sent` | 发送完成后 | ❌ chain 已被切分；发送失败会漏抓 |

v4 源码确认（`astrbot/core/astr_agent_hooks.py:32-40`）：`on_llm_response` 与 `on_agent_done` 都在 `agent_hooks.on_agent_done()` 里连续 `call_event_hook`，即 tool loop 结束后的最终响应。

```python
@filter.on_llm_response()
async def on_astra_reply(self, event: AstrMessageEvent, resp: LLMResponse):
    text = resp.completion_text
    # speaker_id = event.get_self_id()  ← Astra 自己的 QQ 号，自动获取
    # role=assistant
    # platform_message_id = None  ← 此时消息还没发出，QQ msg_id 不存在
```

### 2.3 Astra 的 speaker_id
- **不用 "assistant" 字符串**
- **不需要用户手填** bot_qq_id
- 直接 `event.get_self_id()` 拿 bot 自己的 QQ 号，作为 participants 里的稳定身份主键
- speaker_name = 配置的 `bot_display_name`（默认 "星星"）

### 2.4 记忆检索与注入
```python
from astrbot.core.agent.message import TextPart

@filter.on_llm_request()
async def on_before_llm(self, event: AstrMessageEvent, req: ProviderRequest):
    # 用 event.message_str 做检索 query（不用 req.prompt，
    # 避免被群聊上下文插件等改写、召回被稀释）
    query = event.get_message_str()
    memories = await retriever.recall(session_id, chat_type, query, speaker_id)
    if memories:
        # v4 官方推荐姿势：append 到 extra_user_content_parts + mark_as_temp
        # 效果：只在本轮 provider 请求追加，不进 contexts 历史，
        #      不动 system_prompt，prompt cache 完好
        req.extra_user_content_parts.append(
            TextPart(text=f"[相关记忆]\n{memories}\n[/相关记忆]").mark_as_temp()
        )
```

**关键**：v4 的 `ProviderRequest` 有 `extra_user_content_parts: list[ContentPart]` 字段，专门用于本轮追加内容；`TextPart.mark_as_temp()` 会把 `_no_save=True`，保证 provider 发完请求后不会持久化到对话上下文（`entities.py:100`，`agent/message.py:66`）。

v4 官方 `astrbot/builtin_stars/astrbot/group_chat_context.py:194-197` 就是这个用法（虽然它没调 `mark_as_temp` —— 但对我们的场景 mark_as_temp 更保险）。

---

## 3. SQLite Schema

DB 路径：**`data/plugin_data/astrbot_plugin_astra_memoir/memoir.db`**（AstrBot 官方推荐插件持久数据位置，插件升级不受影响）。

### 3.1 recent_messages（近期原文缓存，幂等）
```sql
CREATE TABLE recent_messages (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    dedupe_key          TEXT NOT NULL UNIQUE,       -- 幂等键，防重复写入
    platform            TEXT NOT NULL DEFAULT 'qq', -- qq / discord（预留）
    chat_type           TEXT NOT NULL,              -- private / group
    session_id          TEXT NOT NULL,              -- unified_message_origin
    group_id            TEXT,                       -- 群号，私聊为 NULL
    platform_message_id TEXT,                       -- QQ 原始 msg_id（assistant 可为 NULL）
    speaker_id          TEXT NOT NULL,              -- QQ 号（bot 自己也用 QQ 号）
    speaker_name        TEXT NOT NULL,              -- 当时昵称快照
    role                TEXT NOT NULL,              -- user / assistant
    content             TEXT NOT NULL,
    reply_to_id         TEXT,                       -- 引用回复的 platform_message_id
    trigger_raw_id      INTEGER,                    -- Astra 回复的触发 raw_id；用户消息为 NULL
    created_at          INTEGER NOT NULL,           -- unix ts
    processed_at        INTEGER                     -- NULL = 未消化
);
CREATE INDEX idx_rm_session_time ON recent_messages(session_id, created_at);
CREATE INDEX idx_rm_unprocessed ON recent_messages(processed_at, session_id) WHERE processed_at IS NULL;
CREATE INDEX idx_rm_platform_msg ON recent_messages(session_id, platform_message_id) WHERE platform_message_id IS NOT NULL;
```

**dedupe_key 生成规则**：
- 用户消息：`user:{session_id}:{platform_message_id}`
- Astra 回复：`assistant:{session_id}:{trigger_raw_id}`
  （trigger_raw_id = 触发本次 LLM 的用户消息在 recent_messages 里的自增 id）

Astra 回复的 trigger_raw_id 通过 `find_raw_by_platform_msg(session_id, platform_message_id)` 反查——on_llm_response hook 里根据 `event` 拿到"这一轮 LLM 是由哪条 platform msg 触发"的原始 QQ msg_id，用它反查 recent_messages 拿到自增 id。**如果反查不到（用户消息还没入库、插件重启丢状态）→ V1 warning + 跳过该 assistant 写入，不生成不稳定 fallback key**。

好处：
- 所有内部关系统一走 raw_id，平台 message_id 只负责"定位原消息"
- 一条 inbound 消息对应最多一条 assistant final reply，UNIQUE 天然保证
- assistant 行的 `trigger_raw_id` 字段直接指向触发它的用户 raw，事件证据链清晰

用 `INSERT OR IGNORE`，无论 hook 重复触发多少次都不会脏库。

### 3.2 episodes（事件本体）
```sql
CREATE TABLE episodes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    platform            TEXT NOT NULL DEFAULT 'qq',
    chat_type           TEXT NOT NULL,              -- private / group
    session_id          TEXT NOT NULL,
    group_id            TEXT,
    title               TEXT NOT NULL,
    content             TEXT NOT NULL,              -- 纯正文，不带 <MNEMO_META>
    source_raw_ids      TEXT NOT NULL,              -- JSON array of recent_messages.id
    event_start_at      INTEGER NOT NULL,           -- source raw 最早 created_at
    event_end_at        INTEGER NOT NULL,           -- source raw 最晚 created_at
    extracted_at        INTEGER NOT NULL,           -- 模型完成提取的时间
    last_accessed_at    INTEGER
);
CREATE INDEX idx_ep_session_time ON episodes(session_id, event_end_at);
CREATE INDEX idx_ep_chat_type ON episodes(chat_type);
CREATE INDEX idx_ep_group ON episodes(group_id);
```

**关键变化**：
- `source_message_ids` → **`source_raw_ids`**：统一用 `recent_messages.id`（SQLite 自增），因为 Astra 回复在 on_llm_response 阶段还没有 QQ msg_id
- 时间字段拆成三个：**event_start_at / event_end_at** 是聊天实际发生时间（来自 source raw），**extracted_at** 才是模型提取完成时间。"上午 10 点聊的那个" 用 event_start_at 检索

### 3.3 episode_participants
```sql
CREATE TABLE episode_participants (
    episode_id      INTEGER NOT NULL,
    speaker_id      TEXT NOT NULL,      -- QQ 号，身份主键
    speaker_name    TEXT NOT NULL,      -- 当时昵称快照
    role            TEXT NOT NULL,      -- user / assistant
    PRIMARY KEY (episode_id, speaker_id),
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
);
CREATE INDEX idx_ep_part_speaker ON episode_participants(speaker_id);
```

**防串号关键**：participants 由 `source_raw_ids` → `recent_messages` 回查算出，**LLM 输出的 participants 一律忽略**。

### 3.4 episode_keywords
```sql
CREATE TABLE episode_keywords (
    episode_id      INTEGER NOT NULL,
    keyword         TEXT NOT NULL,
    PRIMARY KEY (episode_id, keyword),
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
);
```

### 3.5 FTS5 全文索引（普通表，手工同步）
```sql
-- 不用 external-content，避免 schema 依赖 episodes 表里不存在的 keywords 列
CREATE VIRTUAL TABLE episodes_fts USING fts5(
    title,
    content,
    keywords,               -- 用空格拼接的关键词字符串
    tokenize='unicode61'
);
```

写 episode 时同步插入：
```python
db.execute(
    "INSERT INTO episodes_fts(rowid, title, content, keywords) VALUES (?, ?, ?, ?)",
    (episode_id, title, content, " ".join(keywords))
)
```

几万条规模多存这点文本无所谓，逻辑简单可靠。

### 3.6 sqlite-vec 向量索引
```sql
CREATE VIRTUAL TABLE episode_vec USING vec0(
    episode_id INTEGER PRIMARY KEY,
    embedding FLOAT[1024] distance_metric=cosine    -- 显式 cosine，不用默认 L2
);
```

sqlite-vec KNN 返回 `distance`（越小越近）。代码统一用 **`cosine_distance`** 语义：
- 相似度阈值："distance < 0.04"（约等于 similarity > 0.96）
- 不在代码里混用 similarity 和 distance

**为什么 sqlite-vec 而不是 Milvus**：
- 一个 `.db` 文件搞定，不需要单独 Docker
- Celii 服务器内存紧张
- 规模一年最多几万 episode，sqlite-vec 足够
- 换 embedding 模型只需 DROP + 重建

---

## 4. 触发时机

统一 scheduler，每 60 秒扫一次"有未处理消息的活跃会话"。

| 场景 | 数量阈值 | idle 阈值 |
|------|----------|-----------|
| 私聊 | **10 个新 user turn**（assistant 行不算，全部作为上下文一起带上） | 30 分钟 |
| 群聊 | **20 条 inbound 群友消息**（Astra 自己回复不占阈值） | 15 分钟 |

任一先满足即触发 `process_batch(session_id)`。**per-session lock**，同一会话同时只跑一次；两个触发条件走同一入口。

**语义清晰**：
- 私聊：你连续发两句 Astra 才回一句，或某条被拦掉没 LLM 回复——都不会导致阈值误判
- 群聊：Astra 参与与否不影响群消息计数

---

## 5. 事件拆分 Prompt

调用 `context.get_provider_by_id(config.extract_provider_id).text_chat(...)`。

### 5.1 私聊 prompt 骨架
```
你不是在总结整段聊天，而是在识别其中相互独立、未来可能被重新提及的事件。
一批可以包含 0 到 N 个事件。不同主题必须拆开；只有确实属于同一件事的发展过程才合并。
不要为了减少数量把午饭、工作、宠物、画图等无关主题写进同一事件。
每个事件必须能脱离本批次独立理解，保留具体人、物、名称、数字、决定和结果。

【本批对话】
[raw_id=101] 陆忱: 中午吃了石锅拌饭
[raw_id=102] 星星: ...
...

输出严格 JSON：
{"events":[
  {
    "title": "简短标题",
    "content": "详细内容，能独立理解",
    "source_raw_ids": [101, 102],
    "keywords": ["石锅拌饭","午饭"]
  }
]}
如没有值得记忆的事件，返回 {"events":[]}
```

### 5.2 群聊 prompt 骨架
输入格式强调 speaker（QQ 号由代码从 event 对象取）：
```
[raw_id=8871][qq=111111][name=小雨] 我昨天已经把那个插件重装了
[raw_id=8872][qq=222222][name=温局] 我这边重装以后还是报错
```

### 5.3 Overlap（跨批次上下文）
群聊每批附带**上一批最后 4 条**作为 `<context_only>`，私聊 **2 条**。硬规则：
> 每个输出 episode 至少必须引用一个 new_messages 的 raw_id。

代码在事件写入前校验；任一事件不满足硬规则则整批重试一次，仍失败时 raw 保持未处理。旧消息只帮理解上下文，不会单独生成新事件。

### 5.4 幂等约束
- `process_batch` 加 **per-session asyncio.Lock**
- episodes / participants / keywords / FTS 与 `processed_at` 在**单个 transaction** 里完成；vec 是可重建索引，在 transaction 提交后写入
- embedding 生成失败时不落库；vec 写入失败时保留 episode，启动修复或面板手动补齐
- **events=[]** 也标 processed（无有效事件是合法结果，不能死循环重跑）
- 任一步失败：raw 保持 unprocessed，下次调度再试

---

## 6. 检索与注入

### 6.1 触发时机
`on_llm_request` hook，用 **`event.message_str`** 做检索 query（不用 `req.prompt`，因为可能已被群聊上下文插件改写、导致召回被稀释）。

### 6.2 混合召回（RRF）
**BM25 分数与 cosine distance 不同量纲，不能直接相加。用 RRF 融合**：

```python
# 向量召回 top 20
vec_hits = sqlite_vec_knn(query_embedding, k=20)  # [(episode_id, distance), ...]

# FTS5 召回 top 20
fts_hits = fts5_search(query, k=20)  # [episode_id, ...]

# RRF (k=60 是文献推荐值)
scores = {}
for rank, (eid, _) in enumerate(vec_hits):
    scores[eid] = scores.get(eid, 0) + 1 / (60 + rank)
for rank, eid in enumerate(fts_hits):
    scores[eid] = scores.get(eid, 0) + 1 / (60 + rank)

# 加分（都不做硬过滤）
for eid in scores:
    if current_speaker in participants[eid]:
        scores[eid] += 0.02  # participant bonus
    days_ago = (now - event_end_at[eid]) / 86400
    scores[eid] += 0.01 * exp(-days_ago / 30)  # recency

# 取 top_k
```

### 6.3 私聊 / 群聊边界（信息可见性）
- **配置在 `cross_group_owner_ids` 中的 owner 私聊时**：召回范围 = 当前私聊 episode ∪ **所有 QQ 群聊 episode**
  - 所以能问 "上午群里那个插件后来怎么了"
- 其他私聊用户：仅能召回自己的私聊 episode
- **Astra 在群里时**：召回范围 = **仅当前群 episode**
  - 不带出任何私聊或其他群
  - 防止群里泄漏隐私

### 6.4 注入格式（extra_user_content_parts + mark_as_temp）
```python
from astrbot.core.agent.message import TextPart

memory_text = "\n\n".join([
    f"[{fmt_date(ep.event_start_at)} · {chat_type_label} · {ep.title}]\n"
    f"{ep.content}\n"
    f"参与者：{', '.join(ep.participants)}"
    for ep in top_episodes
])

req.extra_user_content_parts.append(
    TextPart(text=f"[相关记忆]\n{memory_text}\n[/相关记忆]").mark_as_temp()
)
```

**不带**：embedding score、内部 ID、JSON、关系图。

**为什么 mark_as_temp**：
- `_no_save=True` 让这块内容只在**本轮 provider 请求**里出现
- 不会被写入 `contexts` 历史，下一轮不会重复出现
- system_prompt / prompt 全部不动，provider 侧 prompt cache 命中率不受影响

---

## 7. TTL 与清理

后台任务每天扫一次 `recent_messages`：
- 已 processed 且 `created_at < now - retention_days` → 删除
- 未处理的永不删（保护未消化数据）

默认 `retention_days = 30`。episode 永久保留（Phase 1 不做衰减）。

---

## 8. 重复检测（V1 简化）

新事件写入前，用 embedding 余弦距离查同一 session、事件时间相差 1 天内的 episode；距离 ≤ 0.025 判定重复，跳过写入。也比较同批新事件，避免同批重复。raw 仍标记 processed。

**不调用 LLM 判断重复**。误合并比多存一条更难修，V1 宁可多存。

---

## 9. Config Schema

（详见仓库 `_conf_schema.json`）

```jsonc
{
  "extract_provider_id":       "事件拆分用哪个 LLM provider ID",
  "embedding_provider_id":     "必选的 AstrBot Embedding provider ID",
  // "bot_qq_id" 已删除 —— 用 event.get_self_id() 自动获取
  "bot_display_name":          "星星在记忆里的显示名",
  "private_batch_user_turns":  10,     // 私聊：10 个 user turn
  "private_idle_minutes":      30,
  "group_batch_inbound":       20,     // 群聊：20 条 inbound 群消息
  "group_idle_minutes":        15,
  "overlap_group_messages":    4,
  "overlap_private_messages":  2,
  "raw_retention_days":        30,
  "retrieval_top_k":           4,
  "enable_group_recall_in_private": true,
  "cross_group_owner_ids":     "owner QQ，多个用逗号分隔",
  "scheduler_interval_seconds": 60
}
```

---

## 10. Phase 1 验收清单

跑一段时间必须能做到：

1. ✅ 中午说"今天吃了虾仁炒饭"，晚上问星星"我今天吃了啥"，能召回
2. ✅ 上午聊某插件 bug，下午说"那插件后来怎样了"，能召回对应 episode
3. ✅ 群里小雨说自己感冒，第二天陆忱私聊问"小雨怎么了"，能从群聊 episode 回答
4. ✅ 同一批群聊同时讨论"插件"和"吃饭"，拆成至少两个独立 episode
5. ✅ 一批只有"哈哈哈哈"、表情、无意义寒暄时，events=[]，不新增 episode，raw 仍标 processed
6. ✅ 群聊跨批次同一件事，overlap 能帮理解，每个 episode 必须引用至少一个新批次 raw_id
7. ✅ 群友 A 的事实绝不归到群友 B 名下（QQ ID 由代码回查决定）
8. ✅ Astra 群里参与时，Astra 出现在 participants（用 event.get_self_id()）
9. ✅ 私聊自动召回群聊事件；群聊自动召回不带出任何私聊 episode
10. ✅ raw message 过期删除后，episode 与检索仍正常
11. ✅ hook 重复触发（插件 reload / 消息重放）不会脏库（dedupe_key UNIQUE）
12. ✅ prompt cache 不被破坏（system_prompt / prompt / contexts 全部不动，记忆走 extra_user_content_parts + mark_as_temp）
13. ✅ process_batch 并发触发不会重复消化同一批 raw（per-session lock）

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
│   ├── raw_cache.py                 # 消息入库（含 dedupe_key 生成）
│   ├── scheduler.py                 # 60s 轮询 + per-session lock
│   ├── extractor.py                 # LLM 事件拆分 + JSON 校验
│   ├── writer.py                    # episode 落库 + FTS + vec + 幂等
│   └── retriever.py                 # RRF 检索 + prompt 前置注入
└── utils/
    ├── __init__.py
    ├── dedupe.py                    # dedupe_key 生成
    └── logging.py
```

---

## 12. 依赖

```txt
sqlite-vec>=0.1.6
```

其他全部用 AstrBot 已有依赖（httpx、pydantic、asyncio）。

---

## 13. 开工顺序

1. ✅ 骨架 + DESIGN v1（commit 1）
2. ✅ DESIGN v2 采纳 GPT 反馈（本 commit）
3. schema.sql + storage/db.py + storage/vec_store.py（含 sqlite-vec 加载、cosine 指定、dedupe_key UNIQUE）
4. main.py 完善 hook + pipeline/raw_cache.py（含 dedupe_key 生成、event.get_self_id()）
5. pipeline/scheduler.py + pipeline/extractor.py + pipeline/writer.py（per-session lock、events=[] 标 processed、事件校验硬规则）
6. pipeline/retriever.py（event.message_str 做 query、RRF 融合、prompt 前置注入）
7. 跑通端到端 + 13 个验收 case
