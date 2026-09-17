# astrbot_plugin_astra_memoir

星星的自动记忆库 — 私聊 / 群聊事件消化，能记住每个人聊了什么。

---

## 定位

- **世界书**管固定身份 / 关系设定
- **astra-room + OMBER** 管日报 / 日记（小卧室）
- **本插件**只管"发生过什么"——把每天真实对话自动消化成可检索的事件

---

## 只做一件事

```
QQ 消息（私聊 + 群聊）
   → SQLite 近期原文缓存
     → 达到阈值 / 空闲超时
       → LLM 拆事件（DeepSeek 之类的便宜快模型）
         → episode 存储 + FTS + 向量索引
           → 语义 + 关键词混合检索
             → 临时注入到 LLM 请求（不污染 system_prompt）
```

---

## 安装

### 首次安装

```bash
# 1. 进 astrbot 插件目录（首尔那台）
cd /path/to/AstrBot/data/plugins

# 2. 克隆
git clone https://github.com/monieckbo-svg/astrbot_plugin_astra_memoir.git

# 3. 装依赖（AstrBot 已有 sqlite3 支持；只需 sqlite-vec）
pip install sqlite-vec
# 或者用 astrbot 的方式
cd astrbot_plugin_astra_memoir && pip install -r requirements.txt
```

### 后续升级

```bash
cd /path/to/AstrBot/data/plugins/astrbot_plugin_astra_memoir
git pull
# 然后在 AstrBot 面板重载插件即可
```

---

## 配置

在 AstrBot 面板 → 插件管理 → **astrbot_plugin_astra_memoir** → 配置：

必填两项：

| 字段 | 说明 |
|---|---|
| `extract_provider_id` | 事件拆分 LLM provider ID（**强烈建议先建一个便宜快的 provider 如 DeepSeek Flash**，别用跟星星聊天的主 provider 抢配额） |
| `embedding_provider_id` | 必须从已配置的 Embedding provider 中选择；插件会自动获取维度并补齐缺失向量 |

`owner_qq_id` 只用于身份标识与调试显示，**不控制记忆可见性**。旧版 `cross_group_owner_ids`、`enable_group_recall_in_private` 不再生效。

所有场景都从**全库未归档 episode** 取候选。私聊统一排序，默认最多 3 条（硬上限 5）；群聊默认最多 3 条群聊来源＋1 条私聊来源、总共 4 条，都是上限而非凑数。当前会话/群和参与者只得轻微加分，不过滤其它来源。“刚刚群里”“昨天上午群里”一类宽泛回忆问法走来源＋时间窗口查询。群里任何人、任何私聊用户都可能通过相关提问触发其他会话的记忆注入；Astra 是否说出由其人设决定，请只在接受这一隐私风险时安装此版本。

新事件由提取模型给出 `importance=1..5`；1 不写入长期 episode，但原文照常标记已处理。2 超过 30 天未强化会归档，不物理删除；面板仍可查看。3/4/5 按时间衰减检索排序，5 不衰减。只有真实用户消息明确包含完整事件标题，或新提取事件与旧事件高度相似时才强化；单纯召回或 Debug Recall 不强化。默认注入上限 3 条，硬上限 5 条。

其他可以先用默认。想调优时看：

- `retrieval_max_cosine_distance`（默认 0.9）—— 相关性阈值，改小 = 更严格，改大 = 更宽松
- `private_batch_user_turns` / `group_batch_inbound` —— 触发消化的阈值
- `raw_retention_days`（默认 30） —— 原文保留天数

---

## 启动检查

重启 astrbot 后，日志里出现这些关键字说明插件跑起来了：

```
[Memoir] DB path: /path/to/data/plugin_data/astrbot_plugin_astra_memoir/memoir.db
[Memoir] 插件加载完成 (embedding_provider=..., dim=1024)
[Memoir] scheduler 已启动 (interval=60s)
```

聊几轮后，应该看到：

```
[Memoir] trigger process_batch: session=... chat_type=private reason=private idle 1830s
[Memoir] extracted 2 event(s) from 4 new msg(s) (chat_type=private)
[Memoir] batch done: session=... new=4 events_ok=2 events_fail=0
```

星星回复前，看到：

```
[Memoir] injected 2 memory item(s) into LLM request (chat_type=private)
```

---

## DB 检查命令

数据在 `data/plugin_data/astrbot_plugin_astra_memoir/memoir.db`。用 sqlite3 直接查：

```bash
DB=/path/to/data/plugin_data/astrbot_plugin_astra_memoir/memoir.db

# 有多少条事件
sqlite3 $DB "SELECT COUNT(*) FROM episodes;"

# 最近 10 条事件
sqlite3 $DB "SELECT id, chat_type, title, datetime(event_start_at, 'unixepoch', 'localtime') FROM episodes ORDER BY id DESC LIMIT 10;"

# 某条事件的完整内容
sqlite3 $DB "SELECT title, content FROM episodes WHERE id = 1;"

# 某条事件的参与者
sqlite3 $DB "SELECT speaker_id, speaker_name, role FROM episode_participants WHERE episode_id = 1;"

# 未处理的原文数
sqlite3 $DB "SELECT session_id, COUNT(*) FROM recent_messages WHERE processed_at IS NULL GROUP BY session_id;"

# FTS 搜索
sqlite3 $DB "SELECT e.id, e.title FROM episodes_fts JOIN episodes e ON e.id=episodes_fts.rowid WHERE episodes_fts MATCH '插件';"

# 向量表条数（跟 episode 数应一致）
sqlite3 $DB "SELECT COUNT(*) FROM episode_vec;"
```

---

## 调试

打开 debug 日志（AstrBot 层面设置 log level = DEBUG）后能看到：

```
[Memoir] extract: chat_type=private, new=4, overlap=2
[Memoir] recall: eid=3 title='小雨感冒了' final=0.0183 rrf=0.0167 vec_dist=0.234 fts=True parts=['小雨']
```

`vec_dist` / `fts` / `final` 是调 `retrieval_max_cosine_distance` 的依据。

---

## 边界

**做**：
- 私聊事件 + 群聊事件消化
- 语义 + FTS 混合检索
- 全会话统一候选；群聊对 group/private 分别限额（明确授权，存在泄漏风险）
- 每人的话严格归属（QQ 号做身份主键）
- 30 天原文 TTL

**不做**：
- pin / feel / plan / anchor / letter（有 OMBER + 世界书）
- 群友画像 / 关系图
- 日报 / 日记（astra-room 的活）
- Discord（预留字段，代码不写）
- 复杂记忆合并 / dream / cron 打包

---

## 架构

代码结构：

```
astrbot_plugin_astra_memoir/
├── main.py                    # 装配 + 3 个 hook
├── metadata.yaml
├── _conf_schema.json
├── requirements.txt
├── DESIGN.md                  # 完整设计文档（含 GPT 全部反馈）
├── storage/
│   ├── schema.sql             # SQL 建表
│   ├── db.py                  # SQLite 封装（同步 + asyncio.to_thread）
│   └── vec_store.py           # sqlite-vec 封装
├── pipeline/
│   ├── raw_cache.py           # 消息入库
│   ├── extractor.py           # LLM 事件拆分
│   ├── writer.py              # episode 落库 + embedding
│   ├── scheduler.py           # 60s 轮询 + per-session lock
│   └── retriever.py           # 检索 + 注入
├── utils/dedupe.py
└── test/
    ├── integration_test.py    # Phase 1d 端到端（5 场景）
    └── retriever_test.py      # Phase 1e 检索验收（10 场景）
```

详细数据流、SQL schema、hook 挂载理由、API 选择依据都在 [DESIGN.md](./DESIGN.md)。

---

## 本地跑测试

```bash
cd astrbot_plugin_astra_memoir
pip install sqlite-vec
python3 test/integration_test.py
python3 test/retriever_test.py
```
