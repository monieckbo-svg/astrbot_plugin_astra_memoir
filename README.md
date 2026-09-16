# astrbot_plugin_astra_memoir

星星的自动记忆库 — 私聊 / 群聊事件消化，能记住每个人聊了什么。

## 定位

- **世界书**管固定身份/关系设定
- **astra-room + OMBER** 管日报 / 日记（小卧室）
- **本插件**只管"发生过什么"——把每天真实对话自动消化成可检索的事件

## 只做一件事

QQ 消息（私聊 + 群聊）→ SQLite 近期原文 → 达到阈值 / 空闲超时 → LLM 拆事件 → episode 存储 → 语义 + 关键词检索 → 临时注入 Astra 上下文。

## 需要 Celii 配置

- 一个已在 AstrBot 里配好的 **LLM provider**（推荐 DeepSeek Flash 或类似便宜快模型，用来做事件拆分）
- 一个已在 AstrBot 里配好的 **Embedding provider**（BAAI/bge-large-zh-v1.5 就行）

其他都用默认。

## Phase 1 边界

**做**：私聊事件 + 群聊事件 + 检索注入 + 近期原文 30 天 TTL

**不做**：pin / feel / plan / anchor / letter / 群友画像 / Discord / 日报 / 日记 / 衰减 / OB 迁移 / 知识图谱
