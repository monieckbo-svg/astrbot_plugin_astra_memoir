# 原文采集中断修复与验收

## 已确认的代码问题

旧启动流程先解析/探测 embedding，再建立 SQLite 和 raw cache。provider 暂时不存在或请求失败会使初始化退出，后续消息 hook 的 ready 检查一直失败；原先没有自动恢复任务。状态页 embedding 状态是固定 ok，scheduler 只检查任务对象存在，提取时间来自 episodes.extracted_at（包含整理生成的记忆）。这些状态不能证明原文采集正常。

这是可复现的代码缺陷，不是对 2026-10-03 02:34 现场原因的最终证明：需要对应 AstrBot 日志、运行库和平台历史来确认当时事件。

## 本次行为

- 先建立原文/正文数据库、采集和调度，再在后台尝试 embedding。
- provider 不可用时显示 degraded，约每 30 秒重试；单次远程探测有超时，不阻塞 raw。
- 写入正文和 FTS 不依赖 embedding；恢复后每轮最多补 20 条缺失向量。
- 高优先级用户 hook 记录观测时间；LLM hook 是额外活动信号。5 分钟仍有活动但没有 raw 增量时告警。如果所有 hook 都没有信号，只能显示 unknown，不能证明 AstrBot 没有聊天。
- 状态页拆分采集、提取、调度、provider、索引，显示 7 个时间戳。有效提取时间排除空输出及整理写入。
- 采集错误持久保存到数据目录 memoir-health.json。磁盘本身不可写时仍记录日志和内存错误，不能保证错误文件落盘。
- 卸载等待后台任务和 SQLite 工作结束，避免取消线程任务后过早释放数据库锁。

## 更新后的现场验收

1. 更新插件并重载，打开状态页。私聊和群聊各发一条带文本的新消息，确认 hook/raw/对应来源时间都更新，并在近期原文中找到它们。
2. 临时让所选 embedding 不可用：provider 应 degraded，新消息仍出现在近期原文；正文可以产生缺失向量。
3. 恢复同一个 provider，等自动重试与补向量，确认 coverage 恢复。不要修改模型 ID 来掩盖故障。
4. 没有新消息时，有效提取时间不应因 scheduler tick 或历史整理而变化。
5. 若仍停止，提供 memoir-health.json、同时间段 AstrBot 日志与状态截图，注意先脱敏。

## 只读历史检查与安全补录

状态页“检查 AstrBot 历史（只读）”输入原文中的 session_id 和实际平台实例 ID，可分页检查平台保存的消息。LLM 会话历史仅报告是否存在，不等于完整聊天证据。平台历史的内部 history_id 不能冒充 QQ 的原始 message_id。

能否恢复 02:34 之后的消息取决于 AstrBot/QQ 适配器实际保留了什么；本地代码测试没有访问线上历史，不能承诺可恢复。

拿到真实导出后使用 tools/backfill_raw.py。JSON 必须是数组，每条包含 platform、chat_type、session_id、group_id（私聊 null）、platform_message_id、speaker_id、speaker_name、role=user、content、created_at（原始 Unix 秒）、source_ref（证据文件出处）。不接受推测的身份、消息 ID 或时间，不补猜测的机器人回复。

在插件目录中预览（替换路径及实际结束时间，范围左闭右开）：

```sh
python tools/backfill_raw.py --db /真实路径/memoir.db --source /真实路径/messages.json --start 2026-10-03T02:34:00+08:00 --end 2026-10-04T00:00:00+08:00
```

检查新增/重复数量和来源内容，停止 AstrBot 后，在相同命令后追加 `--apply --astrbot-stopped --confirm-sha256 预览输出的sha256`。工具先用 SQLite backup 备份，再事务插入未处理 raw，保留原始时间，保存补录审计；相同消息不会重复插入，冲突拒绝覆盖。重启插件后由正常批处理/idle 提取处理，不直接捏造 episode。

本工具要求离线应用。不要在 AstrBot 仍运行时传入“已停止”确认，也不要直接用聊天总结或截图 OCR 凑字段。
