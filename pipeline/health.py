"""Persistent component diagnostics and raw-ingestion health state."""
import json
import os
import time
from pathlib import Path
from astrbot.api import logger


RAW_MATCH_TOLERANCE_SECONDS = 5
RAW_PENDING_SECONDS = 60
RAW_IDLE_SECONDS = 300


def get_raw_ingestion_health(now, last_hook_at, last_raw_at, recent_hook_count=0):
    """Classify one source using evidence, never raw age alone."""
    now = int(now)
    hook = int(last_hook_at or 0)
    raw = int(last_raw_at or 0)
    count = max(0, int(recent_hook_count or 0))
    if not hook:
        return "idle"
    if count == 0 and raw >= hook - RAW_MATCH_TOLERANCE_SECONDS:
        return "healthy" if now - hook <= RAW_IDLE_SECONDS else "idle"
    age = max(0, now - hook)
    if age < RAW_PENDING_SECONDS and not (count >= 2 and age >= 20):
        return "pending"
    return "error"


_MESSAGES = {
    "healthy": "近期消息已正常写入原文库",
    "idle": "近期暂无新消息，采集器待命中",
    "pending": "检测到新消息，等待原文入库",
    "error": "检测到消息 hook，但原文未及时入库",
}


class HealthState:
    def __init__(self, path):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self.data, dict):
                self.data = {}
        except (OSError, ValueError):
            self.data = {}
        self.data["started_at"] = int(time.time())
        self.data["ingestion_status"] = "idle"
        self.data["embedding_status"] = "degraded"

    def update(self, **values):
        self.data.update(values)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            self.data["diagnostics_write_error"] = str(exc)
            logger.exception("[Memoir] 无法持久化健康状态")

    def error(self, component, exc):
        now = int(time.time())
        errors = [*self.data.get("errors", []),
                  {"at": now, "component": component, "message": str(exc)[:1000]}][-30:]
        self.update(errors=errors, **{component + "_error": str(exc)[:1000]})

    def user_hook(self, chat_type, now=None):
        """Record a text-bearing user hook that is eligible for raw storage."""
        now = int(time.time()) if now is None else int(now)
        prefix = "group" if chat_type == "group" else "private"
        hook_total = int(self.data.get(f"{prefix}_hook_count_total", 0)) + 1
        raw_total = int(self.data.get(f"{prefix}_raw_count_total", 0))
        self.update(
            last_user_hook_at=now,
            **{f"last_{prefix}_hook_at": now,
               f"{prefix}_hook_count_total": hook_total,
               f"{prefix}_pending_hook_count": max(0, hook_total - raw_total)},
        )

    def raw_ingested(self, chat_type, now=None, *, duplicate=False):
        """Record a user raw write (or verified dedupe hit) for one source."""
        now = int(time.time()) if now is None else int(now)
        prefix = "group" if chat_type == "group" else "private"
        hook_total = int(self.data.get(f"{prefix}_hook_count_total", 0))
        raw_total = int(self.data.get(f"{prefix}_raw_count_total", 0)) + 1
        self.update(
            last_raw_ingest_at=now,
            **{f"last_{prefix}_raw_at": now,
               f"{prefix}_raw_count_total": raw_total,
               f"{prefix}_pending_hook_count": max(0, hook_total - raw_total)},
            raw_error=None,
            last_raw_skip_reason="重复消息（原文已存在）" if duplicate else None,
        )

    def watchdog(self, now=None):
        now = int(time.time()) if now is None else int(now)
        source_status = {}
        for source in ("private", "group"):
            source_status[source] = get_raw_ingestion_health(
                now,
                self.data.get(f"last_{source}_hook_at", 0),
                self.data.get(f"last_{source}_raw_at", 0),
                self.data.get(f"{source}_pending_hook_count", 0),
            )

        # Independent LLM activity can expose a user hook that stopped firing.
        # Silence without that evidence is idle, never a failure.
        last_user_hook = int(self.data.get("last_user_hook_at", 0) or 0)
        last_raw = int(self.data.get("last_raw_ingest_at", 0) or 0)
        llm = int(self.data.get("last_llm_activity_at", 0) or 0)
        missing_hook = (llm > max(last_user_hook, last_raw) + RAW_PENDING_SECONDS
                        and now - llm <= RAW_IDLE_SECONDS)

        if self.data.get("raw_error") or missing_hook or "error" in source_status.values():
            status = "error"
        elif "pending" in source_status.values():
            status = "pending"
        elif "healthy" in source_status.values():
            status = "healthy"
        else:
            status = "idle"

        message = (_MESSAGES["error"] + ("；AstrBot 有活动但用户 hook 未更新" if missing_hook else "")
                   if status == "error" else _MESSAGES[status])
        warning = message if status == "error" else ""
        values = {
            "ingestion_status": status,
            "ingestion_message": message,
            "private_ingestion_status": source_status["private"],
            "group_ingestion_status": source_status["group"],
            "watchdog_warning": warning,
        }
        if any(self.data.get(k) != v for k, v in values.items()):
            self.update(**values)
            if status == "error":
                logger.error("[Memoir] watchdog: %s", message)
        return dict(self.data)
