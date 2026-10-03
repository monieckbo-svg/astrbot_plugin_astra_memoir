"""Persistent diagnostics kept outside SQLite so DB failures remain visible."""
import json
import os
import time
from pathlib import Path
from astrbot.api import logger


class HealthState:
    def __init__(self, path):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(self.data, dict):
                self.data = {}
        except (OSError, ValueError):
            self.data = {}
        self.data['started_at'] = int(time.time())
        self.data['ingestion_status'] = 'waiting'
        self.data['embedding_status'] = 'degraded'

    def update(self, **values):
        self.data.update(values)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding='utf-8')
            os.replace(tmp, self.path)
        except OSError as exc:
            self.data['diagnostics_write_error'] = str(exc)
            logger.exception('[Memoir] 无法持久化健康状态')

    def error(self, component, exc):
        now = int(time.time())
        errors = [*self.data.get('errors', []),
                  {'at': now, 'component': component, 'message': str(exc)[:1000]}][-30:]
        self.update(errors=errors, **{component + '_error': str(exc)[:1000]})

    def watchdog(self, now=None):
        now = int(time.time()) if now is None else now
        raw = self.data.get('last_raw_ingest_at', 0)
        signal = max(self.data.get('last_message_hook_seen_at', 0),
                     self.data.get('last_llm_activity_at', 0))
        eligible = self.data.get('raw_pending_since', 0)
        started = self.data['started_at']
        # Independent LLM hook activity can expose a missing user hook. No signal
        # at all is "unknown", not proof that AstrBot is receiving messages.
        lost = signal > raw and now - max(raw, started) >= 300 and now - signal <= 300
        pending = eligible and now - eligible >= 300
        if pending or (lost and self.data.get('last_llm_activity_at', 0) > raw):
            status, warning = 'stalled', '检测到聊天/LLM活动，但原文超过5分钟未成功新增，请检查hook和写库错误。'
        elif self.data.get('raw_error'):
            status, warning = 'error', self.data['raw_error']
        elif not signal or now - signal > 300:
            status, warning = 'unknown', '近期没有可观测的消息信号，无法确认采集正常。'
        else:
            status, warning = 'observed', ''
        if warning != self.data.get('watchdog_warning') or status != self.data.get('ingestion_status'):
            self.update(ingestion_status=status, watchdog_warning=warning)
            if status == 'stalled':
                logger.error('[Memoir] watchdog: %s', warning)
        return dict(self.data)
