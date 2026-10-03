"""Offline, evidence-only raw backfill. Preview by default; never reads LLM summaries.

Input JSON array fields: platform, chat_type, session_id, group_id (null for private),
platform_message_id, speaker_id, speaker_name, role='user', content, created_at
(Unix seconds), source_ref (export file / upstream evidence reference).
"""
import argparse
import hashlib
import json
import sqlite3
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage.identity import IdentityStore


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('时间范围必须带时区，如2026-10-03T02:34:00+08:00')
    return int(parsed.timestamp())


def validate(payload, start, end):
    if not isinstance(payload, list) or not 0 < len(payload) <= 10000 or end <= start:
        raise ValueError('要求1~10000条真实原文和有效时间范围')
    records = {}
    fields = ('platform', 'session_id', 'platform_message_id', 'speaker_id', 'speaker_name', 'content', 'source_ref')
    for row in payload:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in fields):
            raise ValueError('缺少真实平台/会话/消息ID/人物/正文/来源，不允许推测补齐')
        if row.get('role') != 'user' or row.get('chat_type') not in ('private','group'):
            raise ValueError('本工具只补用户原消息；不猜测assistant与触发消息的关联')
        if row['chat_type'] == 'group' and (not isinstance(row.get('group_id'), str) or not row['group_id']):
            raise ValueError('群消息必须有真实group_id')
        if row['chat_type'] == 'private' and row.get('group_id'):
            raise ValueError('私聊不得填写group_id')
        ts = row.get('created_at')
        if type(ts) is not int or not start <= ts < end or ts > time.time()+60:
            raise ValueError('消息时间缺失、超范围或位于未来；拒绝用当前时间代填')
        key = f"user:{row['session_id']}:{row['platform_message_id']}"
        if key in records and records[key] != row:
            raise ValueError('导出内相同消息ID对应不同内容')
        records[key] = row
    return records


def backfill(db_path, source, start, end, *, apply=False, expected_sha=None):
    source_bytes = Path(source).read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    records = validate(json.loads(source_bytes.decode('utf-8-sig')), start, end)
    if apply and expected_sha != digest:
        raise ValueError('请先预览，再用预览返回的sha256确认同一份来源文件')
    path = Path(db_path).resolve(strict=True)
    conn = sqlite3.connect(path.as_uri() + ('?mode=rw' if apply else '?mode=ro'), uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        pending = []
        for key, row in records.items():
            old = conn.execute('SELECT * FROM recent_messages WHERE dedupe_key=?', (key,)).fetchone()
            if old:
                if any(old[k] != row.get(k) for k in ('content','speaker_id','chat_type','group_id','platform')):
                    raise ValueError(f'现存消息与来源冲突，拒绝覆盖：{key}')
            else:
                pending.append((key,row))
        report = dict(sha256=digest, input_count=len(records), new_count=len(pending),
                      duplicate_count=len(records)-len(pending), applied=False)
        if not apply or not pending:
            return report
        backup = path.with_name(f'memoir-before-backfill-{uuid.uuid4().hex}.db')
        target = sqlite3.connect(backup)
        try:
            conn.backup(target)
        finally:
            target.close()
        adapter = SimpleNamespace(execute=conn.execute,
                                  fetchone=lambda sql,p=(): conn.execute(sql,p).fetchone())
        identities = IdentityStore(adapter)
        conn.execute('BEGIN IMMEDIATE')
        try:
            conn.execute('CREATE TABLE IF NOT EXISTS raw_backfill_audit('
                         'id INTEGER PRIMARY KEY,source_sha256 TEXT,source_path TEXT,backup_path TEXT,created_at INTEGER,inserted_count INTEGER)')
            for key, row in pending:
                conn.execute('INSERT INTO recent_messages(dedupe_key,platform,chat_type,session_id,group_id,'
                             'platform_message_id,speaker_id,speaker_name,role,content,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                             (key,*[row.get(k) for k in ('platform','chat_type','session_id','group_id',
                               'platform_message_id','speaker_id','speaker_name','role','content','created_at')]))
                identities.observe(row['speaker_id'],row['speaker_name'],'user')
            conn.execute('INSERT INTO raw_backfill_audit(source_sha256,source_path,backup_path,created_at,inserted_count) VALUES(?,?,?,?,?)',
                         (digest,str(Path(source).resolve()),str(backup),int(time.time()),len(pending)))
            conn.execute('COMMIT')
        except BaseException:
            conn.execute('ROLLBACK')
            raise
        return {**report, 'applied':True, 'backup_path':str(backup)}
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--astrbot-stopped', action='store_true')
    parser.add_argument('--confirm-sha256')
    args = parser.parse_args()
    if args.apply and not args.astrbot_stopped:
        parser.error('应用补录前必须停止AstrBot，确认后传入--astrbot-stopped')
    result = backfill(args.db,args.source,timestamp(args.start),timestamp(args.end),
                      apply=args.apply,expected_sha=args.confirm_sha256)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
