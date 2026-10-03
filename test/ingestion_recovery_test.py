"""Reproduce provider unavailable at startup/reload without a real QQ connection."""
import asyncio
import tempfile
import time
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from phase1f_fix_test import SimpleGoodProvider, OkEmbeddingProvider
from astrbot_plugin_astra_memoir.main import AstraMemoir
from astrbot_plugin_astra_memoir.pipeline.health import HealthState
from astrbot_plugin_astra_memoir.storage import MemoirDB
from astrbot_plugin_astra_memoir.tools.backfill_raw import backfill


class Context:
    def __init__(self):
        self.embedding = None
        self.routes = {}
    def get_all_embedding_providers(self):
        return [self.embedding] if self.embedding else []
    def get_using_provider(self): return SimpleGoodProvider()
    def register_web_api(self, path, handler, **kwargs): self.routes[path] = handler


class Event:
    unified_msg_origin = 'qq:GroupMessage:123'
    def __init__(self, mid): self.message_obj = SimpleNamespace(message_id=mid)
    def get_message_str(self): return '插件配置有一个新的决定'
    def get_group_id(self): return '123'
    def get_platform_name(self): return 'qq'
    def get_self_id(self): return '999'
    def get_sender_id(self): return '111'
    def get_sender_name(self): return '陆忱'


async def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        with patch('astrbot_plugin_astra_memoir.main.StarTools.get_data_dir', return_value=root), \
             patch('astrbot_plugin_astra_memoir.main.RECOVERY_INTERVAL_SECONDS', 0.01):
            ctx = Context()
            app = AstraMemoir(ctx, {'embedding_provider_id':'ok', 'nightly_maintenance_enabled':False})
            await app._startup_task
            assert app.startup_error is None
            await app.on_user_message(Event('first'))
            assert app.db.fetchone('SELECT COUNT(*) FROM recent_messages')[0] == 1
            await app.scheduler.process_batch(Event.unified_msg_origin)
            assert app.db.fetchone('SELECT COUNT(*) FROM episodes')[0] == 1
            assert app.db.fetchone('SELECT COUNT(*) FROM episodes_fts')[0] == 1
            assert len(app.vec.missing_episode_ids()) == 1
            stats = (await app.panel.get_stats())['data']
            assert stats['embedding_status'] == 'degraded'
            assert stats['health']['last_raw_ingest_at']
            assert stats['health']['last_nonempty_extraction_at']

            ctx.embedding = OkEmbeddingProvider()
            for _ in range(100):
                if app.embedding_runtime.provider is ctx.embedding and app.vec.count() == 1:
                    break
                await asyncio.sleep(0.01)
            assert app.vec.count() == 1 and not app.vec.missing_episode_ids()

            # A hot-replaced provider is not silently used until revalidated.
            ctx.embedding = OkEmbeddingProvider()
            try:
                await app.embedding_runtime.get_embedding('test')
                assert False
            except RuntimeError:
                pass
            await app.on_user_message(Event('second'))
            assert app.db.fetchone('SELECT COUNT(*) FROM recent_messages')[0] == 2
            assert await app.embedding_runtime.recover()

            # Empty success may advance attempt time but never nonempty time.
            last = (await app.panel.get_stats())['data']['last_extracted_at']
            app.db.execute("INSERT INTO extraction_runs(session_id,source_chat_type,started_at,completed_at,"
                           "new_raw_count,context_overlap_count,memory_budget,status) VALUES('empty','private',?,?,?,?,?,'success')",
                           (int(time.time())+10,int(time.time())+10,1,0,1))
            assert (await app.panel.get_stats())['data']['last_extracted_at'] == last

            with patch.object(app.db, 'insert_raw_message', side_effect=OSError('disk full')):
                await app.on_user_message(Event('bad'))
            assert 'disk full' in app.health.data['raw_error']
            persisted = HealthState(root/'memoir-health.json')
            assert persisted.data['errors'][-1]['message'] == 'disk full'
            now = int(time.time())
            persisted.data.update(started_at=now-600,last_raw_ingest_at=now-600,
                                  last_llm_activity_at=now,raw_pending_since=now-600)
            assert persisted.watchdog(now)['ingestion_status'] == 'stalled'
            await app.terminate()
            assert app._supervisor_task.done() and app._watchdog_task.done()

            # Reload with unavailable provider retains existing vectors and hooks still write.
            ctx.embedding = None
            reloaded = AstraMemoir(ctx, {'embedding_provider_id':'ok','nightly_maintenance_enabled':False})
            await reloaded._startup_task
            assert reloaded.vec.count() == 1
            await reloaded.on_user_message(Event('after-reload'))
            assert reloaded.db.fetchone('SELECT COUNT(*) FROM recent_messages')[0] == 3
            assert ctx.routes['/astrbot_plugin_astra_memoir/stats'].__closure__
            await reloaded.terminate()

        # Offline preview is read-only, imports are backed up and idempotent.
        now = int(time.time())
        source = root/'export.json'
        source.write_text(json.dumps([dict(platform='qq',chat_type='group',session_id='qq:GroupMessage:123',
            group_id='123',platform_message_id='original-id',speaker_id='111',speaker_name='陆忱',
            role='user',content='真实导出原文',created_at=now-3600,source_ref='platform-export:original-id')]), encoding='utf-8')
        preview = backfill(root/'memoir.db',source,now-7200,now)
        assert preview['new_count'] == 1 and not preview['applied']
        applied = backfill(root/'memoir.db',source,now-7200,now,apply=True,expected_sha=preview['sha256'])
        assert Path(applied['backup_path']).exists()
        assert backfill(root/'memoir.db',source,now-7200,now)['new_count'] == 0
        try:
            backfill(root/'memoir.db',source,now-7200,now,apply=True,expected_sha='wrong')
            assert False
        except ValueError:
            pass

        # Cancellation must finish the SQLite thread before releasing its lock.
        db = MemoirDB(root/'cancel.db', 8)
        db.initialize()
        started, release = threading.Event(), threading.Event()
        def operation():
            started.set()
            release.wait(2)
            db.set_meta('worker_finished', 'yes')
        task = asyncio.create_task(db.run(operation))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.01)
        assert db._lock.locked()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert db.get_meta('worker_finished') == 'yes'
        db.close()

        # Extension failure is also independent of ordinary SQLite ingestion.
        import astrbot_plugin_astra_memoir.storage.db as db_module
        with patch.object(db_module, 'sqlite_vec', None):
            db = MemoirDB(root/'no-vec.db', 0)
            db.initialize(defer_vectors=True)
            assert not db.vector_ready
            assert db.fetchone('SELECT COUNT(*) FROM recent_messages')[0] == 0
            db.close()
    print('Ingestion/provider recovery/reload/watchdog tests passed')


if __name__ == '__main__': asyncio.run(main())
