"""Phase 1: bounded extraction, audit atomicity, migration and retention."""
import asyncio
import json
import re
import time
from types import SimpleNamespace
from unittest.mock import patch

from phase1f_fix_test import build_stack, FailingEmbeddingProvider
from astrbot_plugin_astra_memoir.pipeline.extractor import calculate_memory_budget
from astrbot_plugin_astra_memoir.pipeline import panel as panel_mod


class Provider:
    id = 'budget-test'
    model = 'fixture-model'

    def __init__(self, importance, invalid=False, duplicate=False):
        self.importance, self.invalid, self.duplicate = importance, invalid, duplicate
        self.calls = 0

    async def text_chat(self, prompt, system_prompt):
        self.calls += 1
        assert 'memory_budget' in prompt
        assert '三天、一周或一个月' in system_prompt
        block = prompt.split('<new_messages>')[1].split('</new_messages>')[0]
        ids = [int(x) for x in re.findall(r'\[raw_id=(\d+)\]', block)]
        events = [dict(title=f'独立主题{i}', content='同一件事情' if self.duplicate else f'第{i}个有价值的新结果',
                       importance=v, source_raw_ids=[ids[i % len(ids)]], keywords=[])
                  for i, v in enumerate(self.importance)]
        if self.invalid:
            events[-1]['source_raw_ids'] = [999999]
        return SimpleNamespace(completion_text=json.dumps(dict(events=events,
            discard_summary='普通闲聊和重复画图请求无长期信息'), ensure_ascii=False))


def insert(db, count, session='budget', role='user', processed=False):
    ids = []
    for i in range(count):
        rid = db.insert_raw_message(dedupe_key=f'{session}:{role}:{i}', platform='qq',
            chat_type='group', session_id=session, group_id='g', platform_message_id=str(i),
            speaker_id='111' if role == 'user' else '999', speaker_name='陆忱' if role == 'user' else 'Astra',
            role=role, content='普通闲聊画图', reply_to_id=None, trigger_raw_id=None, created_at=int(time.time()))
        ids.append(rid)
    if processed:
        db.mark_processed(ids, int(time.time()))
    return ids


async def case(importance, expected, *, invalid=False, duplicate=False, embedding_failure=False):
    llm = Provider(importance, invalid, duplicate)
    db, vec, _, _, scheduler, _ = await build_stack(llm, FailingEmbeddingProvider() if embedding_failure else None)
    try:
        insert(db, 4, role='assistant', processed=True)  # overlap must not inflate budget
        insert(db, 20)
        await scheduler.process_batch('budget', mode='threshold')
        run = dict(db.fetchone('SELECT * FROM extraction_runs'))
        assert run['memory_budget'] == 2 and run['context_overlap_count'] == 4
        assert run['new_raw_count'] == 20 and run['stored_count'] == expected, run
        assert db.fetchone('SELECT COUNT(*) FROM episodes')[0] == expected
        assert run['model'] == 'fixture-model'
        assert run['completed_at'] is not None
        unprocessed = db.fetchone('SELECT COUNT(*) FROM recent_messages WHERE processed_at IS NULL')[0]
        if invalid:
            assert run['status'] == 'failed' and unprocessed == 20 and run['error']
        else:
            assert run['status'] == 'success' and unprocessed == 0
            assert llm.calls == 1, 'over-budget responses must not cause another LLM call'
            assert run['generated_count'] == len(importance)
            assert run['skipped_importance1_count'] == importance.count(1)
            assert run['discard_summary']
        if importance == [1, 2, 5, 4, 3] and not invalid:
            assert run['budget_trimmed_count'] == 2, run
            assert sorted(r[0] for r in db.fetchall('SELECT importance FROM episodes')) == [4, 5]
        if duplicate:
            assert run['duplicate_count'] == 2 and run['budget_trimmed_count'] == 0

        panel = panel_mod.MemoirPanel(db, vec, None, scheduler, scheduler.writer,
                                     scheduler.writer.embedding_provider, 'fixture', 8)
        stats = await panel.get_stats()
        assert stats['status'] == 'ok', stats
        assert stats['data']['extraction_today']['runs'] == 1
        assert stats['data']['extraction_today']['stored'] == expected
        assert stats['data']['episodes_per_100_raw'] == (None if invalid else expected*5)
        with patch.object(panel_mod, 'astr_request', SimpleNamespace(query={'offset': '0'})):
            assert (await panel.extraction_runs())['data'][0]['run_id'] == run['run_id']

        # Upgrade/restart leaves all truth and indexes intact, and marks stale runs interrupted.
        before = {t: db.fetchone(f'SELECT COUNT(*) FROM {t}')[0]
                  for t in ('episodes', 'episode_vec', 'episodes_fts', 'recent_messages')}
        db.execute("INSERT INTO extraction_runs(session_id,source_chat_type,started_at,new_raw_count,"
                   "context_overlap_count,memory_budget) VALUES('stale','private',0,1,0,1)")
        db.close()


        db.initialize()
        assert {t: db.fetchone(f'SELECT COUNT(*) FROM {t}')[0] for t in before} == before
        assert db.fetchone("SELECT status FROM extraction_runs WHERE session_id='stale'")[0] == 'interrupted'
        db.execute('UPDATE extraction_runs SET completed_at=? WHERE run_id=?', (int(time.time())-31*86400, run['run_id']))
        await scheduler._ttl_cleanup()
        assert db.fetchone('SELECT 1 FROM extraction_runs WHERE run_id=?', (run['run_id'],)) is None
        assert db.fetchone('SELECT COUNT(*) FROM episodes')[0] == expected
    finally:
        db.close()


async def rollback_case():
    db, _, _, _, scheduler, _ = await build_stack(Provider([4]))
    try:
        insert(db, 20)
        db.execute("CREATE TRIGGER audit_failure BEFORE UPDATE OF status ON extraction_runs "
                   "WHEN NEW.status='success' BEGIN SELECT RAISE(ABORT,'simulated audit commit failure'); END")
        await scheduler.process_batch('budget', mode='threshold')
        assert db.fetchone('SELECT COUNT(*) FROM episodes')[0] == 0
        assert db.fetchone('SELECT COUNT(*) FROM recent_messages WHERE processed_at IS NULL')[0] == 20
        assert db.fetchone('SELECT status FROM extraction_runs')[0] == 'failed'
    finally:
        db.close()


async def main():
    assert [calculate_memory_budget('group', n) for n in (0,10,11,20,21,40,41,500)] == [0,1,2,2,3,3,4,4]
    assert calculate_memory_budget('private',8,4,True) == 1
    assert calculate_memory_budget('private',20,10,False) == 2
    assert calculate_memory_budget('private',40,20,True) == 3
    await case([], 0)
    await case([1]*5, 0)
    await case([1,2,5,4,3], 2)
    await case([5,4,3], 0, invalid=True)
    await case([3,3,3], 1, duplicate=True)
    await case([4], 1, embedding_failure=True)
    await rollback_case()
    print('Extraction budget/audit/migration/retention tests passed')


if __name__ == '__main__':
    asyncio.run(main())
