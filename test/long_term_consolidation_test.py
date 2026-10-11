"""Long-term consolidation eligibility, safety, apply and undo regression."""
import asyncio, json, sqlite3, tempfile, time
from pathlib import Path
from types import SimpleNamespace

from retriever_test import MemoirDB, VecStore, ThemeEmbedding
from astrbot_plugin_astra_memoir.pipeline.writer import EpisodeWriter
from astrbot_plugin_astra_memoir.pipeline.maintenance import MaintenanceManager


class CaptureProvider:
    def __init__(self): self.payloads = []
    async def text_chat(self, prompt, system_prompt):
        items = json.loads(prompt.split("<episodes>\n",1)[1].split("\n</episodes>",1)[0])
        self.payloads.extend(items)
        if len(items) >= 2:
            decisions = [{"action":"merge","ids":[x["id"] for x in items],
                "title":"Astra Memoir 长期完善","content":"陆忱持续完善 Astra Memoir 的稳定性与整理机制。",
                "importance":4,"reason":"同一持续项目"}]
        else:
            x = items[0]
            decisions = [{"action":"archive" if x["importance"] == 2 else "keep",
                "ids":[x["id"]],"importance":x["importance"],"reason":"长期价值判断"}]
        return SimpleNamespace(completion_text=json.dumps({"decisions":decisions},ensure_ascii=False))


class Extractor:
    def __init__(self, provider): self.provider=provider
    def _get_provider(self): return self.provider


class BrokenEmbedding:
    async def get_embedding(self, text): raise RuntimeError("embedding offline")


async def main():
    # Upgrade an existing CHECK(run_type IN history/nightly) table without data loss.
    with tempfile.TemporaryDirectory() as migration_tmp:
        migration_path=Path(migration_tmp)/"old.db"
        con=sqlite3.connect(migration_path)
        con.execute("""CREATE TABLE maintenance_runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,run_type TEXT NOT NULL CHECK(run_type IN ('history','nightly')),
            target_start INTEGER NOT NULL,target_end INTEGER NOT NULL,status TEXT NOT NULL,
            backup_path TEXT,source_count INTEGER NOT NULL DEFAULT 0,keep_count INTEGER NOT NULL DEFAULT 0,
            archive_count INTEGER NOT NULL DEFAULT 0,merge_source_count INTEGER NOT NULL DEFAULT 0,
            merge_result_count INTEGER NOT NULL DEFAULT 0,batch_count INTEGER NOT NULL DEFAULT 0,
            completed_batches INTEGER NOT NULL DEFAULT 0,logic_version INTEGER NOT NULL DEFAULT 1,
            created_at INTEGER NOT NULL,applied_at INTEGER,undone_at INTEGER,error TEXT)""")
        con.execute("INSERT INTO maintenance_runs(run_type,target_start,target_end,status,created_at) VALUES('history',0,1,'preview',1)")
        con.commit(); con.close()
        migrated=MemoirDB(migration_path,8); migrated.initialize()
        assert migrated.fetchone("SELECT run_type FROM maintenance_runs WHERE id=1")[0]=="history"
        migrated.execute("INSERT INTO maintenance_runs(run_type,stage,target_start,target_end,status,created_at) VALUES('long_term',1,0,1,'preview',2)")
        migrated.close()

    with tempfile.TemporaryDirectory() as tmp:
        db=MemoirDB(Path(tmp)/"memoir.db",8); db.initialize()
        vec=VecStore(db); embedding=ThemeEmbedding(); writer=EpisodeWriter(embedding,db,vec)
        provider=CaptureProvider(); manager=MaintenanceManager(db,vec,writer,Extractor(provider),batch_size=12)
        old=int(time.time())-200*86400

        async def add(title,content,importance=3,edited=0,pinned=0,stage=0,offset=0):
            eid=db.insert_episode(platform="qq",chat_type="private",session_id="project",group_id=None,
                title=title,content=content,source_raw_ids=[],event_start_at=old+offset,
                event_end_at=old+offset,extracted_at=old,importance=importance)
            db.execute("UPDATE episodes SET edited_by_user=?,pinned=?,consolidation_stage=? WHERE id=?",
                       (edited,pinned,stage,eid))
            db.insert_fts(eid,title,content,[])
            db.insert_participants(eid,[{"speaker_id":"111","speaker_name":"陆忱","role":"user"}])
            vec.upsert(eid,await embedding.get_embedding(title+content),chat_type="private",session_id="project",group_id=None)
            return eid

        edited=await add("Memoir 修复","用户修正后的真实正文",edited=1)
        second=await add("Memoir 整理","继续完善 Memoir 长期整理",offset=10)
        pinned=await add("Memoir 保护项","明确要求长期保留",pinned=1,offset=20)
        importance5=await add("关键共同经历","非常重要",importance=5,offset=30)
        low=await add("过时临时安排","一次性短期安排",importance=2,offset=40)

        counts=manager.long_term_candidate_counts()
        assert counts["1"]==4 and counts["2"]==0 and counts["3"]==0
        preview=await manager.preview_long_term(1)
        source_ids={a["source_episode_id"] for a in preview["actions"]}
        assert edited in source_ids and pinned not in source_ids
        assert any(x["content"]=="用户修正后的真实正文" for x in provider.payloads)
        assert all(a["action"]!="archive" for a in preview["actions"]
                   if a["source_episode_id"]==importance5)
        assert any(a["action"]=="archive" and a["source_episode_id"]==low
                   for a in preview["actions"])

        theme_rows=[{"id":100+i,"title":f"Memoir 插件修复{i}","content":"完善向量和整理机制",
                     "keywords":"Memoir 插件","event_start_at":old+i,"chat_type":"private",
                     "session_id":"project","group_id":None,"participants":[]} for i in range(5)]
        unrelated={"id":200,"title":"今天吃面","content":"晚饭是牛肉面","keywords":"吃饭",
                   "event_start_at":old,"chat_type":"private","session_id":"project",
                   "group_id":None,"participants":[]}
        clustered=manager._long_term_batches([*theme_rows,unrelated])
        assert any(len(group)==5 for group in clustered)
        assert any([row["id"] for row in group]==[200] for group in clustered)

        applied=await manager.apply(preview["id"])
        merged=db.fetchone("SELECT * FROM episodes WHERE created_by_run_id=?",(preview["id"],))
        assert applied["status"]=="applied" and merged
        assert merged["consolidation_stage"]==1
        assert merged["contains_user_edited_source"]==1
        assert len(db.fetchall("SELECT * FROM episode_merge_sources WHERE merged_episode_id=?",(merged["id"],)))>=2
        assert db.fetchone("SELECT status FROM episodes WHERE id=?",(edited,))["status"]=="archived"
        await manager.undo(preview["id"])
        assert db.fetchone("SELECT consolidation_stage FROM episodes WHERE id=?",(edited,))[0]==0
        assert db.fetchone("SELECT status FROM episodes WHERE id=?",(edited,))[0]=="active"
        assert db.fetchone("SELECT status FROM episodes WHERE id=?",(merged["id"],))[0]=="trashed"

        # Completed stage 1 entries no longer re-enter stage 1.
        db.execute("UPDATE episodes SET consolidation_stage=1 WHERE id=?",(second,))
        assert second not in {a["source_episode_id"] for a in (await manager.preview_long_term(1))["actions"]}

        # Editing is truth-first: embedding outage removes/stales vector but never rejects text.
        broken_writer=EpisodeWriter(BrokenEmbedding(),db,vec)
        broken_manager=MaintenanceManager(db,vec,broken_writer,Extractor(provider))
        await broken_manager.edit_episode(low,"用户修改标题","用户修改正文",3)
        changed=db.fetchone("SELECT * FROM episodes WHERE id=?",(low,))
        assert changed["content"]=="用户修改正文" and changed["edited_by_user"]==1
        assert low in vec.missing_episode_ids()

        # Explicit protection is independent from editing/restoring.
        manager.manual_state(low,"pin")
        assert db.fetchone("SELECT pinned FROM episodes WHERE id=?",(low,))[0]==1
        manager.manual_state(low,"unpin")
        assert db.fetchone("SELECT pinned FROM episodes WHERE id=?",(low,))[0]==0
        db.close()
    print("Long-term consolidation tests passed")


if __name__ == "__main__": asyncio.run(main())
