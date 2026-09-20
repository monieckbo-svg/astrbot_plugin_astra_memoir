"""Preview/backup/apply/undo/edit/state flow regression."""
import asyncio, json, tempfile, time
from pathlib import Path
from types import SimpleNamespace

from retriever_test import MemoirDB, VecStore, ThemeEmbedding
from astrbot_plugin_astra_memoir.pipeline.writer import EpisodeWriter
from astrbot_plugin_astra_memoir.pipeline.maintenance import MaintenanceManager


class Provider:
    async def text_chat(self, prompt, system_prompt):
        items = json.loads(prompt.split("<episodes>\n",1)[1].split("\n</episodes>",1)[0])
        ids = [x["id"] for x in items]
        if len(ids) >= 3:
            decisions = [
                {"action":"merge","ids":ids[:2],"title":"持续插件开发",
                 "content":"我和陆忱持续完善了插件。","importance":4,"reason":"同一持续事件"},
                *[{"action":"archive","ids":[i],"importance":1,"reason":"一次性闲聊"} for i in ids[2:]],
            ]
        else:
            decisions = [{"action":"keep","ids":[i],"importance":3,"reason":"仍有价值"} for i in ids]
        return SimpleNamespace(completion_text=json.dumps({"decisions": decisions}, ensure_ascii=False))


class Extractor:
    def _get_provider(self): return Provider()


async def main():
    with tempfile.TemporaryDirectory() as tmp:
        db = MemoirDB(Path(tmp)/"memoir.db", 8); db.initialize()
        vec, emb = VecStore(db), ThemeEmbedding(); writer = EpisodeWriter(emb, db, vec)
        now = int(time.time()) - 86400
        ids=[]
        for i, (title, content) in enumerate((("插件开发1","修复面板"),("插件开发2","继续修复检索"),("玩笑","发了个一次性玩笑"))):
            eid=db.insert_episode(platform="qq",chat_type="private",session_id="s",group_id=None,
                title=title,content=content,source_raw_ids=[],event_start_at=now+i,
                event_end_at=now+i,extracted_at=now,importance=3)
            db.insert_fts(eid,title,content,[])
            vec.upsert(eid,await emb.get_embedding(title+content),chat_type="private",session_id="s",group_id=None)
            ids.append(eid)
        manager=MaintenanceManager(db,vec,writer,Extractor(),batch_size=12)
        scale_rows=[{"id":1000+i,"title":f"主题{i%9}","content":f"内容{i%9}","keywords":"",
                     "event_start_at":now+(i%3)*86400,"chat_type":"private","session_id":"s","group_id":None}
                    for i in range(469)]
        scale_batches=manager._small_batches(scale_rows)
        assert sum(map(len,scale_batches))==469 and max(map(len,scale_batches))<=12
        preview=await manager.preview(now-10,now+100,run_type="history")
        assert preview["status"]=="preview" and preview["source_count"]==3
        assert Path(preview["backup_path"]).exists()
        assert all(db.fetchone("SELECT status FROM episodes WHERE id=?",(i,))[0]=="active" for i in ids)
        applied=await manager.apply(preview["id"])
        assert applied["status"]=="applied"
        merged=db.fetchone("SELECT * FROM episodes WHERE created_by_run_id=?",(preview["id"],))
        assert merged and merged["status"]=="active"
        assert len(db.fetchall("SELECT * FROM episode_merge_sources WHERE merged_episode_id=?",(merged["id"],)))==2
        assert all(db.fetchone("SELECT status FROM episodes WHERE id=?",(i,))[0]=="archived" for i in ids)
        await manager.undo(preview["id"])
        assert all(db.fetchone("SELECT status FROM episodes WHERE id=?",(i,))[0]=="active" for i in ids)
        assert db.fetchone("SELECT status FROM episodes WHERE id=?",(merged["id"],))[0]=="trashed"

        preview2=await manager.preview(now-10,now+100,run_type="history")
        await manager.edit_episode(ids[0],"用户标题","用户修正文",5)
        ep=db.fetchone("SELECT * FROM episodes WHERE id=?",(ids[0],))
        assert ep["edited_by_user"]==1 and ep["protected_until"]>int(time.time())
        assert db.fetchone("SELECT COUNT(*) FROM episode_versions WHERE episode_id=?",(ids[0],))[0]==1
        try:
            await manager.apply(preview2["id"])
            raise AssertionError("Preview 后的用户编辑必须阻止应用")
        except ValueError as exc:
            assert "已变化" in str(exc)
        await manager.undo_latest_edit(ids[0])
        assert db.fetchone("SELECT title FROM episodes WHERE id=?",(ids[0],))[0]=="插件开发1"
        manager.manual_state(ids[0],"trash")
        assert db.fetchone("SELECT status FROM episodes WHERE id=?",(ids[0],))[0]=="trashed"
        manager.manual_state(ids[0],"restore")
        ep=db.fetchone("SELECT * FROM episodes WHERE id=?",(ids[0],))
        assert ep["status"]=="active" and ep["restored_by_user"]==1 and ep["protected_until"]>int(time.time())
        db.close()
    print("Maintenance test passed")


if __name__ == "__main__": asyncio.run(main())
