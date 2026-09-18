"""身份卡：QQ 主键、别名、人工字段、合并和提取提示。"""
import asyncio
import tempfile
from pathlib import Path

from retriever_test import MemoirDB
from astrbot_plugin_astra_memoir.pipeline.extractor import build_user_prompt


def raw(db, mid, qq, name, role="user"):
    return db.insert_raw_message(
        dedupe_key=mid, platform="qq", chat_type="private", session_id="s",
        group_id=None, platform_message_id=mid, speaker_id=qq,
        speaker_name=name, role=role, content="测试消息", reply_to_id=None,
        trigger_raw_id=None, created_at=1,
    )


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "memoir.db"
        db = MemoirDB(path, embedding_dim=8)
        db.initialize()
        first = raw(db, "a", "111", "陆忱")
        raw(db, "b", "111", "喵日天")
        raw(db, "c", "222", "星星", "assistant")
        assert len(db.identities.list_all()) == 2
        card = db.identities.get("111")
        assert card["canonical_name"] == "陆忱"
        assert card["pronoun"] == "TA" and card["person_type"] == "human"
        assert set(card["aliases"]) == {"陆忱", "喵日天"}
        assert db.identities.get("222")["person_type"] == "AI"
        assert db.identities.get("222")["pronoun"] == "TA"

        db.identities.update_many([{
            "qq_id": "111", "canonical_name": "正式陆忱", "pronoun": "她",
            "person_type": "human", "aliases": ["陆忱", "喵日天"],
        }])
        raw(db, "d", "111", "又一个昵称")
        assert db.identities.get("111")["canonical_name"] == "正式陆忱"
        assert db.identities.get("111")["pronoun"] == "她"
        assert "又一个昵称" in db.identities.get("111")["aliases"]
        row = db.fetchone("SELECT * FROM recent_messages WHERE id=?", (first,))
        prompt = build_user_prompt([row], [], "private",
                                   {"111": db.identities.display("111")})
        assert "[qq=111][person=正式陆忱][pronoun=她]" in prompt

        raw(db, "e", "333", "误认马甲")
        merged = db.identities.merge("333", "111")
        assert merged["canonical_name"] == "正式陆忱"
        assert "误认马甲" in merged["aliases"]
        assert "333" in merged["linked_qq_ids"]
        assert db.identities.display("333")[0] == "正式陆忱"
        assert len(db.identities.list_all()) == 2
        assert db.fetchone("SELECT speaker_id FROM recent_messages WHERE dedupe_key='e'")[0] == "333"
        db.close()

        db = MemoirDB(path, embedding_dim=8)
        db.initialize()
        assert db.identities.get("111")["canonical_name"] == "正式陆忱"
        assert db.identities.display("333")[0] == "正式陆忱"
        db.close()
    print("Identity test passed")


if __name__ == "__main__":
    main()
