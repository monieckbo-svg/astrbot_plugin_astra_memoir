"""QQ 身份卡。昵称是别名；合并保留原始 QQ 证据并建立显式映射。"""
from __future__ import annotations

from typing import Any


class IdentityStore:
    def __init__(self, db):
        self.db = db

    @staticmethod
    def _qq(value: Any) -> str:
        qq = str(value or "").strip()
        if not qq or qq.lower() in ("bot", "unknown", "none"):
            raise ValueError("QQ 号不能为空")
        return qq

    def resolve(self, qq_id: Any) -> str:
        qq = self._qq(qq_id)
        seen = set()
        while True:
            if qq in seen:
                raise ValueError("身份合并映射存在循环")
            seen.add(qq)
            row = self.db.fetchone(
                "SELECT target_qq_id FROM identity_redirects WHERE source_qq_id = ?", (qq,)
            )
            if row is None:
                return qq
            qq = row["target_qq_id"]

    def observe(self, qq_id: Any, nickname: str, role: str = "user") -> None:
        try:
            original = self._qq(qq_id)
        except ValueError:
            return
        qq = self.resolve(original)
        name = str(nickname or "").strip() or original
        self.db.execute(
            "INSERT OR IGNORE INTO identities(qq_id, canonical_name, person_type) VALUES (?, ?, ?)",
            (qq, name, "AI" if role == "assistant" else "human"),
        )
        if name != qq:
            self.db.execute(
                "INSERT OR IGNORE INTO identity_aliases(qq_id, alias) VALUES (?, ?)",
                (qq, name),
            )

    def backfill(self) -> None:
        """从已有 raw 与 participant 快照恢复人物；不重评人工字段。"""
        for table in ("recent_messages", "episode_participants"):
            for row in self.db.fetchall(
                f"SELECT speaker_id, speaker_name, role FROM {table} ORDER BY rowid"
            ):
                self.observe(row["speaker_id"], row["speaker_name"], row["role"])

    def get(self, qq_id: Any) -> dict | None:
        try:
            qq = self.resolve(qq_id)
        except ValueError:
            return None
        row = self.db.fetchone("SELECT * FROM identities WHERE qq_id = ?", (qq,))
        if row is None:
            return None
        aliases = self.db.fetchall(
            "SELECT alias FROM identity_aliases WHERE qq_id = ? ORDER BY alias", (qq,)
        )
        linked = self.db.fetchall(
            "SELECT source_qq_id FROM identity_redirects WHERE target_qq_id = ? ORDER BY source_qq_id",
            (qq,),
        )
        return {**dict(row), "aliases": [r["alias"] for r in aliases],
                "linked_qq_ids": [r["source_qq_id"] for r in linked]}

    def list_all(self) -> list[dict]:
        rows = self.db.fetchall(
            "SELECT qq_id FROM identities WHERE qq_id NOT IN "
            "(SELECT source_qq_id FROM identity_redirects) ORDER BY canonical_name, qq_id"
        )
        return [self.get(r["qq_id"]) for r in rows]

    def display(self, qq_id: Any, fallback: str = "") -> tuple[str, str]:
        card = self.get(qq_id)
        if card is None:
            return fallback or str(qq_id), "TA"
        return card["canonical_name"], card["pronoun"]

    def update_many(self, changes: list[dict]) -> list[dict]:
        if not isinstance(changes, list) or not changes or len(changes) > 200:
            raise ValueError("请提供 1～200 张身份卡")
        prepared = []
        for item in changes:
            qq = self.resolve(item.get("qq_id"))
            if self.get(qq) is None:
                raise ValueError(f"人物不存在: {qq}")
            name = str(item.get("canonical_name", "")).strip()
            pronoun = item.get("pronoun")
            person_type = item.get("person_type")
            aliases = item.get("aliases")
            if not name or len(name) > 100:
                raise ValueError("正式名字需为 1～100 字")
            if pronoun not in ("TA", "她", "他"):
                raise ValueError("代词只能为 TA / 她 / 他")
            if person_type not in ("human", "AI"):
                raise ValueError("类型只能为 human / AI")
            if not isinstance(aliases, list) or len(aliases) > 100:
                raise ValueError("别名必须是最多 100 项的数组")
            cleaned = sorted({str(a).strip() for a in aliases if str(a).strip()})
            if any(len(a) > 100 for a in cleaned):
                raise ValueError("单个别名不能超过 100 字")
            prepared.append((qq, name, pronoun, person_type, cleaned))
        with self.db.transaction():
            for qq, name, pronoun, person_type, aliases in prepared:
                self.db.execute(
                    "UPDATE identities SET canonical_name=?, pronoun=?, person_type=? WHERE qq_id=?",
                    (name, pronoun, person_type, qq),
                )
                self.db.execute("DELETE FROM identity_aliases WHERE qq_id=?", (qq,))
                self.db.executemany(
                    "INSERT INTO identity_aliases(qq_id, alias) VALUES (?, ?)",
                    [(qq, alias) for alias in aliases],
                )
        return [self.get(qq) for qq, *_ in prepared]

    def merge(self, source_qq_id: Any, target_qq_id: Any) -> dict:
        source = self.resolve(source_qq_id)
        target = self.resolve(target_qq_id)
        if source == target:
            raise ValueError("请选择不同的来源与目标人物")
        source_card, target_card = self.get(source), self.get(target)
        if source_card is None or target_card is None:
            raise ValueError("来源或目标人物不存在")
        # 显式指定目标 QQ 为主身份；历史 raw、participant QQ 不做破坏性改写。
        with self.db.transaction():
            self.db.execute(
                "UPDATE identity_redirects SET target_qq_id=? WHERE target_qq_id=?",
                (target, source),
            )
            self.db.execute(
                "INSERT INTO identity_redirects(source_qq_id, target_qq_id) VALUES (?, ?)",
                (source, target),
            )
            for alias in {source_card["canonical_name"], *source_card["aliases"]}:
                self.db.execute(
                    "INSERT OR IGNORE INTO identity_aliases(qq_id, alias) VALUES (?, ?)",
                    (target, alias),
                )
        return self.get(target)
