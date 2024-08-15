from datetime import datetime
import random
from typing import List, Optional
from mongoengine import (
    Document,
    StringField,
    DateTimeField,
    DictField,
    ListField,
    BooleanField,
)
from app.core.db import DB
from domain.corpus import CorpusEntry
from repositories.corpus.mongodb_corpus_repository import MongoCorpus
from repositories.training_loss.mongodb_training_loss_repository import (
    MongoTrainingLoss,
)
from .corpus_entry_repository import CorpusEntryRepository


class MongoCorpusEntry(Document):
    id = StringField(primary_key=True)
    entry_type = StringField(choices=["chat", "knowledge"], required=True)
    created_at = DateTimeField(default=datetime.utcnow())
    corpus = StringField(required=True)
    metadata = DictField()
    content = StringField()  # For 'knowledge' type
    messages = ListField(DictField(), default=list)  # For 'chat' type
    sha256 = StringField(unique=True)
    is_reverse_gradient = BooleanField(default=False)

    meta = {"collection": "corpus_entries"}


class MongoDBCorpusEntryRepository(CorpusEntryRepository):
    def __init__(self) -> None:
        super().__init__()
        DB.init()

    def get_by_id(self, entry_id: str) -> Optional[CorpusEntry]:
        mongo_entry = MongoCorpusEntry.objects(id=entry_id).first()
        return self._to_domain(mongo_entry) if mongo_entry else None

    def get_entries_by_ids(self, entry_ids: List[str]) -> List[CorpusEntry]:
        mongo_entries = MongoCorpusEntry.objects(id__in=entry_ids)
        return [self._to_domain(me) for me in mongo_entries]

    def sample_random_entries(self, batch_size: int) -> List[CorpusEntry]:
        mongo_entries = MongoCorpusEntry.objects.aggregate(
            [{"$sample": {"size": batch_size}}]
        )
        # 本地调整：将 _id 改为 id
        adjusted_entries = []
        for entry in mongo_entries:
            entry["id"] = str(entry.pop("_id"))  # 将 _id 转换为字符串并重命名为 id
            adjusted_entries.append(entry)

        return [self._to_domain(MongoCorpusEntry(**me)) for me in adjusted_entries]

    def save(self, entry: CorpusEntry) -> CorpusEntry:
        mongo_entry = self._to_mongo(entry)
        mongo_entry.save()
        try:
            mongo_entry.save()
        except Exception as e:
            if "duplicate key error" in str(e):
                raise ValueError(
                    f"A corpus entry with SHA256 {entry.sha256} already exists."
                )
            raise
        return self._to_domain(mongo_entry)

    def list_by_corpus(
        self, corpus: str, skip: int = 0, limit: int = 100
    ) -> List[CorpusEntry]:
        print("list_by_corpus")
        mongo_entries = MongoCorpusEntry.objects(corpus=corpus).skip(skip).limit(limit)
        return [self._to_domain(me) for me in mongo_entries]

    def sample_new_entries(
        self, batch_size: int, total_entries: int, session_id: str
    ) -> List[CorpusEntry]:
        new_entries = []
        page_size = 100
        skip = 0

        while len(new_entries) < batch_size:
            # 逆序查询语料条目
            entries = (
                MongoCorpusEntry.objects().order_by("-_id").skip(skip).limit(page_size)
            )

            for entry in entries:
                # 检查该条目是否已经被训练过
                if not MongoTrainingLoss.objects(
                    corpus_entry_id=entry.id, session_id=session_id
                ).first():
                    new_entries.append(self._to_domain(entry))
                    if len(new_entries) == batch_size:
                        break

            if len(new_entries) == batch_size:
                break

            skip += page_size
            if skip >= total_entries:
                break  # 已经遍历完所有条目

        # 随机打乱新语料的顺序
        random.shuffle(new_entries)
        assert len(new_entries) <= batch_size
        return new_entries

    def delete(self, entry_id: str) -> bool:
        result = MongoCorpusEntry.objects(id=entry_id).delete()
        return result > 0

    def count(self) -> int:
        return MongoCorpusEntry.objects.count()

    def _to_domain(self, mongo_entry: MongoCorpusEntry) -> CorpusEntry:
        return CorpusEntry(
            id=mongo_entry.id,
            corpus=mongo_entry.corpus,
            content=(
                mongo_entry.content if mongo_entry.entry_type == "knowledge" else None
            ),
            messages=mongo_entry.messages if mongo_entry.entry_type == "chat" else None,
            entry_type=mongo_entry.entry_type,
            created_at=mongo_entry.created_at,
            metadata=mongo_entry.metadata,
            is_reverse_gradient=mongo_entry.is_reverse_gradient,
        )

    def _to_mongo(self, entry: CorpusEntry) -> MongoCorpusEntry:
        return MongoCorpusEntry(
            id=entry.id,
            corpus=entry.corpus,
            content=entry.content if entry.entry_type == "knowledge" else None,
            messages=entry.messages if entry.entry_type == "chat" else None,
            entry_type=entry.entry_type,
            created_at=entry.created_at,
            metadata=entry.metadata,
            sha256=entry.sha256,
            is_reverse_gradient=entry.is_reverse_gradient,
        )
