from datetime import datetime
from typing import List, Optional
from mongoengine import (
    Document,
    StringField,
    DateTimeField,
    DictField,
    ReferenceField,
    ListField,
)
from app.core.db import DB
from domain.corpus import CorpusEntry
from repositories.corpus.mongodb_corpus_repository import MongoCorpus
from .corpus_entry_repository import CorpusEntryRepository


class MongoCorpusEntry(Document):
    id = StringField(primary_key=True)
    entry_type = StringField(choices=["chat", "knowledge"], required=True)
    created_at = DateTimeField(default=datetime.utcnow())
    corpus = StringField(required=True)
    metadata = DictField()
    content = StringField()  # For 'knowledge' type
    messages = ListField(DictField(), default=list)  # For 'chat' type

    meta = {"collection": "corpus_entries"}


class MongoDBCorpusEntryRepository(CorpusEntryRepository):
    def __init__(self) -> None:
        super().__init__()
        DB.init()

    def get_by_id(self, entry_id: str) -> Optional[CorpusEntry]:
        mongo_entry = MongoCorpusEntry.objects(id=entry_id).first()
        return self._to_domain(mongo_entry) if mongo_entry else None

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

    def delete(self, entry_id: str) -> bool:
        result = MongoCorpusEntry.objects(id=entry_id).delete()
        return result > 0

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
        )
