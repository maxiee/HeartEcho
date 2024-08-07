from typing import List, Optional
from mongoengine import (
    connect,
    Document,
    StringField,
    DateTimeField,
)
from domain.corpus import Corpus
from .corpus_repository import CorpusRepository


class MongoCorpus(Document):
    id = StringField(primary_key=True)
    name = StringField(required=True, unique=True)
    description = StringField()
    created_at = DateTimeField(required=True)
    updated_at = DateTimeField(required=True)

    meta = {"collection": "corpora"}


class MongoDBCorpusRepository(CorpusRepository):
    def __init__(self, connection_string: str):
        connect(host=connection_string)

    def get_by_id(self, corpus_id: str) -> Optional[Corpus]:
        mongo_corpus = MongoCorpus.objects(id=corpus_id).first()
        return self._to_domain(mongo_corpus) if mongo_corpus else None

    def save(self, corpus: Corpus) -> Corpus:
        mongo_corpus = self._to_mongo(corpus)
        mongo_corpus.save()
        return self._to_domain(mongo_corpus)

    def list(self, skip: int = 0, limit: int = 100) -> List[Corpus]:
        mongo_corpora = MongoCorpus.objects().skip(skip).limit(limit)
        return [self._to_domain(mc) for mc in mongo_corpora]

    def _to_domain(self, mongo_corpus: MongoCorpus) -> Corpus:
        return Corpus(
            id=str(mongo_corpus.id),
            name=mongo_corpus.name,
            description=mongo_corpus.description,
            created_at=mongo_corpus.created_at,
            updated_at=mongo_corpus.updated_at,
        )

    def _to_mongo(self, corpus: Corpus) -> MongoCorpus:
        return MongoCorpus(
            _id=corpus.id,
            name=corpus.name,
            description=corpus.description,
            created_at=corpus.created_at,
            updated_at=corpus.updated_at,
        )
