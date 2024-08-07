from functools import lru_cache

from app.core.config import settings
from repositories.corpus.mongodb_corpus_repository import MongoDBCorpusRepository
from repositories.corpus_entry.mongodb_corpus_entry_repository import (
    MongoDBCorpusEntryRepository,
)
from services.corpus_management_service import CorpusManagementService


@lru_cache()
def get_corpus_service():
    corpus_repo = MongoDBCorpusRepository(connection_string=settings.MONGODB_URL)
    corpus_entry_repo = MongoDBCorpusEntryRepository(
        connection_string=settings.MONGODB_URL
    )
    return CorpusManagementService(corpus_repo, corpus_entry_repo)
