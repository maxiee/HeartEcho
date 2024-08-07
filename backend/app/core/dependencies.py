from functools import lru_cache

from app.core.config import settings
from repositories.corpus.mongodb_corpus_repository import MongoDBCorpusRepository
from repositories.corpus_entry.mongodb_corpus_entry_repository import (
    MongoDBCorpusEntryRepository,
)
from repositories.training_session.filesystem_training_session_repository import (
    FileSystemTrainingSessionRepository,
)
from services.corpus_management_service import CorpusManagementService
from services.model_training_service import ModelTrainingService
from services.training_session_service import TrainingSessionService


@lru_cache()
def get_corpus_service() -> CorpusManagementService:
    corpus_repo = MongoDBCorpusRepository()
    corpus_entry_repo = MongoDBCorpusEntryRepository()
    return CorpusManagementService(corpus_repo, corpus_entry_repo)


@lru_cache()
def get_training_session_service():
    training_session_repo = FileSystemTrainingSessionRepository()
    return TrainingSessionService(training_session_repo)


@lru_cache()
def get_model_training_service():
    session_repo = FileSystemTrainingSessionRepository()
    return ModelTrainingService(session_repo)
