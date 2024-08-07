from functools import lru_cache

from app.core.config import settings
from repositories.corpus.mongodb_corpus_repository import MongoDBCorpusRepository
from repositories.corpus_entry.mongodb_corpus_entry_repository import (
    MongoDBCorpusEntryRepository,
)
from repositories.model_repository import ModelRepository
from repositories.training_session.filesystem_training_session_repository import (
    FileSystemTrainingSessionRepository,
)
from services.corpus_management_service import CorpusManagementService
from services.model_training_service import ModelTrainingService
from services.training_session_repository_impl import TrainingSessionService


@lru_cache()
def get_corpus_service():
    corpus_repo = MongoDBCorpusRepository(connection_string=settings.MONGODB_URL)
    corpus_entry_repo = MongoDBCorpusEntryRepository(
        connection_string=settings.MONGODB_URL
    )
    return CorpusManagementService(corpus_repo, corpus_entry_repo)


@lru_cache()
def get_training_session_service():
    training_session_repo = FileSystemTrainingSessionRepository()
    model_repo = ModelRepository()
    return TrainingSessionService(training_session_repo, model_repo)


@lru_cache()
def get_model_training_service():
    model_repo = ModelRepository()
    session_repo = FileSystemTrainingSessionRepository()
    return ModelTrainingService(model_repo, session_repo)
