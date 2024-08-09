from functools import lru_cache

from app.core.config import settings
from domain import training_loss
from llm_manager import LLMManager
from repositories.corpus.mongodb_corpus_repository import MongoDBCorpusRepository
from repositories.corpus_entry.mongodb_corpus_entry_repository import (
    MongoDBCorpusEntryRepository,
)
from repositories.training_loss.mongodb_training_loss_repository import (
    MongoDBTrainingLossRepository,
)
from repositories.training_loss.training_loss_repository import TrainingLossRepository
from repositories.training_session.filesystem_training_session_repository import (
    FileSystemTrainingSessionRepository,
)
from services.corpus_management_service import CorpusManagementService
from services.model_training_service import ModelTrainingService
from services.training_loss_service import TrainingLossService
from services.training_session_service import TrainingSessionService


@lru_cache()
def get_corpus_service() -> CorpusManagementService:
    corpus_repo = MongoDBCorpusRepository()
    corpus_entry_repo = MongoDBCorpusEntryRepository()
    return CorpusManagementService(corpus_repo, corpus_entry_repo)


@lru_cache()
def get_training_session_service():
    training_session_repo = FileSystemTrainingSessionRepository()
    return TrainingSessionService(training_session_repo, get_llm_manager())


@lru_cache()
def get_model_training_service() -> ModelTrainingService:
    return ModelTrainingService(
        llm_manager=get_llm_manager(),
        corpus_entry_repo=MongoDBCorpusEntryRepository(),
        training_session_service=get_training_session_service(),
        training_loss_service=get_training_loss_service(),
    )


@lru_cache()
def get_training_loss_service():
    training_loss_repo = MongoDBTrainingLossRepository()
    return TrainingLossService(training_loss_repo=training_loss_repo)


@lru_cache()
def get_llm_manager():
    return LLMManager()
