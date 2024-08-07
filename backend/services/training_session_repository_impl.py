from datetime import datetime
from domain.training_session import TrainingSession
from repositories.corpus.corpus_repository import CorpusRepository
from repositories.model_repository import ModelRepository
from repositories.training_session.training_session_repository import (
    TrainingSessionRepository,
)
from utils.id_generator import IdGenerator


class TrainingSessionService:
    def __init__(
        self,
        session_repo: TrainingSessionRepository,
        model_repo: ModelRepository,
        corpus_repo: CorpusRepository,
    ):
        self.session_repo = session_repo
        self.model_repo = model_repo
        self.corpus_repo = corpus_repo

    def create_session(
        self,
        name: str,
        model_id: str,
    ) -> TrainingSession:
        model = self.model_repo.get_by_id(model_id)
        if not model:
            raise ValueError("Invalid model or corpus ID")

        session = TrainingSession(
            id=IdGenerator.generate(),
            name=name,
            model=model,
            start_time=datetime.now(),
            last_trained=datetime.now(),
        )
        return self.session_repo.create(session)

    def load_session(self, session_id: str) -> TrainingSession:
        session = self.session_repo.get_by_id(session_id)
        if not session:
            raise ValueError(f"Session with id {session_id} not found")
        return session

    def add_checkpoint(self, session_id: str, checkpoint_path: str) -> TrainingSession:
        session = self.load_session(session_id)
        session.add_checkpoint(checkpoint_path)
        return self.session_repo.update(session)

    def update_metrics(self, session_id: str, metrics: dict) -> TrainingSession:
        session = self.load_session(session_id)
        session.update_metrics(metrics)
        return self.session_repo.update(session)
