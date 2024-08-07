from typing import Optional
from datetime import datetime
from domain.training_session import TrainingSession
from repositories.training_session.training_session_repository import (
    TrainingSessionRepository,
)
from utils.id_generator import IdGenerator


class TrainingSessionService:
    def __init__(self, session_repo: TrainingSessionRepository):
        self.session_repo = session_repo
        self.current_session: Optional[TrainingSession] = None

    def create_session(self, name: str, base_model: str) -> TrainingSession:
        session = TrainingSession(
            id=IdGenerator.generate(),
            name=name,
            base_model=base_model,
            start_time=datetime.now(),
            last_trained=datetime.now(),
        )
        created_session = self.session_repo.create(session)
        self.current_session = created_session
        return created_session

    def load_session(self, session_id: str) -> TrainingSession:
        session = self.session_repo.get_by_id(session_id)
        if not session:
            raise ValueError(f"Session with id {session_id} not found")
        self.current_session = session
        return session

    def get_current_session(self) -> Optional[TrainingSession]:
        return self.current_session

    def end_current_session(self) -> None:
        if self.current_session:
            self.current_session.end_session()
            self.session_repo.update(self.current_session)
            self.current_session = None

    def update_metrics(self, metrics: dict) -> TrainingSession:
        if not self.current_session:
            raise ValueError("No active training session")
        self.current_session.update_metrics(metrics)
        return self.session_repo.update(self.current_session)

    def add_checkpoint(self, checkpoint_path: str) -> TrainingSession:
        if not self.current_session:
            raise ValueError("No active training session")
        self.current_session.add_checkpoint(checkpoint_path)
        return self.session_repo.update(self.current_session)

    def list_sessions(self, skip: int = 0, limit: int = 100) -> list[TrainingSession]:
        return self.session_repo.list(skip, limit)
