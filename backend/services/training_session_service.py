from typing import List, Optional, Set
from datetime import datetime
from domain.corpus import CorpusEntry
from domain.training_session import TrainingSession
from llm_manager import LLMManager
from repositories.training_session.training_session_repository import (
    TrainingSessionRepository,
)
from utils.id_generator import IdGenerator


class TrainingSessionService:
    def __init__(
        self, session_repo: TrainingSessionRepository, llm_manager: LLMManager
    ):
        self.session_repo = session_repo
        self.current_session: Optional[TrainingSession] = None
        self.llm_manager = llm_manager

    def create_session(self, name: str, base_model: str) -> TrainingSession:
        assert self.current_session is None, "There is already an active session"
        session = TrainingSession(
            id=IdGenerator.generate(),
            name=name,
            base_model=base_model,
            start_time=datetime.now(),
            last_trained=datetime.now(),
        )
        created_session = self.session_repo.create(session)
        self.current_session = created_session
        self.llm_manager.init_new_model(created_session.base_model)
        self.llm_manager.save_model(created_session)
        return created_session

    def load_session(self, session_id: str) -> TrainingSession:
        assert self.current_session is None, "There is already an active session"
        session = self.session_repo.get_by_id(session_id)
        if not session:
            raise ValueError(f"Session with id {session_id} not found")
        self.current_session = session
        return session

    def save_current_session(self):
        assert self.current_session is not None, "There is no active session"
        self.session_repo.update(self.current_session)
        self.llm_manager.save_model(self.current_session)

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

    def list_sessions(self) -> list[TrainingSession]:
        return self.session_repo.list_sessions()

    def update_tokens_trained(self, new_tokens: int):
        if not self.current_session:
            raise ValueError("No active training session")
        self.current_session.tokens_trained += new_tokens
        # self.session_repo.update(self.current_session)
