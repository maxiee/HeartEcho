import os
import torch
from datetime import datetime
from typing import Dict
from domain.model import Model
from domain.training_session import TrainingSession
from repositories.model_repository import ModelRepository
from repositories.training_session.training_session_repository import (
    TrainingSessionRepository,
)
from utils.id_generator import IdGenerator


class ModelTrainingService:
    def __init__(
        self, model_repo: ModelRepository, session_repo: TrainingSessionRepository
    ):
        self.model_repo = model_repo
        self.session_repo = session_repo

    def create_training_session(self, name: str, model: Model) -> TrainingSession:
        session = TrainingSession(
            id=IdGenerator.generate(),
            name=name,
            model=model,
            start_time=datetime.now(),
            last_trained=datetime.now(),
            metrics={},
        )
        return self.session_repo.create(session)

    def get_training_session(self, session_id: str) -> TrainingSession:
        session = self.session_repo.get_by_id(session_id)
        if not session:
            raise ValueError(f"Training session with id {session_id} not found")
        return session

    def update_training_metrics(
        self, session_id: str, metrics: Dict
    ) -> TrainingSession:
        session = self.get_training_session(session_id)
        session.metrics.update(metrics)
        session.last_trained = datetime.now()
        return self.session_repo.update(session)

    def save_model(self, session: TrainingSession, model_state: Dict):
        session_path = os.path.join("./trained", session.name)
        model_path = os.path.join(session_path, "model.pth")
        torch.save(model_state, model_path)
        session.model.updated_at = datetime.now()
        self.session_repo.update(session)

    def load_model(self, session: TrainingSession):
        session_path = os.path.join("./trained", session.name)
        model_path = os.path.join(session_path, "model.pth")
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found for session {session.name}")
        return torch.load(model_path)

    def list_active_sessions(self):
        return self.session_repo.list_active_sessions()

    def delete_training_session(self, session_id: str) -> bool:
        return self.session_repo.delete(session_id)
