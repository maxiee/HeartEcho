from abc import ABC, abstractmethod
from typing import List, Optional
from domain.training_session import TrainingSession


class TrainingSessionRepository(ABC):
    @abstractmethod
    def create(self, session: TrainingSession) -> TrainingSession:
        pass

    @abstractmethod
    def get_by_id(self, session_id: str) -> Optional[TrainingSession]:
        pass

    @abstractmethod
    def update(self, session: TrainingSession) -> TrainingSession:
        pass

    @abstractmethod
    def list_active_sessions(self) -> List[TrainingSession]:
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        pass
