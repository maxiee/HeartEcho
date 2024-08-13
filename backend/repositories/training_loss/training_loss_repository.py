from abc import ABC, abstractmethod
from typing import List, Optional
from domain.training_loss import TrainingLoss


class TrainingLossRepository(ABC):
    @abstractmethod
    def save(self, training_loss: TrainingLoss) -> TrainingLoss:
        pass

    @abstractmethod
    def get_by_id(self, training_loss_id: str) -> Optional[TrainingLoss]:
        pass

    @abstractmethod
    def get_by_session_id(self, session_id: str) -> List[TrainingLoss]:
        pass

    @abstractmethod
    def get_by_corpus_entry_id_and_session_id(
        self, corpus_entry_id: str, session_id: str
    ) -> List[TrainingLoss]:
        pass

    @abstractmethod
    def count_by_loss_rank(self, session_id: str, loss_rank: str) -> int:
        pass

    @abstractmethod
    def count_by_session_id(self, session_id: str) -> int:
        pass

    @abstractmethod
    def get_highest_loss_entries(
        self, session_id: str, limit: int
    ) -> List[TrainingLoss]:
        pass

    @abstractmethod
    def get_lowest_loss_entries(self, session_id: str, limit: int) -> TrainingLoss:
        pass
