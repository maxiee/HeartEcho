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
    def get_by_corpus_entry_id(self, corpus_entry_id: str) -> List[TrainingLoss]:
        pass
