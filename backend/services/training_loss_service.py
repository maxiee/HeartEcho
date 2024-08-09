from datetime import datetime
from typing import List
from domain.corpus import CorpusEntry
from domain.training_loss import TrainingLoss
from domain.training_session import TrainingSession
from repositories.training_loss.training_loss_repository import TrainingLossRepository
from utils.id_generator import IdGenerator


class TrainingLossService:
    def __init__(self, training_loss_repo: TrainingLossRepository):
        self.training_loss_repo = training_loss_repo

    def update_loss(
        self,
        corpus_entry_id: str,
        loss: float,
        session: TrainingSession,
    ):
        training_loss = TrainingLoss(
            id=IdGenerator.generate(),
            corpus_entry_id=corpus_entry_id,
            session_id=session.id,
            timestamp=datetime.now(),
            loss_value=loss,
            loss_rank=TrainingLoss.calculate_loss_rank(loss),
        )
        self.training_loss_repo.save(training_loss)

    def get_losses_for_session(self, session_id: str) -> List[TrainingLoss]:
        return self.training_loss_repo.get_by_session_id(session_id)

    def get_losses_for_corpus_entry(self, corpus_entry_id: str) -> List[TrainingLoss]:
        return self.training_loss_repo.get_by_corpus_entry_id(corpus_entry_id)
