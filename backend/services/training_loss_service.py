from datetime import datetime
from typing import List
from app.schemas.corpus import LossDistributionItem
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

    def get_loss_distribution(self, session_id: str) -> List[LossDistributionItem]:
        ranges = [
            "0.0",
            "0.5",
            "1.0",
            "1.5",
            "2.0",
            "2.5",
            "3.0",
            "3.5",
            "4.0",
            "4.5",
            "5.0",
            "5.5",
            "6.0",
            "6.5",
            "7.0",
            "7.5",
            "8.0",
            "8.5",
            "9.0",
            "9.5",
        ]  # 生成 0.0 到 10.0 的范围,步长为 0.5
        distribution = []

        for range in ranges:
            count = self.training_loss_repo.count_by_loss_rank(session_id, range)
            distribution.append(LossDistributionItem(lower=range, count=count))

        return distribution
