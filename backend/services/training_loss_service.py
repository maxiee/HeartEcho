from datetime import datetime
from typing import List
from app.schemas.corpus import LossDistributionItem
from domain.corpus import CorpusEntry
from domain.training_loss import TrainingLoss
from domain.training_session import TrainingSession
from repositories.corpus_entry.corpus_entry_repository import CorpusEntryRepository
from repositories.training_loss.training_loss_repository import TrainingLossRepository
from utils.id_generator import IdGenerator


class TrainingLossService:
    def __init__(
        self,
        training_loss_repo: TrainingLossRepository,
        corpus_entry_repo: CorpusEntryRepository,
    ):
        self.training_loss_repo = training_loss_repo
        self.corpus_entry_repo = corpus_entry_repo

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

    def count_trained_entries_for_session(self, session_id: str) -> int:
        return self.training_loss_repo.count_by_session_id(session_id)

    def get_new_corpus_entries_count(
        self, session_id: str, total_corpus_entries: int
    ) -> int:
        trained_entries_count = self.training_loss_repo.count_by_session_id(session_id)
        return max(0, total_corpus_entries - trained_entries_count)

    def get_highest_loss_entries(
        self, session_id: str, batch_size: int
    ) -> List[CorpusEntry]:
        # Get the highest loss TrainingLoss objects
        highest_loss_entries = self.training_loss_repo.get_highest_loss_entries(
            session_id, batch_size
        )

        # Get the corresponding CorpusEntry objects
        corpus_entry_ids = [entry.corpus_entry_id for entry in highest_loss_entries]
        corpus_entries = self.corpus_entry_repo.get_entries_by_ids(corpus_entry_ids)

        # Sort corpus_entries to match the order of highest_loss_entries
        sorted_corpus_entries = sorted(
            corpus_entries, key=lambda x: corpus_entry_ids.index(x.id)
        )

        return sorted_corpus_entries

    def get_lowest_loss_entries(
        self, session_id: str, batch_size: int
    ) -> List[CorpusEntry]:
        # Get the lowest loss TrainingLoss objects
        lowest_loss_entries = self.training_loss_repo.get_lowest_loss_entries(
            session_id, batch_size
        )

        # Get the corresponding CorpusEntry objects
        corpus_entry_ids = [entry.corpus_entry_id for entry in lowest_loss_entries]
        corpus_entries = self.corpus_entry_repo.get_entries_by_ids(corpus_entry_ids)

        # Sort corpus_entries to match the order of lowest_loss_entries
        sorted_corpus_entries = sorted(
            corpus_entries, key=lambda x: corpus_entry_ids.index(x.id)
        )

        return sorted_corpus_entries
