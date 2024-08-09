from typing import List
from domain.training_loss import TrainingLoss
from domain.training_session import TrainingSession
from repositories.training_loss.training_loss_repository import TrainingLossRepository
from utils.id_generator import IdGenerator


class TrainingLossService:
    def __init__(self, training_loss_repo: TrainingLossRepository):
        self.training_loss_repo = training_loss_repo
        self.loss_map = {}

    def update_loss(self, corpus_entry_id: str, loss: float, entry):
        self.loss_map[corpus_entry_id] = (loss, entry)

    def save(self, session: TrainingSession):
        for corpus_entry_id, (loss, entry) in self.loss_map.items():
            training_loss = TrainingLoss(
                id=IdGenerator.generate(),
                corpus_entry_id=corpus_entry_id,
                session_id=session.id,
                timestamp=session.last_trained,
                loss_value=loss,
                loss_rank=TrainingLoss.calculate_loss_rank(loss),
            )
            self.training_loss_repo.save(training_loss)

        # Clear the loss map after saving
        self.loss_map.clear()

    def get_losses_for_session(self, session_id: str) -> List[TrainingLoss]:
        return self.training_loss_repo.get_by_session_id(session_id)

    def get_losses_for_corpus_entry(self, corpus_entry_id: str) -> List[TrainingLoss]:
        return self.training_loss_repo.get_by_corpus_entry_id(corpus_entry_id)
