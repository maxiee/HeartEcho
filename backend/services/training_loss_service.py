from domain.corpus import CorpusEntry
from domain.training_session import TrainingSession


class TrainingLossService:
    def __init__(self):
        self.loss_map = {}

    def update_loss(self, corpus_entry_id: str, loss: float, entry: CorpusEntry):
        self.loss_map[corpus_entry_id] = (loss, entry)

    def save(self, session: TrainingSession):
        # This is an empty method for now
        # We'll implement the actual saving logic later
        pass
