from domain.corpus import CorpusEntry


class TrainingLossService:
    def __init__(self):
        self.loss_map = {}

    def update_loss(self, corpus_entry_id: str, loss: float, entry: CorpusEntry):
        self.loss_map[corpus_entry_id] = (loss, entry)
