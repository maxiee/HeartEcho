from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TrainingLoss:
    id: str
    corpus_entry_id: str
    session_id: str
    timestamp: datetime
    loss_value: float
    loss_rank: int

    def __post_init__(self):
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.fromisoformat(self.timestamp)

    @staticmethod
    def calculate_loss_rank(loss: float) -> float:
        if loss < 0:
            return 0.0
        rank = int(loss / 0.5) * 0.5
        return min(rank, 10.0)
