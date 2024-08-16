from dataclasses import dataclass
from datetime import datetime
from typing import Optional

N = 7  # 定义常量N，可以根据需要调整


@dataclass
class TrainingLoss:
    id: str
    corpus_entry_id: str
    session_id: str
    timestamp: datetime
    loss_value: float
    loss_rank: str
    is_reverse_gradient: bool = False

    def __post_init__(self):
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.fromisoformat(self.timestamp)

    @staticmethod
    def calculate_loss_value(actual_loss: float, is_reverse: bool) -> float:
        if is_reverse:
            return max(N - actual_loss, 0.1)  # 设置最小存储Loss为0.5
        return actual_loss

    @staticmethod
    def calculate_loss_rank(loss: float) -> float:
        if loss < 0:
            return 0.0
        rank = int(loss / 0.5) * 0.5
        return f"{min(rank, 10.0):.1f}"
