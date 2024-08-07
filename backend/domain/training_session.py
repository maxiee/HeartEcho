from dataclasses import dataclass, field
from datetime import datetime
from .model import Model


@dataclass
class TrainingSession:
    """表示一次训练会话，关联了模型和使用的语料库。"""

    id: str
    name: str
    model: Model
    start_time: datetime
    last_trained: datetime
    metrics: dict = field(default_factory=dict)

    def update_metrics(self, new_metrics: dict):
        self.metrics.update(new_metrics)
        self.last_trained = datetime.now()
