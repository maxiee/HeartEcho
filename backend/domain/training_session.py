from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TrainingSession:
    """表示一次训练会话，关联了模型和使用的语料库。"""

    id: str
    name: str
    base_model: str
    start_time: datetime
    last_trained: datetime
    metrics: dict = field(default_factory=dict)

    def update_metrics(self, new_metrics: dict):
        self.metrics.update(new_metrics)
        self.last_trained = datetime.now()
