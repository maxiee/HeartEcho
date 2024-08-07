from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from .model import Model
from .corpus import Corpus


@dataclass
class TrainingSession:
    """表示一次训练会话，关联了模型和使用的语料库。"""

    id: str
    model: Model
    corpus: Corpus
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "in_progress"  # 'in_progress', 'completed', 'failed'
    metrics: dict = field(default_factory=dict)

    def complete(self):
        self.status = "completed"
        self.end_time = datetime.now()

    def fail(self, reason: str):
        self.status = "failed"
        self.end_time = datetime.now()
        self.metrics["failure_reason"] = reason

    def update_metrics(self, new_metrics: dict):
        self.metrics.update(new_metrics)
