# domain/model.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from .value_objects.error_range import ErrorRange


@dataclass
class Model:
    """表示训练的模型，包含模型的基本信息和错误分布。"""

    id: str
    name: str
    base_model: str
    created_at: datetime
    updated_at: datetime
    parameters: int
    current_error: Optional[float] = None
    error_distribution: List[ErrorRange] = field(default_factory=list)

    def update_error(self, new_error: float):
        self.current_error = new_error
        self.updated_at = datetime.now()

    def add_to_error_distribution(self, error_range: ErrorRange):
        self.error_distribution.append(error_range)
