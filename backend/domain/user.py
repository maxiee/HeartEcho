from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class User:
    """表示系统用户，可以拥有多个模型。"""

    id: str
    username: str
    email: str
    created_at: datetime
    last_login: datetime
    models: List[str] = field(default_factory=list)

    def add_model(self, model_id: str):
        if model_id not in self.models:
            self.models.append(model_id)

    def remove_model(self, model_id: str):
        if model_id in self.models:
            self.models.remove(model_id)
