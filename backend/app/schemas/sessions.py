from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from domain.training_session import TrainingSession


class TrainingSessionCreate(BaseModel):
    name: str
    base_model: str


class TrainingSessionResponse(BaseModel):
    id: str
    name: str
    base_model: str
    start_time: datetime
    last_trained: datetime
    last_trained: Optional[datetime] = None
    tokens_trained: int = 0
    metrics: dict

    @classmethod
    def from_domain(cls, session: TrainingSession):
        return cls(
            id=session.id,
            name=session.name,
            base_model=session.base_model,
            start_time=session.start_time,
            last_trained=session.last_trained,
            tokens_trained=session.tokens_trained,
            metrics=session.metrics,
        )
