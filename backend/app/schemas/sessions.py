from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class TrainingSessionCreate(BaseModel):
    name: str
    model_id: str


class TrainingSessionResponse(BaseModel):
    id: str
    name: str
    model_id: str
    start_time: datetime
    last_trained: datetime
    status: str
    metrics: dict
