from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class CorpusBase(BaseModel):
    name: str
    description: Optional[str] = None


class CorpusCreate(CorpusBase):
    pass


class CorpusResponse(CorpusBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class CorpusListResponse(BaseModel):
    items: List[CorpusResponse]
    total: int
    skip: int
    limit: int


class CorpusEntryBase(BaseModel):
    content: str
    entry_type: str


class CorpusEntryCreate(CorpusEntryBase):
    pass


class CorpusEntryResponse(CorpusEntryBase):
    id: str
    corpus_id: str
    created_at: datetime
    metadata: dict = {}

    class Config:
        orm_mode = True
