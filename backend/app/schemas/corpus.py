from pydantic import BaseModel
from typing import Dict, Optional, List
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


class CorpusEntryResponse(BaseModel):
    id: str
    corpus: str
    entry_type: str
    created_at: datetime
    content: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    metadata: dict = {}
    sha256: str

    class Config:
        orm_mode = True
