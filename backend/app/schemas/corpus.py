from pydantic import BaseModel, Field
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


class CorpusEntryCreate(BaseModel):
    entry_type: str = Field(..., description="Type of the entry: 'chat' or 'knowledge'")
    content: Optional[str] = Field(
        None, description="Content for knowledge type entries"
    )
    messages: Optional[List[Dict[str, str]]] = Field(
        None, description="Messages for chat type entries"
    )

    class Config:
        schema_extra = {
            "example": {
                "entry_type": "knowledge",
                "content": "This is a knowledge entry",
            }
        }


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


class LossDistributionItem(BaseModel):
    lower: str
    count: int


class LossDistributionResponse(BaseModel):
    distribution: List[LossDistributionItem]
