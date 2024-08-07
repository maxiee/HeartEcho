from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.core.dependencies import get_corpus_service
from app.schemas.corpus import (
    CorpusCreate,
    CorpusResponse,
    CorpusListResponse,
    CorpusEntryCreate,
    CorpusEntryResponse,
)

router = APIRouter()


@router.post("/", response_model=CorpusResponse)
async def create_corpus(corpus: CorpusCreate, service=Depends(get_corpus_service)):
    created_corpus = service.create_corpus(
        name=corpus.name, description=corpus.description
    )
    return CorpusResponse(**created_corpus.__dict__)


@router.get("/", response_model=CorpusListResponse)
async def get_corpora(
    skip: int = 0, limit: int = 100, service=Depends(get_corpus_service)
):
    corpora = service.list_corpora(skip=skip, limit=limit)
    total = service.count_corpora()
    return CorpusListResponse(
        items=[CorpusResponse(**corpus.__dict__) for corpus in corpora],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.post("/{corpus_id}/entry", response_model=CorpusEntryResponse)
async def add_corpus_entry(
    corpus_id: str, entry: CorpusEntryCreate, service=Depends(get_corpus_service)
):
    try:
        created_entry = service.add_entry_to_corpus(
            corpus_id=corpus_id, content=entry.content, entry_type=entry.entry_type
        )
        return CorpusEntryResponse(**created_entry.__dict__)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{corpus_id}/entries", response_model=List[CorpusEntryResponse])
async def get_corpus_entries(
    corpus_id: str, skip: int = 0, limit: int = 100, service=Depends(get_corpus_service)
):
    entries = service.get_corpus_entries(corpus_id=corpus_id, skip=skip, limit=limit)
    return [CorpusEntryResponse(**entry.__dict__) for entry in entries]
