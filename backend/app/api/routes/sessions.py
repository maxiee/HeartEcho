from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_training_session_service
from app.schemas.sessions import TrainingSessionCreate, TrainingSessionResponse
from services.training_session_repository_impl import TrainingSessionService


router = APIRouter()


@router.post("/", response_model=TrainingSessionResponse)
async def create_training_session(
    session: TrainingSessionCreate,
    service: TrainingSessionService = Depends(get_training_session_service),
):
    try:
        created_session = service.create_session(
            name=session.name, model_id=session.model_id, corpus_id=session.corpus_id
        )
        return TrainingSessionResponse(**created_session.__dict__)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{session_id}", response_model=TrainingSessionResponse)
async def get_training_session(
    session_id: str,
    service: TrainingSessionService = Depends(get_training_session_service),
):
    try:
        session = service.load_session(session_id)
        return TrainingSessionResponse(**session.__dict__)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/sessions/{session_id}/metrics", response_model=TrainingSessionResponse)
async def update_session_metrics(
    session_id: str,
    metrics: dict,
    service: TrainingSessionService = Depends(get_training_session_service),
):
    try:
        session = service.update_metrics(session_id, metrics)
        return TrainingSessionResponse(**session.__dict__)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
