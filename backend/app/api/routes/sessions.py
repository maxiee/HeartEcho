from fastapi import APIRouter, Depends, HTTPException
from app.core.dependencies import get_training_session_service
from app.schemas.sessions import TrainingSessionCreate, TrainingSessionResponse
from services.training_session_service import TrainingSessionService

router = APIRouter()


@router.post("/", response_model=TrainingSessionResponse)
async def create_training_session(
    session: TrainingSessionCreate,
    service: TrainingSessionService = Depends(get_training_session_service),
):
    try:
        created_session = service.create_session(
            name=session.name, base_model=session.base_model
        )
        return TrainingSessionResponse.from_domain(created_session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/current", response_model=TrainingSessionResponse)
async def get_current_session(
    service: TrainingSessionService = Depends(get_training_session_service),
):
    session = service.get_current_session()
    if not session:
        raise HTTPException(status_code=404, detail="No active training session")
    return TrainingSessionResponse.from_domain(session)


@router.post("/end", response_model=None)
async def end_current_session(
    service: TrainingSessionService = Depends(get_training_session_service),
):
    service.end_current_session()
    return {"message": "Current session ended successfully"}


@router.put("/metrics", response_model=TrainingSessionResponse)
async def update_session_metrics(
    metrics: dict,
    service: TrainingSessionService = Depends(get_training_session_service),
):
    try:
        session = service.update_metrics(metrics)
        return TrainingSessionResponse.from_domain(session)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/list", response_model=list[TrainingSessionResponse])
async def list_sessions(
    skip: int = 0,
    limit: int = 100,
    service: TrainingSessionService = Depends(get_training_session_service),
):
    sessions = service.list_sessions(skip, limit)
    return [TrainingSessionResponse.from_domain(session) for session in sessions]
