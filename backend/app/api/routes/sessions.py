from fastapi import APIRouter, Depends, HTTPException
from app.core.dependencies import (
    get_training_loss_service,
    get_training_session_service,
)
from app.schemas.sessions import TrainingSessionCreate, TrainingSessionResponse
from services.training_loss_service import TrainingLossService
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


@router.post("/load/{session_id}", response_model=TrainingSessionResponse)
async def load_training_session(
    session_id: str,
    service: TrainingSessionService = Depends(get_training_session_service),
):
    try:
        loaded_session = service.load_session(session_id)
        return TrainingSessionResponse.from_domain(loaded_session)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/save", response_model=TrainingSessionResponse)
async def save_current_session(
    training_session_service: TrainingSessionService = Depends(
        get_training_session_service
    ),
    training_loss_service: TrainingLossService = Depends(get_training_loss_service),
):
    try:
        # Save the current session and model
        saved_session = training_session_service.save_current_session()

        # Call the empty save method in TrainingLossService
        training_loss_service.save(saved_session)

        return TrainingSessionResponse.from_domain(saved_session)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


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
    service: TrainingSessionService = Depends(get_training_session_service),
):
    sessions = service.list_sessions()
    return [TrainingSessionResponse.from_domain(session) for session in sessions]
