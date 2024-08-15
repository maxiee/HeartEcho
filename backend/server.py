import os
import time
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from app.core.dependencies import (
    get_llm_manager,
    get_model_training_service,
    get_training_session_service,
)
from llm_manager import LLMManager

import logging
import app.api.routes.corpus as corpus_routes
import app.api.routes.sessions as sessions_routes
from app.core.config import settings
from services.model_training_service import ModelTrainingService
from services.training_session_service import TrainingSessionService

# 如果环境变量中设置了 MODEL_SAVE_PATH,则使用环境变量的值
if "MODEL_SAVE_PATH" in os.environ:
    settings.MODEL_SAVE_PATH = os.environ["MODEL_SAVE_PATH"]

logger = logging.getLogger(__name__)

app = FastAPI()

print(corpus_routes.router.routes)

app.include_router(corpus_routes.router, prefix="/corpus", tags=["corpus"])
app.include_router(sessions_routes.router, prefix="/sessions", tags=["sessions"])
# app.include_router(model.router, prefix="/model", tags=["model"])
# app.include_router(training.router, prefix="/training", tags=["training"])


class ChatInput(BaseModel):
    history: List[Dict[str, Any]]


@app.post("/chat")
def chat(
    chat_input: ChatInput,
    llm_manager: LLMManager = Depends(get_llm_manager),
    training_session_service: TrainingSessionService = Depends(
        get_training_session_service
    ),
):
    try:
        response = llm_manager.chat(
            chat_input.history, training_session_service.get_current_session().name
        )
        return {"response": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/smelt_new_corpus")
async def smelt_new_corpus(
    model_training_service: ModelTrainingService = Depends(get_model_training_service),
    training_session_service: TrainingSessionService = Depends(
        get_training_session_service
    ),
):
    session = training_session_service.get_current_session()
    if not session:
        raise HTTPException(status_code=404, detail="Training session not found")

    result = model_training_service.smelt_new_corpus()
    return result


@app.post("/smelt_new_old")
async def smelt_new_old(
    model_training_service: ModelTrainingService = Depends(get_model_training_service),
    training_session_service: TrainingSessionService = Depends(
        get_training_session_service
    ),
):
    session = training_session_service.get_current_session()
    if not session:
        raise HTTPException(status_code=404, detail="Training session not found")

    result = model_training_service.smelt_new_old()
    return result


@app.post("/train_single_entry/{entry_id}")
async def train_single_entry(
    entry_id: str,
    model_training_service: ModelTrainingService = Depends(get_model_training_service),
):
    try:
        result = model_training_service.train_single_entry(entry_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/treat_overfitting")
async def treat_overfitting(
    model_training_service: ModelTrainingService = Depends(get_model_training_service),
    training_session_service: TrainingSessionService = Depends(
        get_training_session_service
    ),
):
    session = training_session_service.get_current_session()
    if not session:
        raise HTTPException(status_code=404, detail="Training session not found")

    result = model_training_service.treat_overfitting()
    return result


@app.post("/create_training_session")
async def create_training_session():
    try:
        # Create a new directory for the training session
        session_name = f"training_session_{int(time.time())}"
        session_dir = os.path.join(settings.model_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)

        # Initialize the model for the new session
        llm_manager.init_new_model(session_dir)

        return {"message": "New training session created", "session_name": session_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/random_sample_training")
async def random_sample_training(
    model_training_service: ModelTrainingService = Depends(get_model_training_service),
    training_session_service: TrainingSessionService = Depends(
        get_training_session_service
    ),
):
    session = training_session_service.get_current_session()
    if not session:
        raise HTTPException(status_code=404, detail="Training session not found")

    result = model_training_service.random_sample_training()
    return result


# @app.middleware("http")
# async def log_responses(request: Request, call_next):
#     print("---")
#     print(f"start Endpoint: {request.url.path}")

#     response = await call_next(request)

#     response_body = b""
#     async for chunk in response.body_iterator:
#         response_body += chunk

#     print(f"raw response: {response_body}")
#     # Reconstruct the response
#     response = JSONResponse(
#         content=json.loads(response_body),
#         status_code=response.status_code,
#         headers=dict(response.headers),
#     )

#     print(f"Endpoint: {request.url.path}")
#     print(f"Response: {response_body.decode()}")
#     print("---")

#     return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=1127)

# how to run the server
# set port to 1127
# uvicorn backend.server:app --reload --port 1127
# uvicorn backend.server:app --port 1127
