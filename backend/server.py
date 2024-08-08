import json
import os
import time
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.core.dependencies import get_training_session_service
from domain.corpus import CorpusEntry
from llm_manager import LLMManager

import logging
from repositories.corpus.mongodb_corpus_repository import MongoDBCorpusRepository
from repositories.corpus_entry.mongodb_corpus_entry_repository import (
    MongoDBCorpusEntryRepository,
)
from services.corpus_management_service import CorpusManagementService
import app.api.routes.corpus as corpus_routes
import app.api.routes.sessions as sessions_routes
from app.core.config import settings

logger = logging.getLogger(__name__)

app = FastAPI()

print(corpus_routes.router.routes)

app.include_router(corpus_routes.router, prefix="/corpus", tags=["corpus"])
app.include_router(sessions_routes.router, prefix="/sessions", tags=["sessions"])
# app.include_router(model.router, prefix="/model", tags=["model"])
# app.include_router(training.router, prefix="/training", tags=["training"])


def get_corpus_service():
    corpus_repo = MongoDBCorpusRepository(connection_string=settings.mongodb_url)
    corpus_entry_repo = MongoDBCorpusEntryRepository(
        connection_string=settings.mongodb_url
    )
    return CorpusManagementService(corpus_repo, corpus_entry_repo)


class ChatInput(BaseModel):
    history: List[Dict[str, Any]]


class LearnInput(BaseModel):
    chat: List[List[Dict[str, Any]]]
    knowledge: List[str]


class MessageInput(BaseModel):
    role: str = Field(
        ...,
        description="The role of the message sender (e.g., 'user', 'assistant', 'system')",
    )
    content: str = Field(..., description="The content of the message")


class CorpusEntryInput(BaseModel):
    corpus_name: str = Field(
        ..., description="The name of the corpus this entry belongs to"
    )
    entry_type: str = Field(
        ..., description="The type of the entry: 'chat' or 'knowledge'"
    )
    content: Optional[str] = Field(
        None, description="The content for 'knowledge' type entries"
    )
    messages: Optional[List[MessageInput]] = Field(
        None, description="The list of messages for 'chat' type entries"
    )

    class Config:
        schema_extra = {
            "example": {
                "corpus_name": "my_corpus",
                "entry_type": "chat",
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {
                        "role": "assistant",
                        "content": "I'm doing well, thank you for asking. How can I assist you today?",
                    },
                ],
            }
        }


@app.post("/chat")
def chat(chat_input: ChatInput):
    try:
        response = llm_manager.chat(chat_input.history)
        return {"response": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/learn")
def learn(input: LearnInput):
    try:
        result = llm_manager.learn(input.chat, input.knowledge)
        return {"message": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_model")
def save_model():
    try:
        result = llm_manager.save_model()
        return {"saved": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/saved_models")
async def get_saved_models():
    try:
        # Get the list of saved models from the model directory
        model_dir = settings.model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        saved_models = [
            f
            for f in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, f))
        ]
        return {"models": saved_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


class LoadModelInput(BaseModel):
    model_name: str


@app.post("/load_model")
async def load_model(input: LoadModelInput):
    try:
        model_path = os.path.join(settings.model_dir, input.model_name)
        print(f"Loading model from {model_path}")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")

        # Load the specified model
        llm_manager.load_model(model_path)

        return {"message": f"Model {input.model_name} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SmeltCorpusInput(BaseModel):
    session: str


@app.post("/smelt_new_corpus")
async def smelt_new_corpus(input: SmeltCorpusInput):
    try:
        result = llm_manager.smelt_new_corpus(input.session)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/error_distribution")
async def get_error_distribution(
    session: str = Query(..., description="Training session name")
):
    distribution = llm_manager.get_error_distribution(session)
    ranges = ErrorRange.objects.order_by("lower_bound")

    data = []
    range_index = 0
    for range in ranges:
        count = 0
        if range_index < len(distribution) and distribution[range_index]["_id"] == str(
            range.id
        ):
            count = distribution[range_index]["count"]
            range_index += 1
        data.append(
            {
                "range": f"{range.lower_bound:.1f}-{range.upper_bound:.1f}",
                "count": count,
            }
        )

    return {"distribution": data}


@app.get("/new_corpus_entries_count")
async def get_new_corpus_entries_count(
    session: str = Query(..., description="Training session name")
):
    try:
        total_corpus_entries = CorpusEntry.objects.count()
        trained_entries_count = TrainingError.objects(session=session).count()
        new_entries_count = total_corpus_entries - trained_entries_count
        return {"new_entries_count": max(0, new_entries_count)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
