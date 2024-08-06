from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import os

from database import init_db
from config import settings
from llm_manager import LLMManager
from models.corpus import Corpus
from models.corpus_entry import CorpusEntry

# Initialize database connection
init_db()

app = FastAPI()

# Initialize LLMManager
llm_manager = LLMManager()


class ChatInput(BaseModel):
    history: List[Dict[str, Any]]


class LearnInput(BaseModel):
    chat: List[List[Dict[str, Any]]]
    knowledge: List[str]


class CorpusInput(BaseModel):
    name: str
    description: str = ""


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


# ================================================================
# LLM 模型部分
# ================================================================

IGNORE_TOKEN_ID = LabelSmoother.ignore_index  # 设置忽略令牌的ID，用于损失计算时忽略

TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"


# 加载模型和分词器
if os.path.exists(settings.model_dir):
    model = AutoModelForCausalLM.from_pretrained(settings.model_dir)
else:
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name, torch_dtype="auto", device_map="auto"
    )
tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_name)


def preprocess_chat(messages, tokenizer, max_len):
    print("preprocessing")
    print(messages)

    message_with_prompt = tokenizer.apply_chat_template(
        messages,
        chat_template=TEMPLATE,
        tokenize=True,
        add_generation_prompt=False,
        padding="max_length",
        max_length=max_len,
        truncation=True,
    )
    return message_with_prompt


@app.post("/chat")
def chat(chat_input: ChatInput):
    return {"response": api_model.chat(chat_input.history)}


@app.post("/learn")
def learn(input: LearnInput):
    api_model.learn(input.chat, input.knowledge)
    return "ok"


@app.post("/save_model")
def save_model():
    # 这里应该是你的模型保存逻辑，现在只是返回一个简单的响应
    return {"saved": True}


@app.post("/corpus")
async def create_corpus(corpus_input: CorpusInput):
    try:
        corpus = Corpus(name=corpus_input.name, description=corpus_input.description)
        corpus.save()
        return {"message": "Corpus created successfully", "id": str(corpus.id)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/corpus_entry")
async def create_corpus_entry(corpus_entry_input: CorpusEntryInput):
    try:
        corpus = Corpus.objects(name=corpus_entry_input.corpus_name).first()
        if not corpus:
            raise HTTPException(status_code=404, detail="Corpus not found")

        corpus_entry = CorpusEntry(
            entry_type=corpus_entry_input.entry_type,
            corpus=corpus,
        )

        if corpus_entry_input.entry_type == "chat":
            corpus_entry.messages = corpus_entry_input.messages
        else:
            corpus_entry.content = corpus_entry_input.content

        corpus_entry.save()
        corpus.update_timestamp()

        return {
            "message": "corpus_entry created successfully",
            "id": str(corpus_entry.id),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/corpus")
async def get_corpora():
    corpora = Corpus.objects().to_json()
    return {"corpora": corpora}


@app.get("/corpus/{corpus_id}/corpus_entries")
async def get_corpus_entries(corpus_id: str, skip: int = 0, limit: int = 100):
    corpus = Corpus.objects(id=corpus_id).first()
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")

    total_count = CorpusEntry.objects(corpus=corpus).count()
    entries = CorpusEntry.objects(corpus=corpus).skip(skip).limit(limit)

    return {
        "total": total_count,
        "entries": entries.to_json(),
        "skip": skip,
        "limit": limit,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=1127)

# how to run the server
# set port to 1127
# uvicorn backend.server:app --reload --port 1127
# uvicorn backend.server:app --port 1127
