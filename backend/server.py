from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import os

device = "cuda"
model_dir = "./0.5B-trained"

# ================================================================
# LLM 模型部分
# ================================================================

# 加载模型和分词器
if os.path.exists(model_dir):
    model = AutoModelForCausalLM.from_pretrained(model_dir)
else:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-0.5B-Chat", torch_dtype="auto", device_map="auto"
    )
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")

IGNORE_TOKEN_ID = LabelSmoother.ignore_index  # 设置忽略令牌的ID，用于损失计算时忽略


def preprocess(messages, tokenizer, max_len):
    print("preprocessing")

    texts = []
    for message in messages:
        # 将对话格式应用于每组消息
        texts.append(
            tokenizer.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=False,
                padding=True,
                max_length=max_len,
                truncation=True,
            )
        )
    input_ids = torch.tensor(texts, dtype=torch.long)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
    )


class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_len):
        messages = [example["messages"] for example in raw_data]
        data_dict = preprocess(messages, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.target_ids = data_dict["target_ids"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.target_ids[i],
            attention_mask=self.attention_mask[i],
        )


class KnowledgeDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_len):
        texts = tokenizer(
            raw_data,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_len,
        )
        print("总Token数：", texts.input_ids.numel())
        self.input_ids = texts.input_ids
        self.target_ids = self.input_ids.clone()
        self.target_ids[self.target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
        self.attention_mask = texts.attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.target_ids[i],
            attention_mask=self.attention_mask[i],
        )


# ================================================================
# FastAPI 部分
# ================================================================

app = FastAPI()


class ChatInput(BaseModel):
    history: List[Dict[str, Any]]


class KnowledgeInput(BaseModel):
    knowledge: str


class ChatMLInput(BaseModel):
    conversations: List[Dict[str, Any]]


# 定义模拟的大语言模型
class LargeLanguageModel:
    def chat(self, history):
        print(history)
        text = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        return f"{response}"

    def learn_knowledge(self, knowledge):
        # 这里应该是你的模型逻辑，现在只是返回一个简单的响应
        return {"learned": True, "params": {"knowledge": knowledge}}

    def learn_chat(self, conversations):
        # 这里应该是你的模型逻辑，现在只是返回一个简单的响应
        return {"learned": True, "params": {"conversations": len(conversations)}}


# 实例化模型
api_model = LargeLanguageModel()


@app.post("/chat")
def chat(chat_input: ChatInput):
    return {"response": api_model.chat(chat_input.history)}


@app.post("/learn_knowledge")
def learn_knowledge(knowledge_input: KnowledgeInput):
    return api_model.learn_knowledge(knowledge_input.knowledge)


@app.post("/learn_chat")
def learn_chat(chatml_input: ChatMLInput):
    return api_model.learn_chat(chatml_input.conversations)


@app.post("/save_model")
def save_model():
    # 这里应该是你的模型保存逻辑，现在只是返回一个简单的响应
    return {"saved": True}


# how to run the server
# set port to 1127
# uvicorn backend.server:app --reload --port 1127
# uvicorn backend.server:app --port 1127
