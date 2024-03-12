from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import os

device = "cuda"
model_dir = "./trained"

# ================================================================
# LLM 模型部分
# ================================================================

# 加载模型和分词器
if os.path.exists(model_dir):
    model = AutoModelForCausalLM.from_pretrained(model_dir)
else:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-1.8B-Chat", torch_dtype="auto", device_map="auto"
    )
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

# if os.path.exists(model_dir):
#     model = AutoModelForCausalLM.from_pretrained(model_dir)
# else:
#     model = AutoModelForCausalLM.from_pretrained(
#         "Qwen/Qwen1.5-0.5B-Chat", torch_dtype="auto", device_map="auto"
#     )
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")

IGNORE_TOKEN_ID = LabelSmoother.ignore_index  # 设置忽略令牌的ID，用于损失计算时忽略

def preprocess(messages, tokenizer, max_len):
    print("preprocessing")
    print(messages)

    message_with_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        padding=True,
        max_length=max_len,
        truncation=True,
    )
    message_with_prompt = message_with_prompt + "<|im_end|>"

    print(message_with_prompt)

    texts = tokenizer(
        message_with_prompt,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_len,
    )

    input_ids = texts.input_ids
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = texts.attention_mask

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
    )


class SupervisedDataset(Dataset):

    def __init__(self, messages, tokenizer, max_len):
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
        model_inputs = tokenizer(text, return_tensors="pt").to(device)
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        return response

    def learn_knowledge(self, knowledge):
        print(f"知识长度 {len(knowledge)}")
        train_dataset = KnowledgeDataset(knowledge, tokenizer, max_len=1024)
        # 增量训练模型
        # 注意：你需要根据你的实际训练环境调整此部分
        training_args = TrainingArguments(
            output_dir="./results",  # 输出目录
            num_train_epochs=1,  # 总训练轮次
            per_device_train_batch_size=1,  # 每个设备的批大小
            warmup_steps=0,  # 预热步骤
            weight_decay=0.01,  # 权重衰减
            logging_dir="./logs",  # 日志目录
            learning_rate=2e-5,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,  # 使用新的训练数据
            # 这里可能还需要一个评估数据集
        )
        trainer.train()
        return "ok"

    def learn_chat(self, conversations):
        # 这里应该是你的模型逻辑，现在只是返回一个简单的响应
        train_dataset = SupervisedDataset(conversations, tokenizer, 1024)
        # 增量训练模型
        # 注意：你需要根据你的实际训练环境调整此部分
        training_args = TrainingArguments(
            output_dir="./results",  # 输出目录
            num_train_epochs=1,  # 总训练轮次
            per_device_train_batch_size=1,  # 每个设备的批大小
            warmup_steps=0,  # 预热步骤
            weight_decay=0.01,  # 权重衰减
            logging_dir="./logs",  # 日志目录
            learning_rate=2e-5,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,  # 使用新的训练数据
            # 这里可能还需要一个评估数据集
        )
        trainer.train()
        return "ok"


# 实例化模型
api_model = LargeLanguageModel()


@app.post("/chat")
def chat(chat_input: ChatInput):
    return {"response": api_model.chat(chat_input.history)}


@app.post("/learn_knowledge")
def learn_knowledge(knowledge_input: KnowledgeInput):
    return {"response": api_model.learn_knowledge(knowledge_input.knowledge)}


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
