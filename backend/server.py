from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import os

device = "cuda"
model_dir = "./trained"

# ================================================================
# LLM 模型部分
# ================================================================

model_name = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer_name = "Qwen/Qwen1.5-1.8B-Chat"

# model_name = "Qwen/Qwen1.5-0.5B-Chat"
# tokenizer_name = "Qwen/Qwen1.5-0.5B-Chat"


# 加载模型和分词器
if os.path.exists(model_dir):
    model = AutoModelForCausalLM.from_pretrained(model_dir)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index  # 设置忽略令牌的ID，用于损失计算时忽略


def preprocess_chat(messages, tokenizer, max_len):
    print("preprocessing")
    print(messages)

    message_with_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    message_with_prompt = message_with_prompt + "<|im_end|>"
    return message_with_prompt


class HeartEchoDataset(Dataset):
    def __init__(
        self,
        chats: List[List[Dict[str, Any]]],
        knowledges: List[str],
        tokenizer,
        max_len,
    ):
        print(max_len)
        str_list = []

        for chat in chats:
            str_list.append(preprocess_chat(chat, tokenizer, max_len))

        for knowledge in knowledges:
            str_list.append(knowledge)

        self.examples = tokenizer(
            str_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_len,
        )

        self.input_ids = self.examples.input_ids
        self.target_ids = self.input_ids.clone()
        self.target_ids[self.target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
        self.attention_mask = self.examples.attention_mask

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.target_ids[i],
        }


# ================================================================
# FastAPI 部分
# ================================================================

app = FastAPI()


class ChatInput(BaseModel):
    history: List[Dict[str, Any]]


class LearnInput(BaseModel):
    chat: List[List[Dict[str, Any]]]
    knowledge: List[str]


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

    def learn(self, chat, knowledge):
        train_dataset = HeartEchoDataset(chat, knowledge, tokenizer, 1024)
        # 增量训练模型
        # 注意：你需要根据你的实际训练环境调整此部分
        training_args = TrainingArguments(
            output_dir="./results",  # 输出目录
            num_train_epochs=1,  # 总训练轮次
            per_device_train_batch_size=100,  # 每个设备的批大小
            warmup_steps=0,  # 预热步骤
            weight_decay=0.01,  # 权重衰减
            logging_dir="./logs",  # 日志目录
            learning_rate=5e-5,
        )

        # def custom_collate_fn(batch):
        #     # 使用DataCollatorWithPadding确保批处理中所有项都填充到相同长度
        #     collator = DataCollatorWithPadding(
        #         padding=True, tokenizer=tokenizer, return_tensors="pt"
        #     )
        #     return collator(batch)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,  # 使用新的训练数据
            # data_collator=custom_collate_fn,  # 使用自定义的数据合并函数
            # 这里可能还需要一个评估数据集
        )
        trainer.train()
        return "ok"


# 实例化模型
api_model = LargeLanguageModel()


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


# how to run the server
# set port to 1127
# uvicorn backend.server:app --reload --port 1127
# uvicorn backend.server:app --port 1127
