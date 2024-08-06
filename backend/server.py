from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import os

from database import init_db
from config import settings
from models.corpus import Corpus
from models.corpus_entry import CorpusEntry

device = "cuda"

# Initialize database connection
init_db()

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


class CorpusInput(BaseModel):
    name: str
    description: str = ""


class CorpusEntryInput(BaseModel):
    content: str
    entry_type: str
    corpus_name: str


# 定义模拟的大语言模型
class LargeLanguageModel:
    def chat(self, history):
        print(history)
        text = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer(text, return_tensors="pt").to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
        response = tokenizer.decode(
            generated_ids[0][len(model_inputs.input_ids[0]) :], skip_special_tokens=True
        )
        print(response)
        return response

    def learn(self, chat, knowledge):
        train_dataset = HeartEchoDataset(chat, knowledge, tokenizer, 1024)
        # 增量训练模型
        # 注意：你需要根据你的实际训练环境调整此部分
        training_args = TrainingArguments(
            #  指定训练过程中保存模型检查点和其他输出文件的目录。在这里，模型会被保存在当前工作目录下的 "results" 文件夹中。
            output_dir="./results",  # 输出目录
            #  设置训练的总轮数。在这里，模型将只训练一个完整的 epoch。对于增量学习或在线学习场景，这是合理的设置。
            num_train_epochs=1,  # 总训练轮次
            # 指定每个 GPU（如果有多个）上的训练批量大小。这个值相对较大，可能需要根据您的 GPU 内存来调整。
            # 每设备批量大小（per_device_train_batch_size）设置为 100 是相当大的。这可能需要大量的 GPU 内存。如果遇到内存不足的问题，您可能需要减小这个值。
            per_device_train_batch_size=32,  # 每个设备的批大小
            # 梯度累积允许您在多个小批次上累积梯度，然后一次性更新模型参数。
            # 允许使用更大的有效批量大小，而不增加 GPU 内存使用。
            # 可以提高训练稳定性，特别是对于小型数据集。
            gradient_accumulation_steps=4,
            # 设置学习率预热的步数。这里设为 0 意味着没有预热阶段，学习率将直接从初始值开始。对于短期或增量训练，这可能是合适的。
            # 没有学习率预热（warmup_steps = 0）对于短期训练来说是可以的，但如果您决定增加训练轮数，可能需要考虑添加预热步骤。
            warmup_steps=0,  # 预热步骤
            # 设置权重衰减率，用于 L2 正则化以防止过拟合。0.01 是一个常见的初始值，但可能需要根据具体情况调整。
            # 权重衰减（weight_decay）设置为 0.01 是一个好的起点，但您可能需要根据模型的表现来调整这个值。
            weight_decay=0.01,  # 权重衰减
            # 指定存储训练日志的目录。日志将被保存在当前工作目录下的 "logs" 文件夹中。
            logging_dir="./logs",  # 日志目录
            # 设置初始学习率。5e-5（0.00005）是一个较为保守的学习率，适合于大型语言模型的微调。
            # 学习率（learning_rate）设置为 5e-5 对于微调来说是比较合理的。但是，对于增量学习，您可能想尝试稍微更高的学习率，如 1e-4，以加快适应新数据的速度。
            learning_rate=5e-5,
            # FP16（半精度浮点数）：提供更大的内存节省和速度提升，但可能在某些情况下导致数值不稳定。
            fp16=True,
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
            content=corpus_entry_input.content,
            entry_type=corpus_entry_input.entry_type,
            corpus=corpus,
        )
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
