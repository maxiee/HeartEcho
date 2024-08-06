import os
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_pt_utils import LabelSmoother
from config import settings

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"


class HeartEchoDataset(Dataset):
    def __init__(self, chats, knowledges, tokenizer, max_len):
        self.examples = []
        for chat in chats:
            chat_text = tokenizer.apply_chat_template(
                chat.messages,
                tokenize=False,
                add_generation_prompt=False,
                chat_template=TEMPLATE,
            )
            self.examples.append(chat_text)
        for knowledge in knowledges:
            self.examples.append(knowledge.content)
        self.encodings = tokenizer(
            self.examples,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=max_len,
        )

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.encodings.input_ids[i],
            "attention_mask": self.encodings.attention_mask[i],
            "labels": self.encodings.input_ids[i].clone(),
        }


class LLMManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        if os.path.exists(settings.model_dir):
            self.model = AutoModelForCausalLM.from_pretrained(settings.model_dir)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.model_name, torch_dtype="auto", device_map="auto"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_name)
        self.model.to(self.device)

    def chat(self, history):
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        text = self.tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
        response = self.tokenizer.decode(
            generated_ids[0][len(model_inputs.input_ids[0]) :], skip_special_tokens=True
        )
        return response.strip()

    def learn(self, chat_entries, knowledge_entries):
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        chats = [entry for entry in chat_entries if entry.entry_type == "chat"]
        knowledges = [
            entry for entry in knowledge_entries if entry.entry_type == "knowledge"
        ]
        train_dataset = HeartEchoDataset(
            chats, knowledges, self.tokenizer, max_len=1024
        )

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir="./logs",
            learning_rate=5e-5,
            fp16=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()
        return "ok"

    def save_model(self):
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.model.save_pretrained(settings.model_dir)
        self.tokenizer.save_pretrained(settings.model_dir)
        return True
