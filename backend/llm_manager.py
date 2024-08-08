import os
import random
from typing import List
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_pt_utils import LabelSmoother
from domain.corpus import CorpusEntry
from domain.training_session import TrainingSession
from models.training_loss import TrainingLoss
from app.core.config import settings

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
        self.cached_errors = {}

    def load_model(self, model_path):
        print(f"Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)

    def _load_model_if_not_loaded(self, session_name: str):
        model_dir = self._get_model_dir_from_session_name(session_name)
        if not self.model or not self.tokenizer:
            self.load_model(model_dir)

    def init_new_model(self, base_model: str):
        # Initialize a new model from the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

    def chat(self, history):
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        text = self.tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=2048,
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
            chats, knowledges, self.tokenizer, max_len=2048
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
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()
        return "ok"

    def train_on_entries(self, session_name: str, entries: List[CorpusEntry]) -> float:
        self._load_model_if_not_loaded(session_name)
        # Prepare data for training
        chat_entries = [entry for entry in entries if entry.entry_type == "chat"]
        knowledge_entries = [
            entry for entry in entries if entry.entry_type == "knowledge"
        ]

        train_dataset = HeartEchoDataset(
            chat_entries, knowledge_entries, self.tokenizer, max_len=2048
        )

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_steps=0,
            logging_steps=1,
            learning_rate=5e-6,
            save_strategy="no",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Train and get the loss
        train_result = trainer.train()
        return train_result.training_loss

    def get_error_distribution(self):
        current_session = self.training_session_service.get_current_session()
        if not current_session:
            raise ValueError("No active training session")

        # Get distribution from database
        db_distribution = TrainingError.get_distribution(current_session.name)

        # Convert db_distribution to a dictionary for easier manipulation
        distribution_dict = {
            str(item["_id"]): item["count"] for item in db_distribution
        }

        # Get cached errors for this session
        cached_errors = self.cached_errors.get(current_session.name, {})

        # Process cached errors
        for entry_id, error in cached_errors.items():
            error_range = ErrorRange.get_range_for_error(error)
            if error_range:
                range_id = str(error_range.id)
                if range_id in distribution_dict:
                    # If this range already exists in db_distribution, increment the count
                    distribution_dict[range_id] += 1
                else:
                    # If this is a new range, add it to the distribution
                    distribution_dict[range_id] = 1

        # Convert the updated distribution back to the original format
        updated_distribution = [
            {"_id": range_id, "count": count}
            for range_id, count in distribution_dict.items()
        ]

        # Sort the distribution by error range
        updated_distribution.sort(
            key=lambda x: ErrorRange.objects(id=x["_id"]).first().lower_bound
        )

        return updated_distribution

    def save_model(self, session: TrainingSession):
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")

        model_dir = self._get_model_dir_from_session_name(session.name)

        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        return True

    def _get_model_dir_from_session_name(self, session_name: str):
        return os.path.join("./trained", session_name)
