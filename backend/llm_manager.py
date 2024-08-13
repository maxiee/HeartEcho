import os
import random
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
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


class DynamicHeartEchoDataset(Dataset):
    def __init__(self, entries: List[CorpusEntry], tokenizer):
        self.entries = entries
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        if entry.entry_type == "chat":
            text = self.tokenizer.apply_chat_template(
                entry.messages,
                tokenize=False,
                add_generation_prompt=False,
                chat_template=TEMPLATE,
            )
        else:  # knowledge
            text = entry.content

        encodings = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        # 修改：创建 attention_mask
        attention_mask = torch.ones_like(encodings.input_ids[0])

        return {
            "input_ids": encodings.input_ids[0],
            "attention_mask": attention_mask,
            "labels": encodings.input_ids[0].clone(),
        }


def collate_fn(batch):
    max_length = max(len(item["input_ids"]) for item in batch)

    input_ids = torch.full((len(batch), max_length), IGNORE_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    labels = torch.full((len(batch), max_length), IGNORE_TOKEN_ID, dtype=torch.long)

    for i, item in enumerate(batch):
        input_len = len(item["input_ids"])
        input_ids[i, :input_len] = item["input_ids"]
        attention_mask[i, :input_len] = item["attention_mask"]
        labels[i, :input_len] = item["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
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

        # 修改：设置 pad_token_id
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

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

    def chat(self, history, session_name: str):
        self._load_model_if_not_loaded(session_name)

        text = self.tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        attention_mask = torch.ones(
            input_ids.shape, dtype=torch.long, device=self.device
        )
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask,
            max_new_tokens=2048,
            do_sample=True,
            # Temperature (温度)
            # 温度参数控制生成过程中的随机性。
            # 较低的温度（例如 0.7）会让模型更“保守”，即更倾向于选择概率更高的下一个词，从而生成更确定性的输出；
            # 较高的温度（例如 1.5）会增加随机性，使模型更愿意选择那些概率相对较低的词，从而生成更多样化和出乎意料的输出。
            temperature=0.9,
            # Top-p (核采样)
            # 这个参数会在模型生成时考虑概率累积达到 p 值的词集合。
            # 例如，top_p=0.95 意味着只考虑那些使得概率总和达到 95% 的词，而忽略剩下的低概率词。
            # 这种方法可以动态地调整被考虑的词数量，确保模型生成的内容在合理范围内又不失随机性。
            # top_p 像是在生成时只考虑“最有可能的一群词”，而不是所有可能的词，从而保持生成的合理性和多样性之间的平衡。
            top_p=0.85,
            # Top-k (最高 k 采样)
            # 在生成过程中，只从概率最高的前 k 个词中进行采样。
            # 设置 top_k=50，意味着模型只会从最可能的前 50 个词中选择下一个词。这种方法能有效地减少生成中引入的随机性，确保输出的连贯性。
            # top_k 就像是在生成时限制“视野”，只看最可能的几个词，忽略那些可能性极低的词，从而使生成更加稳妥。
            top_k=50,
        )
        response = self.tokenizer.decode(
            generated_ids[0][len(model_inputs.input_ids[0]) :], skip_special_tokens=True
        )
        return response.strip()

    def _count_tokens(self, entry: CorpusEntry) -> int:
        if not self.tokenizer:
            raise ValueError("Tokenizer is not initialized. Call load_model() first.")

        if entry.entry_type == "knowledge":
            return len(self.tokenizer.encode(entry.content))
        elif entry.entry_type == "chat":
            # Apply chat template and count tokens
            formatted_chat = self.tokenizer.apply_chat_template(
                entry.messages,
                tokenize=False,
                add_generation_prompt=False,
                chat_template=TEMPLATE,
            )
            return len(self.tokenizer.encode(formatted_chat))
        else:
            raise ValueError(f"Unknown entry type: {entry.entry_type}")

    def train_on_entries(self, session_name: str, entries: List[CorpusEntry]) -> float:
        # 确保模型已加载到正确的设备上
        self._load_model_if_not_loaded(session_name)

        # 创建数据集
        train_dataset = DynamicHeartEchoDataset(entries, self.tokenizer)

        # 创建数据加载器，batch_size=1 确保每次只处理一个条目
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,  # 随机打乱数据
            collate_fn=collate_fn,  # 使用自定义的 collate 函数
        )

        # 将模型设置为训练模式
        self.model.train()

        # 初始化优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        total_loss = 0.0  # 用于累积和计算平均损失
        accumulated_loss = 0.0  # 用于当前梯度累积周期的损失
        total_tokens = sum(self._count_tokens(entry) for entry in entries)
        max_token_length = 0
        max_token_entry = None

        print("开始训练过程！我们将一步步学习新的知识。")
        print(f"我们总共有 {len(train_dataloader)} 条数据要学习。")
        print("我们会每学习16条数据后，就整理一下我们学到的东西。")

        # 遍历数据集
        for step, batch in enumerate(train_dataloader):
            # 计算当前批次的 token 长度
            token_length = batch["input_ids"].size(1)  # 获取序列长度
            print(
                f"\n--- 正在学习第 {step + 1} 条数据 (Token 长度: {token_length}) ---"
            )
            # 每批只有1个条目，打印出这个条目的内容的前100个 token的字符串表示
            print(
                f"内容: {self.tokenizer.decode(batch['input_ids'][0, :100], skip_special_tokens=True)}"
            )

            # 更新最大 token 长度
            if token_length > max_token_length:
                max_token_length = token_length
                max_token_entry = step + 1

            # 将批次数据移动到正确的设备上
            batch = {k: v.to(self.device) for k, v in batch.items()}
            print("1. 我已经仔细阅读了这条数据。")

            try:

                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss
                print(
                    f"2. 我尝试理解这条数据，并估算了我的理解程度。我的理解误差是: {loss.item():.4f}"
                )

                # Calculate the gradient weight based on token proportion
                gradient_weight = token_length / total_tokens
                weighted_loss = loss * gradient_weight
                print(
                    f"3. Weighted loss (based on token proportion): {weighted_loss.item():.4f}"
                )

                # 反向传播
                weighted_loss.backward()
                print("4. Backward pass completed.")

                accumulated_loss += loss.item() * gradient_weight
                total_loss += loss.item() * gradient_weight

                # 每16步或在最后一步执行优化器步骤
                if (step + 1) % 16 == 0 or (step + 1) == len(train_dataloader):
                    # 执行优化器步骤
                    optimizer.step()
                    # 清零梯度
                    optimizer.zero_grad()

                    print(
                        f"5. 我已经学习了16条数据（或所有数据），现在我要整理一下我学到的东西。"
                    )
                    print(
                        f"   在这16条数据中，我的平均理解误差是: {accumulated_loss:.4f}"
                    )
                    accumulated_loss = 0.0
                else:
                    print("5. 我还没学够16条数据，我会继续学习下一条。")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        f"警告：处理第 {step + 1} 条数据时内存不足。这条数据的 token 长度为 {token_length}。"
                    )
                    print("跳过这条数据并继续训练。")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise e

        # 计算平均损失
        average_loss = total_loss
        print(f"\n训练结束！在整个学习过程中，我的平均理解误差是: {average_loss:.4f}")
        print("这个数字越小，说明我学得越好！")
        print(
            f"最长的语料是第 {max_token_entry} 条，长度为 {max_token_length} 个 token。"
        )
        return average_loss

    def calculate_entry_loss(self, entry: CorpusEntry) -> float:
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            if entry.entry_type == "knowledge":
                text = entry.content
            elif entry.entry_type == "chat":
                text = self.tokenizer.apply_chat_template(
                    entry.messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    chat_template=TEMPLATE,
                )
            else:
                raise ValueError(f"Unknown entry type: {entry.entry_type}")

            # Calculate the token count of the input
            input_ids = self.tokenizer.encode(text, add_special_tokens=True)
            token_count = len(input_ids)

            # Set max_length to the minimum of token count and model's max length
            max_length = min(token_count, self.tokenizer.model_max_length)

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = inputs["input_ids"].clone()

            # If the input was truncated, adjust the labels
            # 对于被截断的输入，我们将最后一个 token 的 label 设置为 -100。这是因为最后一个 token 可能是不完整的，我们不希望模型为预测这个不完整的 token 而受到惩罚。
            # 在 PyTorch 中，特别是在处理序列任务时，-100 被广泛用作忽略特定位置损失计算的标志。这是 PyTorch 内部的一个约定。
            # Hugging Face Transformers 库的实现：Hugging Face Transformers 库遵循了这个约定。在其实现中，-100 被用作一个特殊的标记，表示在计算损失时应该忽略这个位置。
            if token_count > max_length:
                print(
                    f"Warning: Input was truncated from {token_count} to {max_length} tokens."
                )
                labels[:, -1] = -100  # Ignore loss for the last token if truncated

            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss.item()

        return loss

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
        return os.path.join(settings.MODEL_SAVE_PATH, session_name)
