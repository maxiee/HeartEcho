from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from typing import Dict, List, Optional


@dataclass
class CorpusEntry:
    """表示单个语料条目，可以是对话或知识。"""

    id: str
    corpus: str
    entry_type: str  # 'chat' or 'knowledge'
    created_at: datetime
    content: Optional[str] = None  # For 'knowledge' type
    messages: Optional[List[Dict[str, str]]] = None  # For 'chat' type
    metadata: dict = field(default_factory=dict)
    sha256: str = field(init=False)

    def __post_init__(self):
        if self.entry_type == "knowledge" and self.content is None:
            raise ValueError("Content is required for knowledge entry type")
        if self.entry_type == "chat" and self.messages is None:
            raise ValueError("Messages are required for chat entry type")
        if self.entry_type not in ["knowledge", "chat"]:
            raise ValueError("Invalid entry type. Must be 'knowledge' or 'chat'")
        if self.entry_type == "knowledge" and self.messages is not None:
            raise ValueError("Messages should not be provided for knowledge entry type")
        if self.entry_type == "chat" and self.content is not None:
            raise ValueError("Content should not be provided for chat entry type")
        self.sha256 = self.calculate_sha256()

    def calculate_sha256(self) -> str:
        if self.entry_type == "knowledge":
            data = self.content
        elif self.entry_type == "chat":
            data = json.dumps(self.messages, sort_keys=True)
        else:
            raise ValueError(f"Unknown entry_type: {self.entry_type}")

        return hashlib.sha256(data.encode()).hexdigest()

    def __repr__(self):
        return f"CorpusEntry(id={self.id}, type={self.entry_type}, created_at={self.created_at}, sha256={self.sha256})"


@dataclass
class Corpus:
    """表示语料库，包含多个语料条目。"""

    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime

    def __repr__(self):
        return f"Corpus(id={self.id}, name={self.name})"
