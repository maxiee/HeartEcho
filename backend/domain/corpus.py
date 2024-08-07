from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class CorpusEntry:
    """表示单个语料条目，可以是对话或知识。"""

    id: str
    corpus_id: str
    content: str
    entry_type: str  # 'chat' or 'knowledge'
    created_at: datetime
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"CorpusEntry(id={self.id}, type={self.entry_type}, created_at={self.created_at})"


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
