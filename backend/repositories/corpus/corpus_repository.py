from abc import ABC, abstractmethod
from typing import List, Optional
from domain.corpus import Corpus, CorpusEntry


class CorpusRepository(ABC):
    """处理语料库的持久化操作。"""

    @abstractmethod
    def get_by_id(self, corpus_id: str) -> Optional[Corpus]:
        pass

    @abstractmethod
    def save(self, corpus: Corpus) -> Corpus:
        pass

    @abstractmethod
    def list(self, skip: int = 0, limit: int = 100) -> List[Corpus]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def delete(self, corpus_id: str) -> bool:
        pass

    @abstractmethod
    def update(self, corpus: Corpus) -> Corpus:
        pass
