from abc import ABC, abstractmethod
from typing import List, Optional
from domain.corpus import CorpusEntry


class CorpusEntryRepository(ABC):
    @abstractmethod
    def get_by_id(self, entry_id: str) -> Optional[CorpusEntry]:
        pass

    @abstractmethod
    def get_entries_by_ids(self, entry_ids: List[str]) -> List[CorpusEntry]:
        pass

    @abstractmethod
    def save(self, entry: CorpusEntry) -> CorpusEntry:
        pass

    @abstractmethod
    def list_by_corpus(
        self, corpus: str, skip: int = 0, limit: int = 100
    ) -> List[CorpusEntry]:
        pass

    @abstractmethod
    def sample_new_entries(
        self, batch_size: int, total_entries: int, session_id: str
    ) -> List[CorpusEntry]:
        pass

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        pass

    @abstractmethod
    def count(self) -> int:
        pass
