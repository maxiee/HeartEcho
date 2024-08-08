from typing import List, Optional, Dict
from domain.corpus import CorpusEntry
from repositories.corpus_entry.corpus_entry_repository import CorpusEntryRepository


class MemoryCorpusEntryRepository(CorpusEntryRepository):
    def __init__(self):
        self.entries: Dict[str, CorpusEntry] = {}
        self.sha256_index: Dict[str, str] = {}  # sha256 -> entry_id

    def get_by_id(self, entry_id: str) -> Optional[CorpusEntry]:
        return self.entries.get(entry_id)

    def save(self, entry: CorpusEntry) -> CorpusEntry:
        if entry.sha256 in self.sha256_index:
            raise ValueError("A duplicate entry already exists in the corpus")

        self.entries[entry.id] = entry
        self.sha256_index[entry.sha256] = entry.id
        return entry

    def list_by_corpus(
        self, corpus: str, skip: int = 0, limit: int = 100
    ) -> List[CorpusEntry]:
        corpus_entries = [
            entry for entry in self.entries.values() if entry.corpus == corpus
        ]
        return corpus_entries[skip : skip + limit]

    def delete(self, entry_id: str) -> bool:
        if entry_id in self.entries:
            entry = self.entries[entry_id]
            del self.sha256_index[entry.sha256]
            del self.entries[entry_id]
            return True
        return False
