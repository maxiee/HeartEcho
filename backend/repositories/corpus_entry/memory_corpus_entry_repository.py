from typing import List, Optional, Dict
from app.core.db import DB
from domain.corpus import CorpusEntry
from repositories.corpus_entry.corpus_entry_repository import CorpusEntryRepository


class MemoryCorpusEntryRepository(CorpusEntryRepository):
    def __init__(self):
        DB.init()
        self.entries: Dict[str, CorpusEntry] = {}

    def get_by_id(self, entry_id: str) -> Optional[CorpusEntry]:
        return self.entries.get(entry_id)

    def save(self, entry: CorpusEntry) -> CorpusEntry:
        self.entries[entry.id] = entry
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
            del self.entries[entry_id]
            return True
        return False
