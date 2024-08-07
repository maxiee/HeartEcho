from typing import List, Optional, Dict
from domain.corpus import Corpus
from repositories.corpus.corpus_repository import CorpusRepository


class MemoryCorpusRepository(CorpusRepository):
    def __init__(self):
        self.corpora: Dict[str, Corpus] = {}

    def get_by_id(self, corpus_id: str) -> Optional[Corpus]:
        return self.corpora.get(corpus_id)

    def save(self, corpus: Corpus) -> Corpus:
        self.corpora[corpus.id] = corpus
        return corpus

    def list(self, skip: int = 0, limit: int = 100) -> List[Corpus]:
        corpora_list = list(self.corpora.values())
        return corpora_list[skip : skip + limit]

    def delete(self, corpus_id: str) -> bool:
        if corpus_id in self.corpora:
            del self.corpora[corpus_id]
            return True
        return False

    def update(self, corpus: Corpus) -> Corpus:
        if corpus.id in self.corpora:
            self.corpora[corpus.id] = corpus
            return corpus
        raise ValueError(f"Corpus with id {corpus.id} not found")
