import unittest
from datetime import datetime
from domain.corpus import CorpusEntry
from repositories.corpus_entry.memory_corpus_entry_repository import (
    MemoryCorpusEntryRepository,
)


class TestCorpusEntryRepository(unittest.TestCase):
    def setUp(self):
        self.repo = MemoryCorpusEntryRepository()

    def test_get_by_id(self):
        entry = CorpusEntry(
            id="1",
            corpus="corpus1",
            content="Test content",
            entry_type="knowledge",
            created_at=datetime.now(),
            metadata={},
        )
        self.repo.save(entry)
        result = self.repo.get_by_id("1")
        self.assertEqual(result, entry)

    def test_get_by_id_not_found(self):
        result = self.repo.get_by_id("non_existent_id")
        self.assertIsNone(result)

    def test_save(self):
        entry = CorpusEntry(
            id="1",
            corpus="corpus1",
            content="Test content",
            entry_type="knowledge",
            created_at=datetime.now(),
            metadata={},
        )
        result = self.repo.save(entry)
        self.assertEqual(result, entry)
        self.assertEqual(self.repo.get_by_id("1"), entry)

    def test_list_by_corpus(self):
        entries = [
            CorpusEntry(
                id=f"{i}",
                corpus="corpus1",
                content=f"Content {i}",
                entry_type="knowledge",
                created_at=datetime.now(),
                metadata={},
            )
            for i in range(5)
        ]
        for entry in entries:
            self.repo.save(entry)

        # Add an entry from a different corpus
        self.repo.save(
            CorpusEntry(
                id="6",
                corpus="corpus2",
                content="Other corpus content",
                entry_type="chat",
                created_at=datetime.now(),
                metadata={},
            )
        )

        result = self.repo.list_by_corpus("corpus1", skip=0, limit=10)
        self.assertEqual(len(result), 5)
        result_ids = set(entry.id for entry in result)
        expected_ids = set(entry.id for entry in entries)
        self.assertEqual(result_ids, expected_ids)

    def test_list_by_corpus_with_skip_and_limit(self):
        entries = [
            CorpusEntry(
                id=f"{i}",
                corpus="corpus1",
                content=f"Content {i}",
                entry_type="knowledge",
                created_at=datetime.now(),
                metadata={},
            )
            for i in range(10)
        ]
        for entry in entries:
            self.repo.save(entry)

        result = self.repo.list_by_corpus("corpus1", skip=2, limit=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, entries[2:5])

    def test_delete(self):
        entry = CorpusEntry(
            id="1",
            corpus="corpus1",
            content="Test content",
            entry_type="knowledge",
            created_at=datetime.now(),
            metadata={},
        )
        self.repo.save(entry)
        result = self.repo.delete("1")
        self.assertTrue(result)
        self.assertIsNone(self.repo.get_by_id("1"))

    def test_delete_non_existent(self):
        result = self.repo.delete("non_existent_id")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
