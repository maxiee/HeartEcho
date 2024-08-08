import unittest

import pytest
from repositories.corpus.memory_corpus_repository import MemoryCorpusRepository
from repositories.corpus_entry.memory_corpus_entry_repository import (
    MemoryCorpusEntryRepository,
)
from services.corpus_management_service import CorpusManagementService


class TestCorpusManagementService(unittest.TestCase):
    def setUp(self):
        self.corpus_repo = MemoryCorpusRepository()
        self.corpus_entry_repo = MemoryCorpusEntryRepository()
        self.service = CorpusManagementService(self.corpus_repo, self.corpus_entry_repo)

    def test_create_corpus(self):
        corpus = self.service.create_corpus("Test Corpus", "Test Description")
        self.assertIsNotNone(corpus.id)
        self.assertEqual(corpus.name, "Test Corpus")
        self.assertEqual(corpus.description, "Test Description")

        # Verify the corpus is saved in the repository
        saved_corpus = self.corpus_repo.get_by_id(corpus.id)
        self.assertEqual(saved_corpus.name, "Test Corpus")
        self.assertEqual(saved_corpus.description, "Test Description")

    def test_add_entry_to_corpus(self):
        corpus = self.service.create_corpus("Test Corpus", "Test Description")
        entry = self.service.add_entry_to_corpus(corpus.id, "knowledge", "Test Content")
        self.assertIsNotNone(entry.id)
        self.assertEqual(entry.corpus, corpus.id)
        self.assertEqual(entry.content, "Test Content")
        self.assertEqual(entry.entry_type, "knowledge")

        # Verify the entry is saved in the repository
        saved_entry = self.corpus_entry_repo.get_by_id(entry.id)
        self.assertEqual(saved_entry.corpus, corpus.id)
        self.assertEqual(saved_entry.content, "Test Content")
        self.assertEqual(saved_entry.entry_type, "knowledge")

    def test_add_knowledge_entry_to_corpus(self):
        corpus = self.service.create_corpus("Test Corpus", "Test Description")
        content = "This is a test knowledge entry."
        entry = self.service.add_entry_to_corpus(
            corpus_id=corpus.id, entry_type="knowledge", content=content
        )
        self.assertIsNotNone(entry.id)
        self.assertEqual(entry.corpus, corpus.id)
        self.assertEqual(entry.entry_type, "knowledge")
        self.assertEqual(entry.content, content)
        self.assertIsNone(entry.messages)

    def test_add_chat_entry_to_corpus(self):
        corpus = self.service.create_corpus("Test Corpus", "Test Description")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        entry = self.service.add_entry_to_corpus(
            corpus.id, entry_type="chat", messages=messages
        )
        self.assertIsNotNone(entry.id)
        self.assertEqual(entry.corpus, corpus.id)
        self.assertEqual(entry.entry_type, "chat")
        self.assertEqual(entry.messages, messages)
        self.assertIsNone(entry.content)

    def test_add_duplicate_entry_to_corpus(self):
        corpus = self.service.create_corpus("Test Corpus", "Test Description")
        self.service.add_entry_to_corpus(
            corpus.id, entry_type="knowledge", content="Duplicate content"
        )
        with pytest.raises(
            ValueError, match="A duplicate entry already exists in the corpus"
        ):
            self.service.add_entry_to_corpus(
                corpus.id, entry_type="knowledge", content="Duplicate content"
            )

    def test_add_entry_to_non_existent_corpus(self):
        with pytest.raises(ValueError, match="Corpus with id .* does not exist"):
            self.service.add_entry_to_corpus(
                "non_existent_corpus_id", entry_type="knowledge", content="Test content"
            )

    def test_remove_entry_from_corpus(self):
        corpus = self.service.create_corpus("Test Corpus", "Test Description")
        entry = self.service.add_entry_to_corpus(corpus.id, "knowledge", "Test Content")
        result = self.service.remove_entry_from_corpus(entry.id)
        self.assertTrue(result)

        # Verify the entry is removed from the repository
        self.assertIsNone(self.corpus_entry_repo.get_by_id(entry.id))

    def test_remove_non_existent_entry(self):
        result = self.service.remove_entry_from_corpus("non_existent_entry_id")
        self.assertFalse(result)

    def test_get_corpus_entries(self):
        corpus = self.service.create_corpus("Test Corpus", "Test Description")
        entries = [
            self.service.add_entry_to_corpus(corpus.id, "knowledge", f"Content {i}")
            for i in range(5)
        ]

        retrieved_entries = self.service.get_corpus_entries(corpus.id)
        self.assertEqual(len(retrieved_entries), 5)
        retrieved_ids = set(entry.id for entry in retrieved_entries)
        expected_ids = set(entry.id for entry in entries)
        self.assertEqual(retrieved_ids, expected_ids)

    def test_get_corpus_entries_with_pagination(self):
        corpus = self.service.create_corpus("Test Corpus", "Test Description")
        entries = [
            self.service.add_entry_to_corpus(corpus.id, "knowledge", f"Content {i}")
            for i in range(10)
        ]

        retrieved_entries = self.service.get_corpus_entries(corpus.id, skip=2, limit=3)
        self.assertEqual(len(retrieved_entries), 3)
        retrieved_ids = [entry.id for entry in retrieved_entries]
        expected_ids = [entry.id for entry in entries[2:5]]
        self.assertEqual(retrieved_ids, expected_ids)

    def test_get_entries_from_non_existent_corpus(self):
        entries = self.service.get_corpus_entries("non_existent_corpus_id")
        self.assertEqual(len(entries), 0)


if __name__ == "__main__":
    unittest.main()
