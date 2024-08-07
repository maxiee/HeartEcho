import unittest
from datetime import datetime
from domain.corpus import Corpus
from repositories.corpus.memory_corpus_repository import MemoryCorpusRepository


class TestCorpusRepository(unittest.TestCase):
    def setUp(self):
        self.repo = MemoryCorpusRepository()

    def test_get_by_id(self):
        corpus = Corpus(
            id="1",
            name="Test Corpus",
            description="Test Description",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.repo.save(corpus)
        result = self.repo.get_by_id("1")
        self.assertEqual(result.id, corpus.id)
        self.assertEqual(result.name, corpus.name)
        self.assertEqual(result.description, corpus.description)

    def test_get_by_id_not_found(self):
        result = self.repo.get_by_id("non_existent_id")
        self.assertIsNone(result)

    def test_save(self):
        corpus = Corpus(
            id="1",
            name="Test Corpus",
            description="Test Description",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        result = self.repo.save(corpus)
        self.assertEqual(result.id, corpus.id)
        self.assertEqual(result.name, corpus.name)
        self.assertEqual(result.description, corpus.description)

        saved_corpus = self.repo.get_by_id("1")
        self.assertEqual(saved_corpus.id, corpus.id)
        self.assertEqual(saved_corpus.name, corpus.name)
        self.assertEqual(saved_corpus.description, corpus.description)

    def test_list(self):
        corpora = [
            Corpus(
                id=f"{i}",
                name=f"Corpus {i}",
                description=f"Description {i}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            for i in range(5)
        ]
        for corpus in corpora:
            self.repo.save(corpus)

        result = self.repo.list(skip=0, limit=10)
        self.assertEqual(len(result), 5)
        result_ids = set(corpus.id for corpus in result)
        expected_ids = set(corpus.id for corpus in corpora)
        self.assertEqual(result_ids, expected_ids)

    def test_list_with_skip_and_limit(self):
        corpora = [
            Corpus(
                id=f"{i}",
                name=f"Corpus {i}",
                description=f"Description {i}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            for i in range(10)
        ]
        for corpus in corpora:
            self.repo.save(corpus)

        result = self.repo.list(skip=2, limit=3)
        self.assertEqual(len(result), 3)
        result_ids = [corpus.id for corpus in result]
        expected_ids = [corpus.id for corpus in corpora[2:5]]
        self.assertEqual(result_ids, expected_ids)

    def test_delete(self):
        corpus = Corpus(
            id="1",
            name="Test Corpus",
            description="Test Description",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.repo.save(corpus)
        result = self.repo.delete("1")
        self.assertTrue(result)
        self.assertIsNone(self.repo.get_by_id("1"))

    def test_delete_non_existent(self):
        result = self.repo.delete("non_existent_id")
        self.assertFalse(result)

    def test_update(self):
        corpus = Corpus(
            id="1",
            name="Test Corpus",
            description="Test Description",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.repo.save(corpus)

        updated_corpus = Corpus(
            id="1",
            name="Updated Corpus",
            description="Updated Description",
            created_at=corpus.created_at,
            updated_at=datetime.now(),
        )
        result = self.repo.update(updated_corpus)
        self.assertEqual(result.id, updated_corpus.id)
        self.assertEqual(result.name, updated_corpus.name)
        self.assertEqual(result.description, updated_corpus.description)

        saved_corpus = self.repo.get_by_id("1")
        self.assertEqual(saved_corpus.name, "Updated Corpus")
        self.assertEqual(saved_corpus.description, "Updated Description")

    def test_update_non_existent(self):
        non_existent_corpus = Corpus(
            id="999",
            name="Non-existent Corpus",
            description="This corpus doesn't exist",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        with self.assertRaises(ValueError):
            self.repo.update(non_existent_corpus)


if __name__ == "__main__":
    unittest.main()
