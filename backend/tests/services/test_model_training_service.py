import unittest
from unittest.mock import Mock, patch
from datetime import datetime
from domain.model import Model
from domain.corpus import Corpus
from domain.training_session import TrainingSession
from services.model_training_service import ModelTrainingService


class TestModelTrainingService(unittest.TestCase):
    def setUp(self):
        self.mock_model_repo = Mock()
        self.mock_corpus_repo = Mock()
        self.service = ModelTrainingService(self.mock_model_repo, self.mock_corpus_repo)

    @patch("services.model_training_service.generate_id")
    def test_start_training_session(self, mock_generate_id):
        mock_generate_id.return_value = "session1"
        mock_model = Model(
            id="model1",
            name="Test Model",
            base_model="base",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parameters=1000000,
        )
        mock_corpus = Corpus(
            id="corpus1",
            name="Test Corpus",
            description="Test Description",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        self.mock_model_repo.get_by_id.return_value = mock_model
        self.mock_corpus_repo.get_by_id.return_value = mock_corpus

        result = self.service.start_training_session("model1", "corpus1")

        self.assertIsInstance(result, TrainingSession)
        self.assertEqual(result.id, "session1")
        self.assertEqual(result.model, mock_model)
        self.assertEqual(result.corpus, mock_corpus)
        self.assertEqual(result.status, "in_progress")

        self.mock_model_repo.get_by_id.assert_called_once_with("model1")
        self.mock_corpus_repo.get_by_id.assert_called_once_with("corpus1")

    def test_start_training_session_invalid_ids(self):
        self.mock_model_repo.get_by_id.return_value = None
        self.mock_corpus_repo.get_by_id.return_value = None

        with self.assertRaises(ValueError):
            self.service.start_training_session("invalid_model", "invalid_corpus")

    @patch("services.model_training_service.calculate_new_error")
    def test_train(self, mock_calculate_new_error):
        mock_model = Mock()
        mock_corpus = Mock(entries=[Mock(), Mock()])
        mock_session = Mock(model=mock_model, corpus=mock_corpus)

        mock_calculate_new_error.return_value = 0.1

        self.service._train(mock_session)

        self.assertEqual(mock_session.status, "completed")
        mock_model.update_error.assert_called_once_with(0.1)
        mock_session.complete.assert_called_once()

    def test_train_exception(self):
        mock_session = Mock()
        mock_session.corpus.entries = [Mock(side_effect=Exception("Test exception"))]

        self.service._train(mock_session)

        self.assertEqual(mock_session.status, "failed")
        mock_session.fail.assert_called_once_with("Test exception")


if __name__ == "__main__":
    unittest.main()
