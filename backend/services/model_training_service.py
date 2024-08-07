from datetime import datetime
from typing import List
from domain.model import Model
from domain.corpus import Corpus
from domain.training_session import TrainingSession
from repositories.model_repository import ModelRepository
from repositories.corpus.corpus_repository import CorpusRepository
from utils.id_generator import IdGenerator


class ModelTrainingService:
    def __init__(self, model_repo: ModelRepository, corpus_repo: CorpusRepository):
        self.model_repo = model_repo
        self.corpus_repo = corpus_repo

    def start_training_session(self, model_id: str, corpus_id: str) -> TrainingSession:
        model = self.model_repo.get_by_id(model_id)
        corpus = self.corpus_repo.get_by_id(corpus_id)

        if not model or not corpus:
            raise ValueError("Invalid model or corpus ID")

        session = TrainingSession(
            id=IdGenerator.generate(),
            model=model,
            corpus=corpus,
            start_time=datetime.now(),
        )

        # Start the training process (this could be an async task)
        self._train(session)

        return session

    def _train(self, session: TrainingSession):
        try:
            # Actual training logic here
            # This is a simplified example
            for entry in session.corpus.entries:
                # Process each entry
                pass

            # Update model and complete session
            session.model.update_error(calculate_new_error(session))
            session.complete()
        except Exception as e:
            session.fail(str(e))

    def get_training_results(self, session_id: str) -> dict:
        # Retrieve and return training results
        pass
