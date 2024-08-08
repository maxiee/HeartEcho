from llm_manager import LLMManager
from repositories.corpus_entry.corpus_entry_repository import CorpusEntryRepository
from services.training_loss_service import TrainingLossService
from services.training_session_service import TrainingSessionService


class ModelTrainingService:
    def __init__(
        self,
        llm_manager: LLMManager,
        corpus_entry_repo: CorpusEntryRepository,
        training_session_service: TrainingSessionService,
        training_loss_service: TrainingLossService,
    ):
        self.llm_manager = llm_manager
        self.corpus_entry_repo = corpus_entry_repo
        self.training_session_service = training_session_service
        self.training_loss_service = training_loss_service

    def smelt_new_corpus(self, batch_size: int = 16) -> dict:
        assert (
            self.training_session_service.get_current_session()
        ), "No active training session"
        # Randomly sample batch_size entries
        selected_entries = self.corpus_entry_repo.sample_new_entries(batch_size)

        # Train the model
        loss = self.llm_manager.train_on_entries(
            self.training_session_service.get_current_session().name, selected_entries
        )

        for entry in selected_entries:
            self.training_loss_service.update_loss(entry.id, loss, entry)

        return {
            "message": "New corpus smelting completed",
            "loss": loss,
            "entries_trained": len(selected_entries),
        }
