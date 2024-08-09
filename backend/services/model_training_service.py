from typing import List
from domain.corpus import CorpusEntry
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

    def _count_tokens(self, entry: CorpusEntry) -> int:
        if entry.entry_type == "knowledge":
            return len(self.llm_manager.tokenizer.encode(entry.content))
        elif entry.entry_type == "chat":
            return sum(
                len(self.llm_manager.tokenizer.encode(msg["content"]))
                for msg in entry.messages
            )
        else:
            raise ValueError(f"Unknown entry type: {entry.entry_type}")

    def sample_new_entries(self, batch_size: int, session_id: str) -> List[CorpusEntry]:
        # 前置检查：确保新语料数量大于 batch_size
        total_entries = self.corpus_entry_repo.count()
        new_entries_count = self.training_loss_service.get_new_corpus_entries_count(
            session_id, total_entries
        )

        if new_entries_count < batch_size:
            raise ValueError(
                f"New corpus entries count is less than batch size: {new_entries_count} < {batch_size}"
            )

        return self.corpus_entry_repo.sample_new_entries(
            batch_size, total_entries, session_id
        )

    def smelt_new_corpus(self, batch_size: int = 16) -> dict:
        assert (
            self.training_session_service.get_current_session()
        ), "No active training session"
        self.llm_manager._load_model_if_not_loaded(
            self.training_session_service.get_current_session().name
        )
        # Randomly sample batch_size entries
        selected_entries = self.sample_new_entries(
            batch_size, self.training_session_service.get_current_session().id
        )

        total_tokens = sum(self._count_tokens(entry) for entry in selected_entries)

        # Train the model
        loss = self.llm_manager.train_on_entries(
            self.training_session_service.get_current_session().name, selected_entries
        )

        self.training_session_service.update_tokens_trained(total_tokens)

        for entry in selected_entries:
            self.training_loss_service.update_loss(
                entry.id,
                loss,
                self.training_session_service.get_current_session(),
            )

        return {
            "message": "New corpus smelting completed",
            "loss": loss,
            "entries_trained": len(selected_entries),
        }
