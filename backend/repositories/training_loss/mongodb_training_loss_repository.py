from typing import List, Optional
from mongoengine import (
    Document,
    IntField,
    StringField,
    DateTimeField,
    FloatField,
    NotUniqueError,
    Q,
)
from app.core.db import DB
from domain.training_loss import TrainingLoss
from .training_loss_repository import TrainingLossRepository


class MongoTrainingLoss(Document):
    id = StringField(primary_key=True)
    corpus_entry_id = StringField(required=True)
    session_id = StringField(required=True)
    timestamp = DateTimeField(required=True)
    loss_value = FloatField(required=True)
    loss_rank = StringField(required=True, index=True)
    meta = {"collection": "training_losses"}


class MongoDBTrainingLossRepository(TrainingLossRepository):
    def __init__(self):
        DB.init()

    def save(self, training_loss: TrainingLoss) -> TrainingLoss:
        # 查询是否存在匹配的记录
        existing_loss = MongoTrainingLoss.objects(
            Q(corpus_entry_id=training_loss.corpus_entry_id)
            & Q(session_id=training_loss.session_id)
        ).first()

        if existing_loss:
            # 如果存在，更新记录
            existing_loss.loss_rank = training_loss.loss_rank
            existing_loss.timestamp = training_loss.timestamp
            existing_loss.loss_value = training_loss.loss_value
            existing_loss.save()
            print(
                f"Training loss updated for corpus_entry_id: {training_loss.corpus_entry_id}, session_id: {training_loss.session_id}"
            )
            return self._to_domain(existing_loss)
        else:
            # 如果不存在，创建新记录
            mongo_loss = MongoTrainingLoss(
                id=training_loss.id,
                loss_rank=training_loss.loss_rank,
                corpus_entry_id=training_loss.corpus_entry_id,
                session_id=training_loss.session_id,
                timestamp=training_loss.timestamp,
                loss_value=training_loss.loss_value,
            )
            mongo_loss.save()
            print(
                f"New training loss created for corpus_entry_id: {training_loss.corpus_entry_id}, session_id: {training_loss.session_id}"
            )
            return self._to_domain(mongo_loss)

    def get_by_id(self, training_loss_id: str) -> Optional[TrainingLoss]:
        mongo_loss = MongoTrainingLoss.objects(id=training_loss_id).first()
        return self._to_domain(mongo_loss) if mongo_loss else None

    def get_by_session_id(self, session_id: str) -> List[TrainingLoss]:
        mongo_losses = MongoTrainingLoss.objects(session_id=session_id)
        return [self._to_domain(ml) for ml in mongo_losses]

    def get_by_corpus_entry_id_and_session_id(
        self, corpus_entry_id: str, session_id: str
    ) -> TrainingLoss:
        mongo_losse = MongoTrainingLoss.objects(
            corpus_entry_id=corpus_entry_id, session_id=session_id
        ).first()
        return self._to_domain(mongo_losse)

    def count_by_loss_rank(self, session_id: str, loss_rank: str) -> int:
        return MongoTrainingLoss.objects(
            session_id=session_id, loss_rank=loss_rank
        ).count()

    def count_by_session_id(self, session_id: str) -> int:
        return MongoTrainingLoss.objects(session_id=session_id).count()

    def get_highest_loss_entries(
        self, session_id: str, limit: int
    ) -> List[TrainingLoss]:
        mongo_losses = (
            MongoTrainingLoss.objects(session_id=session_id)
            .order_by("-loss_value")
            .limit(limit)
        )
        return [self._to_domain(ml) for ml in mongo_losses]

    def get_lowest_loss_entries(
        self, session_id: str, limit: int
    ) -> List[TrainingLoss]:
        mongo_losses = (
            MongoTrainingLoss.objects(session_id=session_id)
            .order_by("loss_value")
            .limit(limit)
        )
        return [self._to_domain(ml) for ml in mongo_losses]

    def _to_domain(self, mongo_loss: MongoTrainingLoss) -> TrainingLoss:
        return TrainingLoss(
            id=str(mongo_loss.id),
            loss_rank=mongo_loss.loss_rank,
            corpus_entry_id=mongo_loss.corpus_entry_id,
            session_id=mongo_loss.session_id,
            timestamp=mongo_loss.timestamp,
            loss_value=mongo_loss.loss_value,
        )
