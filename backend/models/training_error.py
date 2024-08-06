from mongoengine import Document, FloatField, ReferenceField, StringField, DateTimeField
from datetime import datetime

from models.error_range import ErrorRange


class TrainingError(Document):
    error_range = ReferenceField(ErrorRange, required=True)
    corpus_entry = ReferenceField("CorpusEntry", required=True)
    session = StringField(required=True)
    timestamp = DateTimeField(default=datetime.utcnow)

    meta = {"collection": "training_errors"}

    @classmethod
    def record_error(cls, error, corpus_entry, session):
        error_range = ErrorRange.get_range_for_error(error)
        if error_range:
            cls(
                error_range=error_range, corpus_entry=corpus_entry, session=session
            ).save()

    @classmethod
    def get_distribution(cls, session):
        pipeline = [
            {"$match": {"session": session}},
            {"$group": {"_id": "$error_range", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ]
        return list(cls.objects.aggregate(pipeline))
