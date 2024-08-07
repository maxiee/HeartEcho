from mongoengine import Document, IntField, ReferenceField, StringField, DateTimeField
from datetime import datetime


class TrainingLoss(Document):
    # 指示一个语料，在一个 session 下的微调训练误差
    error_range = IntField(required=True)
    corpus_entry = ReferenceField("CorpusEntry", required=True)
    # 对应的会话 id
    session = StringField(required=True)
    timestamp = DateTimeField(default=datetime.utcnow)

    meta = {"collection": "training_errors"}
