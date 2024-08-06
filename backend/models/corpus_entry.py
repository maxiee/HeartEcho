from mongoengine import (
    Document,
    StringField,
    ReferenceField,
    DateTimeField,
    ListField,
    DictField,
)
from datetime import datetime


class CorpusEntry(Document):
    entry_type = StringField(choices=["chat", "knowledge"], required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    corpus = ReferenceField("Corpus", required=True)

    # For 'knowledge' type
    content = StringField()

    # For 'chat' type
    messages = ListField(DictField(), default=list)

    meta = {"collection": "corpus_entries"}
