from mongoengine import Document, StringField, ReferenceField, DateTimeField
from datetime import datetime


class CorpusEntry(Document):
    content = StringField(required=True)
    corpus_entry_type = StringField(choices=["chat", "knowledge"], required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    corpus = ReferenceField("Corpus", required=True)

    meta = {"collection": "corpus_entries"}
