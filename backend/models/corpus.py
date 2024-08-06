from mongoengine import Document, StringField, DateTimeField
from datetime import datetime


class Corpus(Document):
    name = StringField(required=True, unique=True)
    description = StringField()
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

    meta = {"collection": "corpora"}

    def update_timestamp(self):
        self.updated_at = datetime.utcnow()
        self.save()
