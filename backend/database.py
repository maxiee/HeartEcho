from mongoengine import connect
from .config import settings


def init_db():
    connect(host=settings.mongodb_url)
