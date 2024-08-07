from mongoengine import connect
from app.core.config import settings


class DB:
    inited = False

    @staticmethod
    def init():
        if not DB.inited:
            connect(host=settings.MONGODB_URL)
            DB.inited = True
