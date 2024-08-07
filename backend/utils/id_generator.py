import uuid


class IdGenerator:
    @staticmethod
    def generate() -> str:
        return str(uuid.uuid4())
