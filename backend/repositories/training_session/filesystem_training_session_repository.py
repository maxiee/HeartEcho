import os
import json
from typing import List, Optional
from datetime import datetime
from domain.training_session import TrainingSession
from repositories.training_session.training_session_repository import (
    TrainingSessionRepository,
)


class FileSystemTrainingSessionRepository(TrainingSessionRepository):
    def __init__(self, base_path: str = "./trained"):
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def create(self, session: TrainingSession) -> TrainingSession:
        session_path = os.path.join(self.base_path, session.name)
        if os.path.exists(session_path):
            raise ValueError(f"Session with name {session.name} already exists")
        os.makedirs(session_path)
        self._save_session_info(session)
        return session

    def get_by_id(self, session_id: str) -> Optional[TrainingSession]:
        for session_name in os.listdir(self.base_path):
            session_path = os.path.join(self.base_path, session_name)
            if os.path.isdir(session_path):
                info_path = os.path.join(session_path, "session_info.json")
                if os.path.exists(info_path):
                    with open(info_path, "r") as f:
                        info = json.load(f)
                        if info["id"] == session_id:
                            return self._create_session_from_info(info)
        return None

    def update(self, session: TrainingSession) -> TrainingSession:
        session_path = os.path.join(self.base_path, session.name)
        if not os.path.exists(session_path):
            raise ValueError(f"Session with name {session.name} does not exist")
        self._save_session_info(session)
        return session

    def list_active_sessions(self) -> List[TrainingSession]:
        active_sessions = []
        for session_name in os.listdir(self.base_path):
            session_path = os.path.join(self.base_path, session_name)
            if os.path.isdir(session_path):
                info_path = os.path.join(session_path, "session_info.json")
                if os.path.exists(info_path):
                    with open(info_path, "r") as f:
                        info = json.load(f)
                        session = self._create_session_from_info(info)
                        active_sessions.append(session)
        return active_sessions

    def delete(self, session_id: str) -> bool:
        for session_name in os.listdir(self.base_path):
            session_path = os.path.join(self.base_path, session_name)
            if os.path.isdir(session_path):
                info_path = os.path.join(session_path, "session_info.json")
                if os.path.exists(info_path):
                    with open(info_path, "r") as f:
                        info = json.load(f)
                        if info["id"] == session_id:
                            import shutil

                            shutil.rmtree(session_path)
                            return True
        return False

    def _save_session_info(self, session: TrainingSession):
        session_path = os.path.join(self.base_path, session.name)
        info_path = os.path.join(session_path, "session_info.json")
        info = {
            "id": session.id,
            "name": session.name,
            "model": {
                "id": session.model.id,
                "name": session.model.name,
                "base_model": session.model.base_model,
                "parameters": session.model.parameters,
                "created_at": session.model.created_at.isoformat(),
                "updated_at": session.model.updated_at.isoformat(),
            },
            "start_time": session.start_time.isoformat(),
            "last_trained": session.last_trained.isoformat(),
            "metrics": session.metrics,
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def _create_session_from_info(self, info: dict) -> TrainingSession:
        return TrainingSession(
            id=info["id"],
            name=info["name"],
            model=info["base_model"],
            start_time=datetime.fromisoformat(info["start_time"]),
            last_trained=datetime.fromisoformat(info["last_trained"]),
            metrics=info["metrics"],
        )
