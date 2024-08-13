from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "HeartEcho"
    VERSION: str = "0.1.0"
    MONGODB_URL: str = "mongodb://100.117.209.140:27017/heartecho"
    MODEL_SAVE_PATH: str = "./trained"  # 默认路径

    class Config:
        env_file = ".env"


settings = Settings()
