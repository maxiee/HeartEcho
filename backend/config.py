from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongodb_url: str = "mongodb://100.117.209.140:27017/heartecho"
    model_dir: str = "./trained"
    model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    tokenizer_name: str = "Qwen/Qwen2-1.5B-Instruct"


settings = Settings()
