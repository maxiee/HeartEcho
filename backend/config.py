from pydantic import BaseSettings


class Settings(BaseSettings):
    mongodb_url: str = "mongodb://100.117.209.140:27017/heartecho"
    model_dir: str = "./trained"
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    tokenizer_name: str = "Qwen/Qwen2-7B-Instruct"


settings = Settings()
