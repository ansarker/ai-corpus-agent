from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_NAME: str = "gemma3"
    PERSIST_DIR: str = "db_store"
    EMBEDDING_MODEL_NAME: str = "nomic-embed-text"
    COLLECTION_NAME: str = "corpus_db"
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    ALLOWED_ORIGINS: list = [
        "http://localhost.com",
        "http://127.0.0.1",
        "http://localhost.com:5173",
        "http://127.0.0.1:5173",
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()