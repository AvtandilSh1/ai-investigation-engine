from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "hashing-384"

    chroma_path: str = str(ROOT / "data" / "chroma")
    docs_path: str = str(ROOT / "data" / "documents")
    meta_db_path: str = str(ROOT / "data" / "meta.sqlite3")

    chunk_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 8
    hybrid_alpha: float = 0.55

    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
