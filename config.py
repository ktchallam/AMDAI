from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "meta-llama/Llama-3-70B-Instruct"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    hot_store_collection: str = "hot_store"
    cold_store_collection: str = "cold_store"
    hot_store_chunk_size: int = 256
    cold_store_chunk_size: int = 512
    relevance_gate_threshold: float = 0.75
    top_n_questions: int = 20
    chroma_persist_dir: str = "./chroma_db"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        extra="ignore"
    )

@lru_cache()
def get_config() -> AppConfig:
    """Returns a cached singleton instance of AppConfig."""
    return AppConfig()
