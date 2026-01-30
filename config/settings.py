from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NURA_",  # Phase 8: Environment variable prefix
        extra="ignore"  # Phase 8: Ignore extra env vars (like NURA_OPENAI_KEY)
    )

    sqlite_path: str = "Memory Storage/nura.db"
    embedding_dim: int = 384
    embedding_model: str = "all-MiniLM-L6-v2"
    summary_every_n_turns: int = 12
    default_top_k: int = 8
    use_real_embeddings: bool = True
    vector_index_path: str = "Memory Storage/vector_index.faiss"
    memory_jsonl_path: str = "Memory Storage/memory.jsonl"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cartesia_key: str = "your_cartesia_api_key_here"

settings = Settings()
