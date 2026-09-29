from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    pinecone_api_key: str

    pinecone_index_name: str = "agentic-ai-rag-free"
    pinecone_namespace: str = "agentic-ai"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"

    chunk_size: int = 800
    chunk_overlap: int = 100

    top_k: int = 8
    min_retrieval_score: float = 0.35
    max_context_chunks: int = 5
    grounding_threshold: float = 0.70

    app_title: str = "Agentic AI RAG Chatbot"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()