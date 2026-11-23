"""Settings read from environment variables (or a local .env file)."""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Answer generation. Any OpenAI-compatible chat API works; Groq (free tier) is the default.
    llm_provider: Literal["groq", "openai"] = "groq"
    groq_api_key: str | None = None
    openai_api_key: str | None = None
    llm_api_key: str | None = None   # optional: overrides the provider-specific key
    llm_base_url: str | None = None  # optional: any other OpenAI-compatible endpoint
    llm_model: str | None = None     # optional: overrides the provider's default model

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = "./data/chroma"
    collection_name: str = "documents"

    chunk_size: int = 800      # characters per chunk  (GUESS: tune after trying real docs)
    chunk_overlap: int = 150   # characters shared between neighbouring chunks
    top_k: int = 4             # chunks retrieved per question (plan says 3 to 5)
    max_upload_mb: int = 20

    @property
    def llm(self) -> dict:
        """The resolved LLM settings: name, key, key env var, base URL and model."""
        preset = PROVIDERS[self.llm_provider]
        own_key = self.groq_api_key if self.llm_provider == "groq" else self.openai_api_key
        return {
            "provider": preset["name"],
            "api_key": self.llm_api_key or own_key,
            "key_env": preset["key_env"],
            "base_url": self.llm_base_url or preset["base_url"],
            "model": self.llm_model or preset["model"],
        }


PROVIDERS = {
    # Groq retired its Llama 3.x models on the free tier in Aug 2026; openai/gpt-oss-120b is its
    # recommended replacement. Use LLM_MODEL=openai/gpt-oss-20b for a smaller, faster model.
    "groq": {"name": "Groq", "key_env": "GROQ_API_KEY", "base_url": "https://api.groq.com/openai/v1",
             "model": "openai/gpt-oss-120b"},
    "openai": {"name": "OpenAI", "key_env": "OPENAI_API_KEY", "base_url": None, "model": "gpt-4o-mini"},
}


@lru_cache
def get_settings() -> Settings:
    return Settings()
