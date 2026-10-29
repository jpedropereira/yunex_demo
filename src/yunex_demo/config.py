"""Single source of truth for runtime configuration.

Everything is env-driven so the same image runs under every compose profile
(cpu / gpu / external) with only ``OLLAMA_HOST`` changing. Import ``settings``
rather than reading ``os.environ`` directly.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Where the Ollama HTTP API lives. The `external` profile points this at the
    # host (macOS has no path to the GPU from inside a container).
    ollama_host: str = "http://ollama:11434"

    chroma_host: str = "chroma"
    chroma_port: int = 8000
    chroma_collection: str = "yunex_demo"

    # gemma4 "e2b" is an effective-2B edge build: ~4.3GB, fits a 16GB laptop
    # alongside the embedder. Drop to gemma3:1b-it-qat on 8GB machines.
    model_main: str = "gemma4:e2b-it-qat"
    model_embed: str = "embeddinggemma-2:270m"

    # The 128K window would blow an 8GB card's KV cache for no demo benefit.
    ollama_num_ctx: int = 8192

    # Seconds to keep weights resident; -1 means indefinitely, so nothing
    # unloads mid-presentation. Must be numeric: the API parses a *string*
    # value as a Go duration, and "-1" has no unit, so it 400s.
    ollama_keep_alive: int = -1

    # Generation is deterministic: the talk is about reliability, and a demo that
    # answers differently on each run undercuts the point.
    temperature: float = 0.0
    seed: int = Field(default=7, description="Fixed seed for reproducible output")

    @property
    def chroma_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"

    @property
    def required_models(self) -> list[str]:
        return [self.model_main, self.model_embed]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
