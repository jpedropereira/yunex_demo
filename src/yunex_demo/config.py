"""Single source of truth for runtime configuration.

Everything is env-driven so the same image runs under every compose profile
(cpu / gpu / external) with only ``OLLAMA_HOST`` changing. Import ``settings``
rather than reading ``os.environ`` directly.
"""

from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentFiling(BaseModel):
    """A company the agent may research and trade."""

    ticker: str
    company: str
    cik: int


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

    # The SEC requires automated clients to identify themselves: "Name email".
    # Empty by default, because it holds personal details that belong in .env.
    sec_user_agent: str = ""
    sec_collection: str = "sec_filings"
    embed_batch_size: int = 16

    # The ticker comes from this map, not the filing: Alphabet's lists two share classes.
    # This is also the agent's allowlist, so the two can never disagree.
    agent_filings: list[AgentFiling] = [
        AgentFiling(ticker="NVDA", company="NVIDIA", cik=1045810),
        AgentFiling(ticker="AAPL", company="Apple", cik=320193),
        AgentFiling(ticker="MSFT", company="Microsoft", cik=789019),
        AgentFiling(ticker="GOOGL", company="Alphabet", cik=1652044),
        AgentFiling(ticker="AMZN", company="Amazon", cik=1018724),
    ]

    # gemma4 "e2b" is an effective-2B edge build: ~4.3GB, fits a 16GB laptop
    # alongside the embedder. Drop to gemma3:1b-it-qat on 8GB machines.
    model_main: str = "gemma4:e2b-it-qat"
    model_embed: str = "embeddinggemma-2:270m"

    # Separate from model_main so this page alone can move up to E4B (`make pull-e4b`)
    # if tool calling turns out unreliable on E2B.
    agent_model: str = "gemma4:e2b-it-qat"
    agent_max_turns: int = 6
    trade_max_pct: Decimal = Decimal("5")
    agent_db_path: Path = Path("data/agent_demo.db")
    price_cache_path: Path = Path("data/price_cache.json")

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
        # dict.fromkeys: the agent model is usually the main model, and listed once.
        return list(dict.fromkeys([self.model_main, self.agent_model, self.model_embed]))

    @property
    def agent_tickers(self) -> list[str]:
        return [filing.ticker for filing in self.agent_filings]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
