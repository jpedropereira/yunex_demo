"""Config tests. No running stack required -- these must pass in CI or offline."""

import pytest

from yunex_demo.config import Settings


@pytest.fixture
def clean_env(monkeypatch):
    """Isolate from a developer's real .env and exported overrides."""
    for key in (
        "OLLAMA_HOST",
        "CHROMA_HOST",
        "CHROMA_PORT",
        "MODEL_MAIN",
        "MODEL_EMBED",
        "OLLAMA_NUM_CTX",
    ):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _settings() -> Settings:
    # _env_file=None ignores any .env on disk, so defaults are what we assert on.
    return Settings(_env_file=None)


def test_defaults_target_a_16gb_laptop(clean_env):
    s = _settings()
    assert s.model_main == "gemma4:e2b-it-qat"
    assert s.model_embed == "embeddinggemma-2:270m"
    # The 128K window would blow an 8GB card's KV cache.
    assert s.ollama_num_ctx == 8192


def test_generation_is_deterministic_by_default(clean_env):
    s = _settings()
    assert s.temperature == 0.0
    assert s.seed == 7


def test_weights_stay_resident(clean_env):
    assert _settings().ollama_keep_alive == -1


def test_chroma_url_is_built_from_host_and_port(clean_env):
    s = Settings(_env_file=None, chroma_host="somewhere", chroma_port=9999)
    assert s.chroma_url == "http://somewhere:9999"


def test_required_models_covers_both(clean_env):
    s = _settings()
    assert s.required_models == [s.model_main, s.model_embed]


def test_env_overrides_apply(clean_env):
    clean_env.setenv("MODEL_MAIN", "gemma3:1b-it-qat")
    clean_env.setenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    s = _settings()
    assert s.model_main == "gemma3:1b-it-qat"
    assert s.ollama_host == "http://host.docker.internal:11434"
