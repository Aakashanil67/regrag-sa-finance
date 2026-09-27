import pytest

from src.privacy import PrivacySettings, privacy_settings


def test_privacy_defaults_disable_content_logging_and_cache(monkeypatch):
    for name in (
        "LOG_RAW_CONTENT",
        "LOG_RAW_MODEL_OUTPUT",
        "CACHE_ENABLED",
        "CACHE_TTL_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    assert privacy_settings() == PrivacySettings(
        log_raw_content=False,
        log_raw_model_output=False,
        cache_enabled=False,
        cache_ttl_seconds=86400,
    )


def test_privacy_accepts_only_strict_boolean_values(monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "TRUE")
    monkeypatch.setenv("LOG_RAW_CONTENT", "false")
    monkeypatch.setenv("LOG_RAW_MODEL_OUTPUT", "true")

    settings = privacy_settings()

    assert settings.cache_enabled is True
    assert settings.log_raw_content is False
    assert settings.log_raw_model_output is True


@pytest.mark.parametrize("value", ["1", "yes", "on", "", " true "])
def test_privacy_rejects_malformed_boolean_values(monkeypatch, value):
    monkeypatch.setenv("CACHE_ENABLED", value)

    with pytest.raises(ValueError, match="CACHE_ENABLED"):
        privacy_settings()


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_privacy_requires_a_positive_cache_ttl(monkeypatch, value):
    monkeypatch.setenv("CACHE_TTL_SECONDS", value)

    with pytest.raises(ValueError, match="CACHE_TTL_SECONDS"):
        privacy_settings()
