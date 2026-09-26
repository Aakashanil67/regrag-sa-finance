from types import SimpleNamespace

from src import llm


def test_defaults_to_openai_luna(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    settings = llm.effective_llm_settings()
    assert (settings.provider, settings.model) == ("openai", "gpt-5.6-luna")


def test_empty_env_values_fall_back(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    settings = llm.effective_llm_settings()
    assert (settings.provider, settings.model) == ("openai", "gpt-5.6-luna")


def test_openai_request_and_cost(monkeypatch):
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200),
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: fake)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    result = llm.complete("sys", "user", max_tokens=123)
    assert seen["model"] == "gpt-5.6-luna"
    assert seen["max_completion_tokens"] == 123
    assert "max_tokens" not in seen
    assert "temperature" not in seen
    assert result.cost_usd == (1000 * 0.20 + 200 * 1.20) / 1e6


def test_older_model_keeps_temperature_and_max_tokens(monkeypatch):
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: fake)
    llm.complete("s", "u", max_tokens=50, settings=llm.settings_for("openai", "gpt-4o-mini"))
    assert seen["max_tokens"] == 50
    assert seen["temperature"] == 0.0
