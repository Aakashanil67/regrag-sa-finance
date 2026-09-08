"""Provider-agnostic chat completion: anthropic (default), openai, or a local ollama, picked by
the LLM_PROVIDER env var so the same rag.py code runs on a paid API key or entirely offline.

Cost is an estimate, not a bill: `PRICING_PER_MILLION_TOKENS` in config.py is a snapshot of public
list pricing, not a live lookup, and Ollama is genuinely free (local compute). It exists so
reports/logs can show an order-of-magnitude "well, that call cost about $0.0003", not an
invoice-grade figure.
"""

import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from src.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    PRICING_PER_MILLION_TOKENS,
    RAG_MAX_ANSWER_TOKENS,
)

load_dotenv()


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    host: str | None = None


def effective_llm_settings() -> LLMSettings:
    """The one place provider/model/temperature are resolved from the environment — both the
    generation call and provenance.pipeline_fingerprint() call this, so a fingerprint can never
    silently disagree with what a request actually sent."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
    if not 0 <= temperature <= 1:
        raise ValueError("LLM_TEMPERATURE must be between 0 and 1")

    if provider == "anthropic":
        model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        host = None
    elif provider == "openai":
        model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        host = None
    elif provider == "ollama":
        model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    else:
        raise ValueError(
            f"LLM_PROVIDER={provider!r} not supported — use one of anthropic/openai/ollama"
        )

    return LLMSettings(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=RAG_MAX_ANSWER_TOKENS,
        host=host,
    )


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_rate, output_rate = PRICING_PER_MILLION_TOKENS.get(model, (0.0, 0.0))
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def _complete_anthropic(
    settings: LLMSettings, system: str, user: str, max_tokens: int
) -> LLMResponse:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=settings.model,
        max_tokens=max_tokens,
        temperature=settings.temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = next((block.text for block in response.content if block.type == "text"), "")
    return LLMResponse(
        text=text,
        model=settings.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=_estimate_cost(
            settings.model, response.usage.input_tokens, response.usage.output_tokens
        ),
    )


def _complete_openai(settings: LLMSettings, system: str, user: str, max_tokens: int) -> LLMResponse:
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=settings.model,
        max_tokens=max_tokens,
        temperature=settings.temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    usage = response.usage
    return LLMResponse(
        text=response.choices[0].message.content or "",
        model=settings.model,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cost_usd=_estimate_cost(settings.model, usage.prompt_tokens, usage.completion_tokens),
    )


def _complete_ollama(settings: LLMSettings, system: str, user: str, max_tokens: int) -> LLMResponse:
    """Ollama has no official Python SDK in this project's dependency list — its REST API is
    small and stable enough that raw HTTP is the simpler, lighter-weight choice here."""
    response = httpx.post(
        f"{settings.host}/api/chat",
        json={
            "model": settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": settings.temperature},
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    return LLMResponse(
        text=data["message"]["content"],
        model=settings.model,
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
        cost_usd=0.0,
    )


_PROVIDERS = {
    "anthropic": _complete_anthropic,
    "openai": _complete_openai,
    "ollama": _complete_ollama,
}


def complete(system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
    settings = effective_llm_settings()
    return _PROVIDERS[settings.provider](
        settings, system, user, max_tokens if max_tokens is not None else settings.max_tokens
    )
