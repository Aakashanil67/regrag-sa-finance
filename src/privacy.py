"""Explicit local privacy policy for logging and persistent response caching."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacySettings:
    log_raw_content: bool
    log_raw_model_output: bool
    cache_enabled: bool
    cache_ttl_seconds: int


def _strict_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "true" if default else "false")
    if raw not in {"true", "false", "TRUE", "FALSE"}:
        raise ValueError(f"{name} must be exactly true or false")
    return raw.lower() == "true"


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def privacy_settings() -> PrivacySettings:
    """Read and validate policy at operation time, rather than import time."""
    return PrivacySettings(
        log_raw_content=_strict_bool("LOG_RAW_CONTENT", False),
        log_raw_model_output=_strict_bool("LOG_RAW_MODEL_OUTPUT", False),
        cache_enabled=_strict_bool("CACHE_ENABLED", False),
        cache_ttl_seconds=_positive_int("CACHE_TTL_SECONDS", 86400),
    )
