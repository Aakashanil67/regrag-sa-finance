"""The release protocol's core claim — reports/eval_summary.md and reports/runs/holdout-*.json
describe the code that actually ships — is otherwise enforced only by reviewer discipline before
each commit. This pins it as a real regression gate: any change that alters
provenance.pipeline_fingerprint() silently voids the sealed v1.1.0-rc2 holdout evidence, and this
test is what should turn red first.

Nothing is mocked. LLM_* env vars are cleared so a developer's local .env can't make this pass (or
fail) for a reason CI — which has none of these vars set — wouldn't see; effective_llm_settings()'s
own defaults (anthropic / claude-haiku-4-5 / temperature 0) are exactly what produced the sealed
run, so this needs no API key and no vector store.
"""

import json

from src import provenance
from src.config import REPORTS_DIR

CANONICAL_RUN_PATH = REPORTS_DIR / "runs" / "holdout-0cf5e0821ec7.json"

_LLM_ENV_VARS = (
    "LLM_PROVIDER",
    "ANTHROPIC_MODEL",
    "OPENAI_MODEL",
    "OLLAMA_MODEL",
    "OLLAMA_HOST",
    "LLM_TEMPERATURE",
)


def test_the_shipping_pipeline_fingerprint_still_matches_the_sealed_holdout_run(monkeypatch):
    for var in _LLM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    sealed_fingerprint = json.loads(CANONICAL_RUN_PATH.read_text(encoding="utf-8"))[
        "pipeline_fingerprint"
    ]

    current_fingerprint = provenance.pipeline_fingerprint(k=5)

    assert current_fingerprint == sealed_fingerprint, (
        "pipeline_fingerprint() no longer matches reports/runs/holdout-0cf5e0821ec7.json "
        "(v1.1.0-rc2) — the sealed holdout evidence is void. Either revert whatever changed "
        "the fingerprint, or run a new sealed 30-item holdout and update the canonical run."
    )
