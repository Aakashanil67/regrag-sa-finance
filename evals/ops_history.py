"""Pure preparation helpers for the evaluation-history section of the ops dashboard."""

from __future__ import annotations

import pandas as pd

_LEGACY_METRICS = (
    "answerable_answer_rate",
    "unanswerable_refusal_recall",
    "citation_contract_pass_rate",
    "verified_citation_rate",
)
_V2_METRICS = (
    "answer_coverage",
    "refusal_recall",
    "citation_verification",
    "structural_pass_rate",
    "task_outcome",
)


def prepare_eval_history(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return display data and an explicit numeric metric list for either CSV schema.

    Identifier columns are intentionally never inferred as chart series. The returned display
    frame adds a human-readable scope label so development and holdout/release observations cannot
    be mistaken for one combined population.
    """
    display = frame.copy()
    schema_version = (
        int(display["schema_version"].iloc[0])
        if "schema_version" in display and not display.empty
        else 1
    )
    if "split" in display:
        display["evidence_scope"] = display["split"].map(
            lambda value: "development" if value == "dev" else "release/holdout"
        )
    else:
        display["evidence_scope"] = "unknown"

    if schema_version >= 2:
        metric_cols = [name for name in _V2_METRICS if name in display.columns]
    else:
        metric_cols = [name for name in _LEGACY_METRICS if name in display.columns]
    return display, metric_cols
