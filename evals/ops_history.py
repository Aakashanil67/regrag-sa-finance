"""Pure preparation helpers for the evaluation-history section of the ops dashboard."""

from __future__ import annotations

import pandas as pd

_METRICS = ("answer_rate", "refusal_recall", "task_outcome", "retrieval_all_hit")


def prepare_eval_history(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return display data and the numeric metric columns present in the run history.

    Identifier columns are never inferred as chart series. The scope label keeps development
    and test runs from being read as one population.
    """
    display = frame.copy()
    if "split" in display:
        display["evidence_scope"] = display["split"].map(
            lambda value: "test" if value == "test" else "development"
        )
    else:
        display["evidence_scope"] = "unknown"
    return display, [name for name in _METRICS if name in display.columns]
