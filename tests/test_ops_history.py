import pandas as pd

from evals.ops_history import prepare_eval_history


def test_ops_history_charts_only_metric_columns_and_labels_scope():
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-09-24T00:00:00Z",
                "run_id": "dev-1",
                "split": "dev",
                "label": "baseline",
                "n": 60,
                "answer_rate": 0.5,
                "refusal_recall": 1.0,
                "task_outcome": 0.6,
                "retrieval_all_hit": 0.7,
                "cost_usd": 0.05,
            },
            {"split": "test", "run_id": "test-1", "answer_rate": 0.6},
        ]
    )

    display, metric_cols = prepare_eval_history(frame)

    assert metric_cols == ["answer_rate", "refusal_recall", "task_outcome", "retrieval_all_hit"]
    assert list(display["evidence_scope"]) == ["development", "test"]


def test_ops_history_skips_missing_metric_columns():
    display, metric_cols = prepare_eval_history(pd.DataFrame([{"answer_rate": 0.5}]))

    assert metric_cols == ["answer_rate"]
    assert display.loc[0, "evidence_scope"] == "unknown"
