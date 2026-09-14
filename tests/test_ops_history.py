import pandas as pd

from evals.ops_history import prepare_eval_history


def test_ops_history_uses_only_explicit_legacy_numeric_metric_columns():
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-09-10T00:00:00Z",
                "run_id": "dev-1",
                "split": "dev",
                "label": "baseline",
                "answerable_answer_rate": 0.5,
                "unanswerable_refusal_recall": 1.0,
                "citation_contract_pass_rate": 0.8,
                "verified_citation_rate": 1.0,
            }
        ]
    )

    display, metric_cols = prepare_eval_history(frame)

    assert metric_cols == [
        "answerable_answer_rate",
        "unanswerable_refusal_recall",
        "citation_contract_pass_rate",
        "verified_citation_rate",
    ]
    assert "run_id" not in metric_cols
    assert "label" not in metric_cols
    assert "split" not in metric_cols
    assert display.loc[0, "evidence_scope"] == "development"


def test_ops_history_uses_v2_rates_and_marks_release_scope():
    frame = pd.DataFrame(
        [
            {
                "schema_version": 2,
                "timestamp": "2026-09-10T00:00:00Z",
                "run_id": "holdout-1",
                "split": "holdout",
                "label": "candidate",
                "answer_coverage_numerator": 1,
                "answer_coverage_denominator": 2,
                "answer_coverage": 0.5,
                "refusal_recall_numerator": 1,
                "refusal_recall_denominator": 1,
                "refusal_recall": 1.0,
                "citation_verification_numerator": 2,
                "citation_verification_denominator": 2,
                "citation_verification": 1.0,
                "structural_pass_rate_numerator": 1,
                "structural_pass_rate_denominator": 1,
                "structural_pass_rate": 1.0,
                "task_outcome_numerator": 1,
                "task_outcome_denominator": 2,
                "task_outcome": 0.5,
            }
        ]
    )

    display, metric_cols = prepare_eval_history(frame)

    assert metric_cols == [
        "answer_coverage",
        "refusal_recall",
        "citation_verification",
        "structural_pass_rate",
        "task_outcome",
    ]
    assert "run_id" not in metric_cols
    assert "label" not in metric_cols
    assert "split" not in metric_cols
    assert display.loc[0, "evidence_scope"] == "release/holdout"
