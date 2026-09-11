"""Release evidence must report hit-rate as a fraction with a confidence interval, not just a bare
percentage — a bare 93% looks identical whether it came from 28/30 or 2790/3000, and the plan's
release gate depends on readers being able to tell those apart. The Wilson interval math itself
lives in evals/stats.py and is tested there; this file only covers write_report's formatting."""

from evals.retrieval_bench import write_report


def test_report_includes_fractions_and_wilson_interval(tmp_path, monkeypatch):
    import evals.retrieval_bench as bench

    monkeypatch.setattr(bench, "REPORTS_DIR", tmp_path)
    results = {
        "per_question": [
            {"id": "q1", "question": "a?", "doc_id": "d1", "page": 1, "first_hit_rank": 1},
            {"id": "q2", "question": "b?", "doc_id": "d2", "page": 2, "first_hit_rank": None},
        ],
        "hit_rates": {3: 0.5, 5: 0.5, 10: 0.5},
        "mrr": 0.5,
    }
    write_report(results, split="holdout")
    text = (tmp_path / "retrieval_bench.md").read_text(encoding="utf-8")
    assert "1/2" in text
    assert "95% CI" in text


def test_report_shows_a_dash_not_a_crash_for_an_empty_question_set(tmp_path, monkeypatch):
    import evals.retrieval_bench as bench

    monkeypatch.setattr(bench, "REPORTS_DIR", tmp_path)
    results = {"per_question": [], "hit_rates": {3: 0.0, 5: 0.0, 10: 0.0}, "mrr": 0.0}

    write_report(results, split="dev")

    text = (tmp_path / "retrieval_bench.md").read_text(encoding="utf-8")
    assert "0/0" in text
    assert "—" in text
