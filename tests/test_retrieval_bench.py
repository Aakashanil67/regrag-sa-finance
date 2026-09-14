"""Release evidence must report hit-rate as a fraction with a confidence interval, not just a bare
percentage — a bare 93% looks identical whether it came from 28/30 or 2790/3000, and the plan's
release gate depends on readers being able to tell those apart. The Wilson interval math itself
lives in evals/stats.py and is tested there; this file only covers write_report's formatting."""

from evals.retrieval_bench import source_coverage, write_report
from src.retrieve import RetrievedChunk


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


def test_multi_document_coverage_can_pass_any_source_but_fail_all_sources():
    results = [
        RetrievedChunk("a", "doc-a", "a", 1, 1, "", 0.9),
        RetrievedChunk("other", "doc-c", "c", 1, 1, "", 0.8),
    ]

    coverage = source_coverage(
        results, [{"doc_id": "doc-a", "page": 1}, {"doc_id": "doc-b", "page": 2}]
    )

    assert coverage["any_required_source_hit"] is True
    assert coverage["all_required_source_hit"] is False
    assert coverage["first_relevant_rank"] == 1


def test_comparison_report_contains_each_variant_and_question(tmp_path):
    from scripts.compare_retrieval import render_comparison_markdown

    result = {
        "selection": {"selected_variant": "A", "reason": "synthetic"},
        "variants": [
            {
                "label": label,
                "chunk_target_tokens": 240,
                "chunk_overlap_tokens": 32,
                "tokenizer_mode": "synthetic",
                "strategy": "semantic",
                "summary": {
                    "generation_any": 0,
                    "generation_all": 0,
                    "retrieval_any": 0,
                    "retrieval_all": 0,
                    "multi_document_all_count": 0,
                    "multi_document_count": 0,
                },
                "warm_latency_ms": {"p95": 1.0},
                "chunk_count": 1,
                "embedding_build_seconds": 0.1,
                "qualifies": False,
                "per_item": [
                    {
                        "set": "golden_dev",
                        "id": "g1",
                        "required_sources": [],
                        "any_required_source_hit": False,
                        "all_required_source_hit": False,
                        "first_relevant_rank": None,
                        "retrieved": [],
                        "embedding_truncated_retrieved": 0,
                        "reranker_truncated_retrieved": 0,
                    },
                    {
                        "set": "retrieval_dev",
                        "id": "r1",
                        "required_sources": [],
                        "any_required_source_hit": False,
                        "all_required_source_hit": False,
                        "first_relevant_rank": None,
                        "retrieved": [],
                        "embedding_truncated_retrieved": 0,
                        "reranker_truncated_retrieved": 0,
                    },
                ],
            }
            for label in ("A", "B", "C")
        ],
    }

    text = render_comparison_markdown(result)

    assert all(label in text for label in ("A", "B", "C"))
    assert "g1" in text and "r1" in text
