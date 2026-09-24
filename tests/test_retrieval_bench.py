from evals.retrieval_bench import first_hit_rank, render

ITEM = {"evidence": [{"doc_id": "a", "page": 4}]}


def test_first_hit_rank_finds_second_context():
    ctx = [
        {"doc_id": "b", "page_start": 4, "page_end": 4},
        {"doc_id": "a", "page_start": 3, "page_end": 5},
    ]
    assert first_hit_rank(ITEM, ctx) == 2


def test_first_hit_rank_none_when_absent():
    assert first_hit_rank(ITEM, [{"doc_id": "a", "page_start": 6, "page_end": 7}]) is None


def test_render_has_one_row_per_strategy_and_rerank():
    rows = [
        {"strategy": s, "rerank": r, "n": 10, "any_hit": 9, "all_hit": 8, "multi_all_hit": 2,
         "multi_n": 3, "mrr": 0.5, "p50_ms": 12}
        for s in ("semantic", "bm25") for r in (False, True)
    ]
    out = render([{"config": "x", "chunks": 100, "truncated_chunks": 1, "rows": rows}], "dev")
    assert sum(1 for line in out.splitlines() if line.startswith("| x |")) == 4
