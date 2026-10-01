"""evals/stats.py, pure statistics helpers with no project dependencies (stdlib math only), so
anything that needs a confidence interval (the retrieval benchmark, the release summary renderer)
can import it without pulling in retrieval_bench's own module-level chromadb/sentence-transformers
import chain."""

from evals.stats import wilson_interval


def test_wilson_interval_is_narrower_with_more_trials():
    small_lo, small_hi = wilson_interval(9, 10)
    large_lo, large_hi = wilson_interval(90, 100)
    assert (large_hi - large_lo) < (small_hi - small_lo)


def test_wilson_interval_bounds_are_within_zero_and_one():
    lo, hi = wilson_interval(3, 30)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_on_a_perfect_rate_still_has_a_nonzero_margin():
    # 30/30 (100%) is an observation, not a guarantee the true rate is exactly 1.0, a small-n
    # holdout with a perfect score should still show real uncertainty, not collapse to (1.0, 1.0)
    lo, hi = wilson_interval(30, 30)
    assert lo < 1.0
    assert hi == 1.0
