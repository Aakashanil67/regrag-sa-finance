"""Pure statistics helpers, stdlib-only — no project imports. Kept separate from
evals/retrieval_bench.py deliberately: that module imports src.retrieve at module scope, which
pulls in chromadb and sentence-transformers, and evals/render_summary.py needs to compute
confidence intervals from a saved run artifact without a vector store anywhere in reach.
"""

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95%-by-default Wilson score interval for a binomial proportion — the small-n-appropriate
    alternative to a normal-approximation interval, which can extend past 0 or 1 exactly where a
    30-item holdout needs it most."""
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)
