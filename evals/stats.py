"""Pure statistics helpers, stdlib-only, no project imports. Kept separate from
the modules that import src.retrieve: those pull in chromadb and sentence-transformers, and
callers here compute confidence intervals from a saved run artifact without a vector store.
"""

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for a binomial proportion, 95% by default.
    Unlike the normal approximation, it stays within 0 and 1 for a 30-item holdout."""
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)
