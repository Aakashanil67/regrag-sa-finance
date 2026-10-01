"""Measures the actual latency and cost impact of src/cache.py's response cache.

Runs the same 15 questions through obslog.timed_answer() twice: once cold (cache empty, every
call hits real retrieval + the live API) and once warm (every call is a cache hit). p50/p95
latency and mean cost per query, cold vs warm, go to reports/perf.md.

    python -m scripts.perf_bench
"""

import json

from src.config import CACHE_DB_PATH, QUESTIONS_DEV_PATH, REPORTS_DIR
from src.obslog import timed_answer

N_QUESTIONS = 15


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def _run_pass(questions: list[str]) -> dict:
    latencies = []
    costs = []
    cache_hits = 0
    for question in questions:
        timed = timed_answer(question)
        latencies.append(timed.latency_ms)
        costs.append(timed.result.llm_response.cost_usd)
        cache_hits += int(timed.cache_hit)
    return {
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "mean_cost_usd": sum(costs) / len(costs),
        "cache_hits": cache_hits,
        "n": len(questions),
    }


def run() -> dict:
    golden = [
        json.loads(line) for line in QUESTIONS_DEV_PATH.read_text(encoding="utf-8").splitlines()
    ]
    factual = [item for item in golden if item["type"] == "single"][:N_QUESTIONS]
    questions = [item["question"] for item in factual]

    if CACHE_DB_PATH.exists():
        CACHE_DB_PATH.unlink()  # start cold, a warm cache would silently skip the "cold" pass

    cold = _run_pass(questions)
    warm = _run_pass(questions)
    return {"cold": cold, "warm": warm, "n": len(questions)}


def write_report(results: dict) -> None:
    cold, warm = results["cold"], results["warm"]
    lines = [
        "# Performance: response cache impact",
        "",
        f"{results['n']} factual questions, run once with an empty cache and once with a warm "
        "cache via `obslog.timed_answer()`. The cold pass calls retrieval and the model. "
        "The table records the cache hits in each pass.",
        "",
        "| | p50 latency | p95 latency | mean cost/query | cache hits |",
        "|---|---|---|---|---|",
        f"| cold | {cold['p50_ms']:.0f} ms | {cold['p95_ms']:.0f} ms | "
        f"${cold['mean_cost_usd']:.5f} | {cold['cache_hits']}/{cold['n']} |",
        f"| warm | {warm['p50_ms']:.0f} ms | {warm['p95_ms']:.0f} ms | "
        f"${warm['mean_cost_usd']:.5f} | {warm['cache_hits']}/{warm['n']} |",
        "",
        "A cache hit skips retrieval and generation. `RAGResult.retrieved_chunks` is empty, "
        "so the API's retrieval debug view has no chunks to show.",
        "",
        "The cache matches normalised question text (`src/cache.py`). A rephrased question "
        "is a miss. Pipeline settings also form part of the cache key.",
    ]
    (REPORTS_DIR / "perf.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(
        f"wrote reports/perf.md — cold p50={cold['p50_ms']:.0f}ms warm p50={warm['p50_ms']:.0f}ms"
    )


if __name__ == "__main__":
    write_report(run())
