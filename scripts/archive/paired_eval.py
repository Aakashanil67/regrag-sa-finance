"""Paired before/after comparison of the two RAGAS runs, plus intervals on the retrieval benchmark.

Why this exists: `reports/eval_summary.md` reports a mean over whichever items scored on that run,
and the two runs didn't score the same items. The pre-improvement run scored 34; the post run
scored 39, because reranking pulled six previously-refused questions into an actual answer (and
pushed one the other way). Comparing those two means directly conflates two different things —
a change in answer quality on a fixed set of questions, and a change in *which* questions are in
the average at all. The second effect is real and worth reporting, but it isn't a quality gain.

So this reports three things instead of one:
  1. the raw means over each full run, which is what a naive before/after table shows;
  2. the same metrics restricted to the 33 items both runs scored, which is the paired comparison
     that actually isolates quality;
  3. a per-item win/loss count with a two-sided sign test, because a mean can drift on noise while
     the underlying items split evenly — which is exactly what happens to faithfulness here.

Retrieval hit-rate gets Wilson score intervals for the same reason: 85% -> 95% on a 20-question
benchmark is a difference of two questions, and the interval makes the weight of that obvious in
a way the point estimate does not.

    python -m scripts.paired_eval
"""

import json
import math
import re
import statistics

from src.config import EVALS_DIR, REPORTS_DIR

BASELINE_PATH = EVALS_DIR / "baselines" / "ragas_500_norerank.json"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# from reports/retrieval_bench.md, before and after the chunk-size + reranking change
RETRIEVAL_N = 20
RETRIEVAL_HITS_BEFORE = 17  # 85%
RETRIEVAL_HITS_AFTER = 19  # 95%

_ROW = re.compile(
    r"^\|\s*(g\d+)\s*\|.*?\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*$"
)


def _parse_summary(path) -> dict[str, dict[str, float]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if m:
            scores = (float(m.group(i)) for i in range(2, 6))
            rows[m.group(1)] = dict(zip(METRICS, scores, strict=True))
    return rows


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — holds up at small n and near the boundary, where the textbook
    normal-approximation interval on 19/20 would run past 100%."""
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign test on the items that moved. Ties carry no directional
    information and are excluded, which is the standard treatment."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare() -> dict:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    before = baseline["per_item"]
    after = _parse_summary(REPORTS_DIR / "eval_summary.md")
    shared = sorted(set(before) & set(after))

    per_metric = {}
    for metric in METRICS:
        wins = sum(1 for k in shared if after[k][metric] > before[k][metric])
        losses = sum(1 for k in shared if after[k][metric] < before[k][metric])
        per_metric[metric] = {
            "before_all": statistics.mean(v[metric] for v in before.values()),
            "after_all": statistics.mean(v[metric] for v in after.values()),
            "before_paired": statistics.mean(before[k][metric] for k in shared),
            "after_paired": statistics.mean(after[k][metric] for k in shared),
            "wins": wins,
            "losses": losses,
            "ties": len(shared) - wins - losses,
            "p_value": _sign_test_p(wins, losses),
        }

    return {
        "n_before": len(before),
        "n_after": len(after),
        "n_shared": len(shared),
        "entered": sorted(set(after) - set(before)),
        "left": sorted(set(before) - set(after)),
        "per_metric": per_metric,
        "retrieval": {
            "before": (RETRIEVAL_HITS_BEFORE, _wilson(RETRIEVAL_HITS_BEFORE, RETRIEVAL_N)),
            "after": (RETRIEVAL_HITS_AFTER, _wilson(RETRIEVAL_HITS_AFTER, RETRIEVAL_N)),
        },
    }


def write_report(r: dict) -> None:
    pm = r["per_metric"]
    before_hits, before_ci = r["retrieval"]["before"]
    after_hits, after_ci = r["retrieval"]["after"]

    lines = [
        "# Is the improvement real? A paired re-analysis",
        "",
        "The headline before/after table in the README compares the mean of one RAGAS run against "
        "the mean of another. Those two runs did not score the same questions: the pre-improvement "
        f"run scored {r['n_before']} items and the post-improvement run scored {r['n_after']}, "
        f"overlapping on {r['n_shared']}. Reranking pulled {len(r['entered'])} previously-refused "
        f"questions ({', '.join(f'`{i}`' for i in r['entered'])}) into a real answer, and pushed "
        f"{len(r['left'])} the other way ({', '.join(f'`{i}`' for i in r['left'])}).",
        "",
        "That matters, because a mean over a changing item set moves for two unrelated reasons: "
        "the answers got better, or the mix of questions being averaged changed. Only the first "
        "is a quality claim. This report separates them.",
        "",
        "## Paired comparison, on the 33 items both runs scored",
        "",
        "| metric | before (all) | after (all) | before (paired) | after (paired) | paired Δ | improved | regressed | tied | sign test |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for metric in METRICS:
        m = pm[metric]
        delta = m["after_paired"] - m["before_paired"]
        lines.append(
            f"| {metric.replace('_', ' ')} | {m['before_all']:.3f} | {m['after_all']:.3f} | "
            f"{m['before_paired']:.3f} | {m['after_paired']:.3f} | {delta:+.3f} | "
            f"{m['wins']} | {m['losses']} | {m['ties']} | p = {m['p_value']:.2f} |"
        )

    faith, prec, rec, rel = (
        pm["faithfulness"],
        pm["context_precision"],
        pm["context_recall"],
        pm["answer_relevancy"],
    )
    lines += [
        "",
        "### What survives",
        "",
        f"**Context precision is the strongest result, though not a significant one at this "
        f"sample size.** It gains {prec['after_paired'] - prec['before_paired']:+.3f} on the paired "
        f"set, with {prec['wins']} items improving against {prec['losses']} regressing — a sign "
        f"test p of {prec['p_value']:.2f}, which falls just outside the conventional 0.05 "
        "threshold and would not clear peer review on its own. What makes me treat it as real "
        "rather than lucky is that it is the only metric where the effect has a mechanism behind "
        "it: dropping chunks the bi-encoder surfaced on loose topical similarity is precisely and "
        "solely what a cross-encoder does, so a precision-shaped gain is the prediction, not a "
        "post-hoc reading of whichever number happened to move. Thirty-three items is too few to "
        "confirm it; a larger golden set is the fix, and it isn't built.",
        "",
        f"**Faithfulness does not survive.** The headline gain of "
        f"{faith['after_all'] - faith['before_all']:+.3f} shrinks to "
        f"{faith['after_paired'] - faith['before_paired']:+.3f} once the item set is held fixed, and the "
        f"per-item split is {faith['wins']} improved against {faith['losses']} regressed — a coin "
        f"flip (p = {faith['p_value']:.2f}). Most of the apparent gain was composition: the six "
        "questions that entered the average happened to score above the old mean. I am not "
        "claiming reranking improved faithfulness.",
        "",
        f"**Context recall was already at the ceiling.** It reads "
        f"{rec['after_all'] - rec['before_all']:+.3f} across the full runs but only "
        f"{rec['after_paired'] - rec['before_paired']:+.3f} paired, with {rec['ties']} of "
        f"{r['n_shared']} items completely unchanged and exactly {rec['wins']} item moving. At a "
        f"paired baseline of {rec['before_paired']:.3f} there was almost nothing left to win.",
        "",
        f"**Answer relevancy drifts slightly negative** ({rel['after_paired'] - rel['before_paired']:+.3f} "
        f"paired, {rel['wins']} improved against {rel['losses']} regressed, p = {rel['p_value']:.2f}). "
        "Also indistinguishable from noise, and reported rather than dropped.",
        "",
        "## Retrieval benchmark: 20 questions is a small ruler",
        "",
        "Hit-rate@5 went from 85% to 95%. In absolute terms that is "
        f"{before_hits}/{RETRIEVAL_N} to {after_hits}/{RETRIEVAL_N} — **two questions**. Wilson "
        "score intervals:",
        "",
        "| | hit-rate@5 | 95% CI |",
        "|---|---|---|",
        f"| before | {before_hits}/{RETRIEVAL_N} ({before_hits / RETRIEVAL_N:.0%}) | "
        f"{before_ci[0]:.0%} – {before_ci[1]:.0%} |",
        f"| after | {after_hits}/{RETRIEVAL_N} ({after_hits / RETRIEVAL_N:.0%}) | "
        f"{after_ci[0]:.0%} – {after_ci[1]:.0%} |",
        "",
        "The intervals overlap heavily. On this benchmark alone the hit-rate difference is not "
        "separable from sampling noise, and a 20-question set cannot resolve a two-question gap. "
        "MRR (0.654 → 0.808) is the better-powered signal in the same data, because it moves on "
        "*where* the correct chunk ranks rather than only whether it cleared a cutoff, so a "
        "question going from rank 4 to rank 1 registers instead of being scored identically.",
        "",
        "## What I would say about this config, honestly",
        "",
        "Reranking at 800-token chunks earns its place on one measured effect and one unambiguous "
        "one. The measured effect is context precision, directionally clear and mechanistically "
        "expected but short of significance on 33 items. The unambiguous one is coverage: five net "
        "refusals became answered questions, and that needs no significance test, because it is a "
        "count of behaviour changing, not an estimate of a mean. It did not demonstrably improve "
        "faithfulness or recall, and the retrieval hit-rate gain sits inside the noise floor of a "
        "20-question benchmark. The costs are equally concrete — a cross-encoder pass over 20 "
        "candidates on every query, and a coarser citation, since an 800-token chunk spans more "
        "pages than a 500-token one and the citation inherits that span.",
        "",
        "The honest summary is a coverage win with a probable precision win attached, not an "
        "across-the-board improvement. The README says that rather than the four-green-arrows "
        "version, which the raw means would have supported and the paired data does not.",
    ]

    (REPORTS_DIR / "paired_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"wrote reports/paired_comparison.md — paired on {r['n_shared']} items; "
        f"precision {prec['after_paired'] - prec['before_paired']:+.3f} (p={prec['p_value']:.2f}), "
        f"faithfulness {faith['after_paired'] - faith['before_paired']:+.3f} (p={faith['p_value']:.2f})"
    )


if __name__ == "__main__":
    write_report(compare())
