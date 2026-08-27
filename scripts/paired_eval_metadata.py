"""Paired before/after comparison for the source-type/currency metadata fix, same discipline as
scripts/paired_eval.py applied to the chunk-size/reranking change: don't compare two means over
two different item sets and call it a result.

This comparison's answer is different in shape from the chunk-size one, though. There the paired
RAGAS deltas were mostly noise and the real, defensible win was retrieval coverage (refusals
converted to answers). Here even that's mostly absent — the paired RAGAS deltas are smaller still,
and the one refusal that changed direction (g45) isn't a capability loss at all: retrieval for that
question returns the exact same five chunks before and after this change (verified directly against
src.retrieve.retrieve, not inferred), none of which is the National Credit Act's own text — only a
DTIC brochure *about* it. Before this fix, the model treated the brochure's summary as equivalent to
"the Act states X." After, told explicitly that source is a "Regulator explainer brochure," it
correctly refused to claim the Act's own text says something it never saw. That's the fix working
as designed, showing up as a debit in the crude refused/answered ledger.

    python -m scripts.paired_eval_metadata
"""

import json
import math
import re
import statistics

from src.config import EVALS_DIR, REPORTS_DIR

BASELINE_PATH = EVALS_DIR / "baselines" / "ragas_800_rerank_no_metadata.json"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

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


def _sign_test_p(wins: int, losses: int) -> float:
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
    }


def write_report(r: dict) -> None:
    pm = r["per_metric"]
    lines = [
        "# Source-type/currency metadata: paired re-analysis",
        "",
        "Same method as `reports/paired_comparison.md`, applied to the source-authority and "
        "currency metadata fix (document type, issuer, title, and dated-instrument/third-party "
        "notices — see the module docstring in `src/rag.py`). Before: chunk_size=800+rerank with "
        "no source metadata in the prompt. After: the same config with source-type lines added to "
        "context and `source_notices` added to the result.",
        "",
        f"{len(r['entered'])} item entered the scored set that was previously refused "
        f"({', '.join(f'`{i}`' for i in r['entered']) or 'none'}), and "
        f"{len(r['left'])} item newly refused ({', '.join(f'`{i}`' for i in r['left']) or 'none'}).",
        "",
        "## Paired comparison",
        "",
        "| metric | before (paired) | after (paired) | Δ | improved | regressed | tied | sign test |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for metric in METRICS:
        m = pm[metric]
        delta = m["after_paired"] - m["before_paired"]
        lines.append(
            f"| {metric.replace('_', ' ')} | {m['before_paired']:.3f} | {m['after_paired']:.3f} | "
            f"{delta:+.3f} | {m['wins']} | {m['losses']} | {m['ties']} | p = {m['p_value']:.2f} |"
        )

    lines += [
        "",
        "## What this fix actually did",
        "",
        "Nothing here moves a RAGAS mean by more than noise — every paired delta is small, every "
        "sign test is indistinguishable from a coin flip, and that's the honest result. This fix "
        "was never a retrieval or generation-quality change; it's a disclosure change, and RAGAS's "
        "four metrics don't have a dimension for \"did the answer correctly attribute what kind of "
        "document this is.\" The two changes that matter aren't visible in this table:",
        "",
        "**`g24` went from refusal to a correct, cited answer.** This is the specific failure "
        "`reports/failure_analysis.md` diagnosed: the model couldn't tell which retrieved document "
        'was actually "Guideline 004/2025" because no guideline\'s own number reached the prompt '
        "anywhere. Adding each document's real title (extracted from the PDF's own cover page, not "
        "paraphrased — the first attempt at this fix used paraphrased titles and didn't work, "
        "which is how the paraphrasing was caught) fixed it directly.",
        "",
        "**`g45` went the other way, and it's not a capability loss.** Retrieval for that question "
        "returns the same five chunks before and after this change — verified directly against "
        "`src.retrieve.retrieve`, not inferred from the score — and none of them is the National "
        "Credit Act's own text, only a DTIC brochure describing it. Before this fix, the model "
        "answered anyway, treating the brochure's summary as equivalent to the Act's own words. "
        'After, told explicitly that the source is a "Regulator explainer brochure," it correctly '
        'refused to attribute a claim to "the Act itself" when the Act\'s own text was never in '
        "front of it. That's the fix working as designed. It reads as a regression only if refusal "
        "count is the metric, which is exactly the kind of thing a metric-only readout misses.",
        "",
        "Live-verified separately (not part of this batch, checked by hand against the model's "
        "actual output): citing the PwC IFRS 9 guide now attaches a third-party-source notice, and "
        "citing either 2004 SARB Circular attaches a superseded-instrument-type notice — both as "
        "structured `source_notices`, not text folded into the graded answer.",
    ]

    (REPORTS_DIR / "paired_comparison_metadata.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote reports/paired_comparison_metadata.md — entered {r['entered']}, left {r['left']}")


if __name__ == "__main__":
    write_report(compare())
