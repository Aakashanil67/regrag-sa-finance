"""Paired before/after comparison for the source-type/currency metadata fix, same discipline as
scripts/paired_eval.py applied to the chunk-size/reranking change: don't compare two means over
two different item sets and call it a result.

This comparison's answer is different in shape from the chunk-size one, though. There the paired
RAGAS deltas were mostly noise and the real, defensible win was retrieval coverage (refusals
converted to answers). Here even that's mostly absent, the paired RAGAS deltas are smaller still,
and the one refusal that changed direction (g45) isn't a capability loss introduced by this fix: the
cross-encoder reranker demotes the National Credit Act's own text below four NCR guideline chunks
for that specific query (checked directly against src.retrieve.retrieve, the Act's stated-purpose
chunk ranks 2nd by bi-encoder score but 6th after reranking, outside the k=5 the pipeline uses), a
pre-existing weakness the retrieval code doesn't touch. Before this fix, the model
answered anyway from a DTIC brochure describing the Act, treating its summary as equivalent to "the
Act states X." After, told explicitly that source is a "Regulator explainer brochure," it correctly
declines to attribute a claim to the Act's own text when that text was never retrieved. The fix
didn't create the reranker gap. It stopped the model from papering over it.

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
        "Same method as `paired_comparison.md`, applied to the source-authority and "
        "currency metadata fix. The context gained document type, issuer and title, "
        "and the result gained status and third-party notices (`src/rag.py`). Before: chunk_size=800+rerank with "
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
        "The paired deltas are small and none of the sign tests establishes a reliable change. "
        "The metadata labels source authority and currency. These RAGAS metrics do not directly "
        "measure whether an answer identifies the kind of source it cites.",
        "",
        "`g24` changed from refusal to a correct, cited answer. The model had not been shown "
        "the document's title, so it could not identify Guideline 004/2025. Adding the title "
        "copied from the PDF cover fixed this case. Paraphrased titles had failed in the "
        "first attempt.",
        "",
        "`g45` changed from an answer to a refusal. In the recorded retrieval check, "
        "`nca_act_34_2005` p.1-2 ranked 2nd before reranking, inside the top-5, but 6th "
        "after reranking, outside k=5. The reranker put four NCR guideline chunks ahead of "
        "the Act's purpose section. The retrieval code did not change with the metadata fix. "
        "Before the fix, the model answered from a brochure. After being told that the source "
        "was a regulator explainer, it refused to attribute the claim to the Act. Source "
        "labelling exposed this retrieval gap.",
        "",
        "Separate manual checks found that the PwC IFRS 9 guide attached a third-party notice "
        "and either 2004 SARB Circular attached an instrument-status notice. Both were "
        "structured `source_notices`, separate from the graded answer. These checks used "
        "the historical source classifications, which were later corrected.",
    ]

    (REPORTS_DIR / "paired_comparison_metadata.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote reports/paired_comparison_metadata.md — entered {r['entered']}, left {r['left']}")


if __name__ == "__main__":
    write_report(compare())
