"""One entry point for a complete development or holdout evaluation run.

This is the only code path allowed to update the canonical `reports/eval_summary.md`,
`reports/eval_history.csv`, `reports/retrieval_bench.md`, and `reports/failure_analysis.md`. A run
that hits any generation or judge failure is `partial`: it still writes a timestamped artefact
under `reports/runs/` (so the failure and everything that *did* succeed aren't lost), but it exits
non-zero and never touches the canonical reports or history — a retry starts a fresh run rather
than silently completing a stale one.

Metrics are computed with explicit denominators throughout, because an item can fail to reach a
metric in more than one way (refused vs. failed to generate vs. failed to judge), and folding those
together produces a rate that looks clean but no longer means what its name says.

    python -m evals.run_release --split dev --label corrected-corpus-baseline
    python -m evals.run_release --split holdout --label v1.1.0-rc1
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from evals.stats import wilson_interval
from src.config import (
    EVAL_HISTORY_CSV,
    EVAL_PROTOCOL_PATH,
    GOLDEN_DEV_PATH,
    GOLDEN_HOLDOUT_PATH,
    REPORTS_DIR,
    RETRIEVAL_DEV_PATH,
    RETRIEVAL_HOLDOUT_PATH,
)
from src.provenance import manifest_digest, pipeline_fingerprint

_GOLDEN_PATHS = {"dev": GOLDEN_DEV_PATH, "holdout": GOLDEN_HOLDOUT_PATH}
_RETRIEVAL_PATHS = {"dev": RETRIEVAL_DEV_PATH, "holdout": RETRIEVAL_HOLDOUT_PATH}
_SCORED_TYPES = {"factual", "multi-doc"}
_RAGAS_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

RUNS_DIR = REPORTS_DIR / "runs"
EVAL_HISTORY_V2_CSV = REPORTS_DIR / "eval_history_v2.csv"


class HoldoutNotSealedError(RuntimeError):
    """Raised when --split holdout is requested but evals/protocol.json isn't sealed."""


def _load_golden(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def assert_split_runnable(split: str) -> None:
    if split != "holdout":
        return
    protocol = json.loads(EVAL_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not protocol.get("sealed"):
        raise HoldoutNotSealedError(
            "evals/protocol.json is not sealed — the holdout cannot be run until it is"
        )


def run_item(item: dict, answer_fn) -> dict:
    """Generate for one golden item and classify it structurally. `answer_fn` takes a question and
    returns a RAGResult-shaped object; it's an injected dependency so this function (and the
    metrics built on it) can be tested against a fake pipeline instead of the real LLM/vector
    store.

    `raw_model_output` records the pre-validation text unconditionally — unlike src/obslog.py's
    same-named opt-in column, an eval artifact already stores the full question and answer for
    every item with no privacy gate, since it's evaluation output, not a live user log. Recording
    the raw text alongside the validated one is what makes a refused item's failure mode (a
    citation-format near-miss vs. a genuine hedge vs. a clean model_refusal) auditable from the
    artifact instead of requiring another live API call."""
    result = answer_fn(item["question"])
    citations = [
        {"doc_id": c.doc_id, "page": c.page, "verified": c.verified} for c in result.citations
    ]
    # an unanswerable item passes the citation contract by refusing; an answerable item passes by
    # answering with a citation contract validate_generated_answer already enforced upstream
    citation_contract_pass = item["type"] == "unanswerable" if result.refused else True
    source_notices = [
        {
            "kind": n.kind,
            "text": n.text,
            "evidence": [{"doc_id": e.doc_id, "page": e.page} for e in n.evidence],
        }
        for n in result.source_notices
    ]
    return {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "answer": result.answer,
        # Keep the raw model output and the answer actually served by the validator distinct. A
        # refusal explanation is an artifact field, not a served answer for citation metrics.
        "served_answer": result.answer if not result.refused else None,
        "raw_model_output": result.llm_response.text or None,
        "refused": result.refused,
        "refusal_reason": result.refusal_reason.value if result.refusal_reason else None,
        "reference_answerable": item["type"] != "unanswerable",
        "reference_answerability": (
            "answerable" if item["type"] != "unanswerable" else "unanswerable"
        ),
        "citations": citations,
        "retrieved_chunk_ids": [c.chunk_id for c in result.retrieved_chunks],
        "citation_contract_pass": citation_contract_pass,
        "structural_validator_pass": bool(not result.refused),
        "structural_validator_outcome": "passed" if not result.refused else "refused",
        "all_citations_verified": all(c["verified"] for c in citations) if citations else True,
        "source_notices": source_notices,
    }


def compute_structural_metrics(items: list[dict]) -> dict:
    """Every rate here is computed over its own denominator, printed alongside it — see the module
    docstring for why folding these into one number would misrepresent what happened."""
    answerable = [i for i in items if i["type"] != "unanswerable"]
    unanswerable = [i for i in items if i["type"] == "unanswerable"]
    answerable_answered = [i for i in answerable if not i["refused"]]
    unanswerable_refused = [i for i in unanswerable if i["refused"]]
    contract_pass = [i for i in items if i["citation_contract_pass"]]
    verified = [i for i in items if i["all_citations_verified"]]

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "answerable_count": len(answerable),
        "answerable_answered_count": len(answerable_answered),
        "answerable_answer_rate": rate(len(answerable_answered), len(answerable)),
        "unanswerable_count": len(unanswerable),
        "unanswerable_refused_count": len(unanswerable_refused),
        "unanswerable_refusal_recall": rate(len(unanswerable_refused), len(unanswerable)),
        "citation_contract_pass_count": len(contract_pass),
        "citation_contract_pass_rate": rate(len(contract_pass), len(items)),
        "verified_citation_count": len(verified),
        "verified_citation_rate": rate(len(verified), len(items)),
    }


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def compute_metrics_v2(items: list[dict]) -> dict:
    """Compute the schema-2 structural metrics without reinterpreting schema-1 values.

    Citation occurrences are counted only in non-refused served answers. They are useful for
    validation, but are not independent item-level trials; the summary therefore does not attach
    Wilson intervals to this ratio. Semantic review fields remain explicitly unavailable until a
    real review dataset is joined.
    """

    def is_answerable(item: dict) -> bool:
        return item.get("reference_answerable", item.get("type") != "unanswerable")

    answerable = [item for item in items if is_answerable(item)]
    unanswerable = [item for item in items if not is_answerable(item)]

    def served(item: dict) -> bool:
        if item.get("refused", False):
            return False
        # Legacy-shaped synthetic rows have no explicit served_answer. New rows do, and a null
        # value means the validator did not serve an answer even if raw output was present.
        return item.get("served_answer", item.get("answer")) is not None

    served_items = [item for item in items if served(item)]
    served_citations = [citation for item in served_items for citation in item.get("citations", [])]
    verified_citations = [
        citation for citation in served_citations if citation.get("verified", False)
    ]
    structurally_valid = [
        item
        for item in served_items
        if item.get("structural_validator_pass", item.get("citation_contract_pass", False))
    ]
    answered_answerable = [item for item in answerable if served(item)]
    refused_unanswerable = [item for item in unanswerable if item.get("refused", False)]
    task_successes = [
        item
        for item in answerable
        if served(item)
        and item.get("structural_validator_pass", item.get("citation_contract_pass", False))
    ] + refused_unanswerable

    metrics = {
        "answer_coverage": _ratio(len(answered_answerable), len(answerable)),
        "refusal_recall": _ratio(len(refused_unanswerable), len(unanswerable)),
        "citation_verification": _ratio(len(verified_citations), len(served_citations)),
        "structural_pass_rate": _ratio(len(structurally_valid), len(served_items)),
        "task_outcome": _ratio(len(task_successes), len(items)),
        "reviewed_supported_answer_accuracy": None,
        "reviewed_supported_answer_yield": None,
    }
    # These counts make the denominators machine-readable for history consumers without requiring
    # them to infer meaning from a display label.
    metrics["answerable_count"] = len(answerable)
    metrics["answered_answerable_count"] = len(answered_answerable)
    metrics["unanswerable_count"] = len(unanswerable)
    metrics["refused_unanswerable_count"] = len(refused_unanswerable)
    metrics["served_answer_count"] = len(served_items)
    metrics["structurally_valid_served_answer_count"] = len(structurally_valid)
    metrics["citation_occurrence_count"] = len(served_citations)
    metrics["verified_citation_occurrence_count"] = len(verified_citations)
    metrics["task_success_count"] = len(task_successes)
    return metrics


async def score_ragas(
    items: list[dict], run_results: dict[str, dict], judge_fn
) -> tuple[list[dict], list[dict], list[dict]]:
    """RAGAS-scores every answered (not refused) factual/multi-doc item. `judge_fn` is
    `(question, answer, contexts, reference) -> awaitable[dict[str, float]]` — injected so tests
    exercise the denominator/failure bookkeeping without a real judge model. Returns
    (scored, excluded_refusals, failures)."""
    scored, excluded_refusals, failures = [], [], []
    for item in items:
        if item["type"] not in _SCORED_TYPES or item["id"] not in run_results:
            continue  # a generation failure already excludes this item; nothing to judge
        run_result = run_results[item["id"]]
        if run_result["refused"]:
            excluded_refusals.append({"id": item["id"], "question": item["question"]})
            continue
        try:
            metric_values = await judge_fn(
                item["question"],
                run_result["answer"],
                run_result["retrieved_chunk_ids"],
                item["reference_answer"],
            )
        except Exception as exc:  # noqa: BLE001 - a judge failure is data for the run, not a crash
            failures.append({"id": item["id"], "question": item["question"], "error": str(exc)})
            continue
        scored.append({"id": item["id"], "question": item["question"], **metric_values})
    return scored, excluded_refusals, failures


def _ragas_means(scored: list[dict]) -> dict[str, float] | None:
    if not scored:
        return None
    return {name: sum(row[name] for row in scored) / len(scored) for name in _RAGAS_METRIC_NAMES}


@dataclass
class ReleaseRun:
    run_id: str
    split: str
    label: str
    started_at: str
    finished_at: str
    status: str  # "complete" | "partial"
    holdout_sha256: str | None
    pipeline_fingerprint: str
    manifest_sha256: str
    structural_metrics: dict
    ragas_means: dict | None
    ragas_scored_count: int
    ragas_excluded_refusals: list[dict]
    ragas_failures: list[dict]
    items: list[dict]
    error: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict:
        payload = {
            "run_id": self.run_id,
            "split": self.split,
            "label": self.label,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "holdout_sha256": self.holdout_sha256,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "manifest_sha256": self.manifest_sha256,
            "structural_metrics": self.structural_metrics,
            "ragas_means": self.ragas_means,
            "ragas_scored_count": self.ragas_scored_count,
            "ragas_excluded_refusals": self.ragas_excluded_refusals,
            "ragas_failures": self.ragas_failures,
            "items": self.items,
            "error": self.error,
        }
        # Omitting this field for schema-1 objects keeps historical artifact bytes and the
        # legacy loader contract unchanged. All newly executed runs are explicitly schema 2.
        if self.schema_version >= 2:
            payload["schema_version"] = self.schema_version
        return payload


async def execute_run(
    split: str, label: str, answer_fn, judge_fn, *, k: int = 5, item_failures: dict | None = None
) -> ReleaseRun:
    """`item_failures` lets tests inject a generation failure for a specific item id without a
    real pipeline raising one."""
    assert_split_runnable(split)
    item_failures = item_failures or {}

    started_at = datetime.now(UTC).isoformat()
    golden = _load_golden(_GOLDEN_PATHS[split])

    items: list[dict] = []
    generation_failures: list[dict] = []
    run_results: dict[str, dict] = {}
    for item in golden:
        if item["id"] in item_failures:
            generation_failures.append({"id": item["id"], "error": item_failures[item["id"]]})
            continue
        row = run_item(item, answer_fn)
        run_results[item["id"]] = row
        items.append(row)

    ragas_scored, ragas_excluded_refusals, ragas_failures = await score_ragas(
        golden, run_results, judge_fn
    )

    protocol = (
        json.loads(EVAL_PROTOCOL_PATH.read_text(encoding="utf-8")) if split == "holdout" else None
    )
    status = "partial" if (generation_failures or ragas_failures) else "complete"

    return ReleaseRun(
        run_id=f"{split}-{uuid.uuid4().hex[:12]}",
        split=split,
        label=label,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        status=status,
        holdout_sha256=protocol["holdout_sha256"] if protocol else None,
        pipeline_fingerprint=pipeline_fingerprint(k=k),
        manifest_sha256=manifest_digest(),
        structural_metrics=compute_metrics_v2(items),
        ragas_means=_ragas_means(ragas_scored),
        ragas_scored_count=len(ragas_scored),
        ragas_excluded_refusals=ragas_excluded_refusals,
        ragas_failures=ragas_failures,
        items=items,
        error=(
            f"{len(generation_failures)} generation failure(s), {len(ragas_failures)} judge "
            f"failure(s): {generation_failures + ragas_failures}"
            if status == "partial"
            else None
        ),
        schema_version=2,
    )


def write_run_artifact(run: ReleaseRun, runs_dir=RUNS_DIR) -> "os.PathLike":
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run.run_id}.json"
    path.write_text(
        json.dumps(run.to_dict(), indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    return path


def _write_atomically(path, content: str) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def _write_summary_report(run: ReleaseRun, path) -> None:
    if run.schema_version >= 2:
        return _write_summary_report_v2(run, path)
    m = run.structural_metrics
    lines = [
        "# Release evaluation summary",
        "",
        f"**Split:** {run.split} | **Label:** {run.label} | **Run:** {run.run_id} | "
        f"**Status:** {run.status}",
        "",
        "Fractions shown as `n/d` alongside the percentage — a rate with a small or partial "
        "denominator is not the same claim as one over the full set. The 95% CI is a Wilson "
        "interval on the observed rate, not a claim that the true rate equals the point estimate — "
        "a 30-item holdout leaves real uncertainty even at 100%.",
        "",
        "| metric | value | 95% CI |",
        "|---|---|---|",
        f"| Answerable answer rate | {m['answerable_answered_count']}/{m['answerable_count']}"
        f" ({_pct(m['answerable_answer_rate'])}) |"
        f" {_ci(m['answerable_answered_count'], m['answerable_count'])} |",
        f"| Unanswerable refusal recall | {m['unanswerable_refused_count']}/{m['unanswerable_count']}"
        f" ({_pct(m['unanswerable_refusal_recall'])}) |"
        f" {_ci(m['unanswerable_refused_count'], m['unanswerable_count'])} |",
        f"| Citation-contract pass rate | {m['citation_contract_pass_count']}/{len(run.items)}"
        f" ({_pct(m['citation_contract_pass_rate'])}) |"
        f" {_ci(m['citation_contract_pass_count'], len(run.items))} |",
        f"| Verified-citation rate | {m['verified_citation_count']}/{len(run.items)}"
        f" ({_pct(m['verified_citation_rate'])}) |"
        f" {_ci(m['verified_citation_count'], len(run.items))} |",
    ]
    if run.ragas_means:
        # explicit _RAGAS_METRIC_NAMES order, not dict iteration order: a run loaded back from a
        # JSON artifact (written with sort_keys=True) would otherwise render this row
        # alphabetically instead of matching a freshly-computed run's insertion order
        rendered_metrics = ", ".join(
            f"{name}={run.ragas_means[name]:.3f}" for name in _RAGAS_METRIC_NAMES
        )
        lines.append(
            f"| RAGAS (over {run.ragas_scored_count} answered answerable items, "
            f"{len(run.ragas_excluded_refusals)} refusal(s) excluded) | "
            + rendered_metrics
            + " | — |"
        )
    lines.append("")
    (path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_summary_report_v2(run: ReleaseRun, path) -> None:
    m = run.structural_metrics
    answer = m["answer_coverage"]
    refusal = m["refusal_recall"]
    citation = m["citation_verification"]
    structural = m["structural_pass_rate"]
    outcome = m["task_outcome"]
    lines = [
        "# Release evaluation summary",
        "",
        f"**Schema:** {run.schema_version} | **Split:** {run.split} | **Label:** {run.label} | "
        f"**Run:** {run.run_id} | **Status:** {run.status}",
        "",
        "Rates use the numerator and denominator shown. Wilson 95% intervals apply only to "
        "item-level proportions; citation occurrences are dependent within answers and have no "
        "item-level binomial interval.",
        "",
        "| metric | value | 95% CI |",
        "|---|---|---|",
        f"| Answer coverage | {answer['numerator']}/{answer['denominator']} ({_pct(answer['rate'])}) | "
        f"{_ci(answer['numerator'], answer['denominator'])} |",
        f"| Refusal recall | {refusal['numerator']}/{refusal['denominator']} ({_pct(refusal['rate'])}) | "
        f"{_ci(refusal['numerator'], refusal['denominator'])} |",
        f"| Structurally verified citations | {citation['numerator']}/{citation['denominator']} "
        f"({_pct(citation['rate'])}) | — (citation occurrences) |",
        f"| Structural pass rate among served answers | {structural['numerator']}/"
        f"{structural['denominator']} ({_pct(structural['rate'])}) | "
        f"{_ci(structural['numerator'], structural['denominator'])} |",
        f"| Task outcome | {outcome['numerator']}/{outcome['denominator']} "
        f"({_pct(outcome['rate'])}) | {_ci(outcome['numerator'], outcome['denominator'])} |",
        "| Independently reviewed supported-answer accuracy | unavailable | — |",
        "| Independently reviewed supported-answer yield | unavailable | — |",
    ]
    if run.ragas_means:
        rendered_metrics = ", ".join(
            f"{name}={run.ragas_means[name]:.3f}" for name in _RAGAS_METRIC_NAMES
        )
        lines.append(
            f"| RAGAS (over {run.ragas_scored_count} answered answerable items, "
            f"{len(run.ragas_excluded_refusals)} refusal(s) excluded) | {rendered_metrics} | — |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _pct(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "n/a"


def _ci(successes: int, denominator: int) -> str:
    if not denominator:
        return "—"
    lo, hi = wilson_interval(successes, denominator)
    return f"{lo:.0%}–{hi:.0%}"


def promote_to_canonical(run: ReleaseRun) -> None:
    """Only called for a `complete` run — writes/replaces the canonical summary and appends to
    history atomically, so a crash mid-write can never leave a corrupt or half-updated canonical
    report."""
    summary_path = REPORTS_DIR / "eval_summary.md"
    tmp_summary = REPORTS_DIR / ".eval_summary.md.tmp"
    _write_summary_report(run, tmp_summary)
    os.replace(tmp_summary, summary_path)

    if run.schema_version >= 2:
        _append_history_v2(run)
        return

    is_new = not EVAL_HISTORY_CSV.exists()
    header = "timestamp,run_id,split,label,answerable_answer_rate,unanswerable_refusal_recall,citation_contract_pass_rate,verified_citation_rate\n"
    row = (
        f"{run.finished_at},{run.run_id},{run.split},{run.label},"
        f"{run.structural_metrics['answerable_answer_rate']},"
        f"{run.structural_metrics['unanswerable_refusal_recall']},"
        f"{run.structural_metrics['citation_contract_pass_rate']},"
        f"{run.structural_metrics['verified_citation_rate']}\n"
    )
    with open(EVAL_HISTORY_CSV, "a", encoding="utf-8", newline="\n") as f:
        if is_new:
            f.write(header)
        f.write(row)


def _append_history_v2(run: ReleaseRun) -> None:
    """Append only schema-2 runs to the new history file; the legacy CSV is immutable."""
    metric_names = (
        "answer_coverage",
        "refusal_recall",
        "citation_verification",
        "structural_pass_rate",
        "task_outcome",
    )
    header = ["schema_version", "timestamp", "run_id", "split", "label"]
    for name in metric_names:
        header.extend([f"{name}_numerator", f"{name}_denominator", name])
    header.append("ragas_scored_count")
    row = [run.schema_version, run.finished_at, run.run_id, run.split, run.label]
    for name in metric_names:
        metric = run.structural_metrics[name]
        row.extend([metric["numerator"], metric["denominator"], metric["rate"]])
    row.append(run.ragas_scored_count)
    is_new = not EVAL_HISTORY_V2_CSV.exists()
    with open(EVAL_HISTORY_V2_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "holdout"], required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    from evals._ragas_judge import build_judge
    from src.rag import answer_question

    llm, embeddings = build_judge()
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    async def judge_fn(question, answer, chunk_ids, reference):
        from src.retrieve import retrieve

        contexts = [c.text for c in retrieve(question, k=5, rerank=True)]
        faithfulness, relevancy, precision, recall = await asyncio.gather(
            metrics["faithfulness"].ascore(
                user_input=question, response=answer, retrieved_contexts=contexts
            ),
            metrics["answer_relevancy"].ascore(user_input=question, response=answer),
            metrics["context_precision"].ascore(
                user_input=question, reference=reference, retrieved_contexts=contexts
            ),
            metrics["context_recall"].ascore(
                user_input=question, retrieved_contexts=contexts, reference=reference
            ),
        )
        return {
            "faithfulness": faithfulness.value,
            "answer_relevancy": relevancy.value,
            "context_precision": precision.value,
            "context_recall": recall.value,
        }

    try:
        run = asyncio.run(execute_run(args.split, args.label, answer_question, judge_fn))
    except HoldoutNotSealedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    artifact_path = write_run_artifact(run)
    print(f"wrote {artifact_path} (status={run.status})")

    if run.status != "complete":
        print(f"PARTIAL RUN: {run.error}", file=sys.stderr)
        raise SystemExit(1)

    promote_to_canonical(run)
    print("promoted to reports/eval_summary.md and appended reports/eval_history.csv")


if __name__ == "__main__":
    main()
