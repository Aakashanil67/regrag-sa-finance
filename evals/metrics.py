"""Run metrics. Every rate keeps its own numerator and denominator. Nothing is pooled."""

import re
from collections import Counter

from evals.stats import wilson_interval

_CITATION = re.compile(r"\[([\w\-\.]+),\s*p\.(\d+)(?:-(\d+))?(?:,[^\]]*)?\]")


def ratio(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def ratio_text(r: dict) -> str:
    return f"{r['k']}/{r['n']}" + (f" ({r['rate']:.0%})" if r["n"] else "")


def ci_text(r: dict) -> str:
    lo, hi = r["ci95"]
    return "n/a" if lo is None else f"{lo:.0%}-{hi:.0%}"


def _covered(contexts: list[dict]) -> dict[str, set[int]]:
    pages: dict[str, set[int]] = {}
    for c in contexts:
        pages.setdefault(c["doc_id"], set()).update(range(c["page_start"], c["page_end"] + 1))
    return pages


def raw_citations(text: str | None, contexts: list[dict]) -> tuple[int, int]:
    """(pages cited that were in the context, pages cited) in the model's raw output."""
    covered = _covered(contexts)
    verified = total = 0
    for doc_id, start, end in _CITATION.findall(text or ""):
        for page in range(int(start), int(end or start) + 1):
            total += 1
            verified += page in covered.get(doc_id, set())
    return verified, total


def evidence_hit(item: dict, contexts: list[dict]) -> tuple[bool, bool]:
    """(any evidence page retrieved, every evidence document has a retrieved evidence page)."""
    covered = _covered(contexts)
    hits = [(e["doc_id"], e["page"] in covered.get(e["doc_id"], set())) for e in item["evidence"]]
    if not hits:
        return False, False
    docs = {d for d, _ in hits}
    return any(h for _, h in hits), all(any(h for d2, h in hits if d2 == d) for d in docs)


def compute(items: list[dict]) -> dict:
    answerable = [i for i in items if i["type"] != "unanswerable"]
    unanswerable = [i for i in items if i["type"] == "unanswerable"]
    answered = [i for i in answerable if not i["refused"]]
    refused_unanswerable = [i for i in unanswerable if i["refused"]]
    served = [i for i in items if not i["refused"]]
    raw_ok = raw_total = 0
    for i in items:
        ok, total = raw_citations(i.get("raw_model_output"), i.get("contexts", []))
        raw_ok += ok
        raw_total += total
    served_citations = [c for i in served for c in i.get("citations", [])]
    hits = [evidence_hit(i, i.get("contexts", [])) for i in answerable]
    return {
        "n": len(items),
        "answer_rate": ratio(len(answered), len(answerable)),
        "refusal_recall": ratio(len(refused_unanswerable), len(unanswerable)),
        "task_outcome": ratio(len(answered) + len(refused_unanswerable), len(items)),
        "raw_citation_precision": ratio(raw_ok, raw_total),
        "served_citations": len(served_citations),
        "served_unverified_citations": sum(1 for c in served_citations if not c["verified"]),
        "section_ref_rate": ratio(
            sum(1 for c in served_citations if c.get("section_ref")), len(served_citations)
        ),
        "retrieval_any_hit": ratio(sum(a for a, _ in hits), len(answerable)),
        "retrieval_all_hit": ratio(sum(b for _, b in hits), len(answerable)),
        "repairs": sum(1 for i in items if i.get("repair_attempted")),
        "refusal_reasons": dict(Counter(i["refusal_reason"] for i in items if i["refused"])),
        "failed_items": sum(1 for i in items if i.get("error")),
        "cost_usd": round(sum(i["usage"]["cost_usd"] for i in items), 4),
    }
