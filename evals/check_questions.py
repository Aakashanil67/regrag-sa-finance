"""Validate a practitioner question file: python -m evals.check_questions <file.jsonl>

Every quoted piece of evidence must really be on the cited page, so a reference answer can't
drift from the corpus without this failing.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

from scripts.page_text import page_text
from src.config import MANIFEST_PATH

DEV_COUNTS = {
    "single": 12,
    "threshold": 6,
    "status": 5,
    "multi": 8,
    "unanswerable": 6,
    "false_premise": 3,
}
TYPES = set(DEV_COUNTS)
CORRECTION_WORDS = (
    "not",
    "no longer",
    "superseded",
    "withdrawn",
    "replaced",
    "incorrect",
    "actually",
    "rather",
    "instead",
)
NAMED_INSTRUMENT = re.compile(
    r"(directive|guidance note|guideline|circular|conduct standard)\s+\d"
    r"|\b[a-z]\d{1,2}/20\d\d\b|\b\d{1,3}/20\d\d\b|\bper the\b",
    re.IGNORECASE,
)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    return text.strip()


def _quote_on_page(doc_id: str, page: int, quote: str) -> bool:
    q = normalise(quote)
    try:
        here = normalise(page_text(doc_id, page))
    except (FileNotFoundError, IndexError):
        return False
    try:
        nxt = normalise(page_text(doc_id, page + 1))
    except (FileNotFoundError, IndexError):
        nxt = ""
    return q in here or q in f"{here} {nxt}"


def check(path: Path) -> list[str]:
    problems: list[str] = []
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    manifest = {e["id"] for e in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))}
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    counts: dict[str, int] = {}
    named = 0
    named_pool = 0
    for row in rows:
        rid = row.get("id", "?")
        if not re.fullmatch(r"[dt]\d\d", rid) or rid in seen_ids:
            problems.append(f"{rid}: bad or duplicate id")
        seen_ids.add(rid)
        kind = row.get("type")
        if kind not in TYPES:
            problems.append(f"{rid}: unknown type {kind!r}")
            continue
        counts[kind] = counts.get(kind, 0) + 1
        q = row.get("question", "")
        if not 25 <= len(q) <= 300 or not q.endswith("?"):
            problems.append(f"{rid}: question must be 25-300 chars and end in '?'")
        if q.lower() in seen_questions:
            problems.append(f"{rid}: duplicate question")
        seen_questions.add(q.lower())
        if kind in ("single", "threshold"):
            named_pool += 1
            named += bool(NAMED_INSTRUMENT.search(q))
        evidence = row.get("evidence", [])
        if kind == "unanswerable":
            if evidence:
                problems.append(f"{rid}: unanswerable item has evidence")
            if len(row.get("why_unanswerable", "")) < 40:
                problems.append(f"{rid}: why_unanswerable too short")
            continue
        if not evidence:
            problems.append(f"{rid}: answerable item has no evidence")
        if kind == "multi" and len({e.get("doc_id") for e in evidence}) < 2:
            problems.append(f"{rid}: multi item needs two distinct documents")
        if kind == "false_premise" and not any(
            w in row.get("reference_answer", "").lower() for w in CORRECTION_WORDS
        ):
            problems.append(f"{rid}: reference does not correct the premise")
        for e in evidence:
            doc_id, page, quote = e.get("doc_id"), e.get("page"), e.get("quote", "")
            if doc_id not in manifest:
                problems.append(f"{rid}: doc_id {doc_id!r} not in manifest")
            elif not 30 <= len(quote) <= 300:
                problems.append(f"{rid}: quote must be 30-300 chars")
            elif not _quote_on_page(doc_id, page, quote):
                problems.append(f"{rid}: quote not found on {doc_id} p.{page}")
    if any(r.get("id", "").startswith("d") for r in rows) and counts != DEV_COUNTS:
        problems.append(f"type counts {counts} != {DEV_COUNTS}")
    if named_pool and named / named_pool > 0.25:
        problems.append(
            f"{named}/{named_pool} single+threshold questions name an instrument (max 25%)"
        )
    return problems


def main() -> None:
    problems = check(Path(sys.argv[1]))
    for p in problems:
        print(p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
