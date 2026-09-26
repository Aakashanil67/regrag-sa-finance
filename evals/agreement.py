"""Agreement between two labelled runs (judge vs judge, or judge vs human).

    python -m evals.agreement A.judge-qwen.json B.judge-llama.json --field correctness

Inputs are judge JSON files or CSVs with an `id` column. Only ids labelled in both are compared.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

SKIP = {"refused", "answered_unanswerable", "unparsed", "", None}


def load_labels(path: str | Path, field: str) -> dict[str, str]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))["items"]
    labels = {}
    for row in rows:
        value = (row.get(field) or "").strip() if isinstance(row.get(field), str) else row.get(field)
        if value in SKIP:
            continue
        labels[str(row["id"])] = value
    return labels


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return 0.0
    observed = sum(1 for a, b in pairs if a == b) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum(left[k] * right[k] for k in left) / (n * n)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def paired(a: dict[str, str], b: dict[str, str]) -> list[tuple[str, str]]:
    return [(a[i], b[i]) for i in sorted(a.keys() & b.keys())]


def confusion_table(pairs: list[tuple[str, str]]) -> str:
    labels = sorted({x for p in pairs for x in p})
    counts = Counter(pairs)
    lines = ["| A \ B | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for a in labels:
        lines.append(f"| {a} | " + " | ".join(str(counts[(a, b)]) for b in labels) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--field", default="correctness")
    args = ap.parse_args(argv)
    pairs = paired(load_labels(args.a, args.field), load_labels(args.b, args.field))
    raw = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else 0.0
    print(f"n = {len(pairs)}")
    print(f"raw agreement = {raw:.3f}")
    print(f"kappa = {cohen_kappa(pairs):.3f}\n")
    print(confusion_table(pairs))


if __name__ == "__main__":
    main()
