"""Freeze real model outputs from a dev run into tests/fixtures/replay.json.

Usage: python -m evals.make_fixture reports/runs/dev-<name>.json
"""

import json
import sys
from pathlib import Path

LIMIT = 12
OUT = Path("tests/fixtures/replay.json")
CONTEXT_KEYS = ("chunk_id", "doc_id", "page_start", "page_end", "section", "text")


def build(run_path: Path) -> dict:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    usable = [i for i in run["items"] if i.get("raw_model_output") and i.get("contexts")]
    usable.sort(key=lambda i: i["id"])
    picked = [i for i in usable if i["refused"]]
    for item in usable:
        if len(picked) >= LIMIT:
            break
        if not item["refused"]:
            picked.append(item)
    picked.sort(key=lambda i: i["id"])
    return {
        "source": run_path.name,
        "items": [
            {
                "id": i["id"],
                "raw_model_output": i["raw_model_output"],
                "contexts": [[c[k] for k in CONTEXT_KEYS] for c in i["contexts"]],
                "refused": i["refused"],
                "refusal_reason": i["refusal_reason"],
                "citations": i["citations"],
            }
            for i in picked
        ],
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fixture = build(Path(sys.argv[1]))
    OUT.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(fixture['items'])} items to {OUT}")
