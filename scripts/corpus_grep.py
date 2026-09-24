"""Case-insensitive phrase search over every corpus page.

python -m scripts.corpus_grep "phrase" [--doc <id>] [--max 30]
"""

import argparse
import json
import re
import sys

import fitz

from src.config import CORPUS_DIR, MANIFEST_PATH
from src.ingest import OCR_CACHE_DIR


def _pages(entry: dict):
    if entry.get("text_layer") == "ocr":
        cache = OCR_CACHE_DIR / entry["id"]
        files = sorted(cache.glob("*.txt"), key=lambda p: int(p.stem))
        for f in files:
            yield int(f.stem), f.read_text(encoding="utf-8")
        return
    with fitz.open(CORPUS_DIR / entry["filename"]) as doc:
        for i, page in enumerate(doc, start=1):
            yield i, page.get_text()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("phrase")
    ap.add_argument("--doc")
    ap.add_argument("--max", type=int, default=30)
    args = ap.parse_args()
    pattern = re.compile(re.escape(args.phrase).replace(r"\ ", r"\s+"), re.IGNORECASE)
    found = 0
    for entry in json.loads(MANIFEST_PATH.read_text(encoding="utf-8")):
        if args.doc and entry["id"] != args.doc:
            continue
        for page, text in _pages(entry):
            m = pattern.search(text)
            if not m:
                continue
            snippet = " ".join(text[max(0, m.start() - 80) : m.end() + 80].split())
            print(f"{entry['id']} p.{page}: ...{snippet}...")
            found += 1
            if found >= args.max:
                return


if __name__ == "__main__":
    main()
