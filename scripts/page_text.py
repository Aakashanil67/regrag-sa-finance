"""Print one corpus page's text: python -m scripts.page_text <doc_id> <page>"""

import json
import sys

import fitz

from src.config import CORPUS_DIR, MANIFEST_PATH
from src.ingest import OCR_CACHE_DIR


def page_text(doc_id: str, page: int) -> str:
    entry = next(
        e for e in json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if e["id"] == doc_id
    )
    if entry.get("text_layer") == "ocr":
        return (OCR_CACHE_DIR / doc_id / f"{page}.txt").read_text(encoding="utf-8")
    with fitz.open(CORPUS_DIR / entry["filename"]) as doc:
        return doc[page - 1].get_text()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(page_text(sys.argv[1], int(sys.argv[2])))
