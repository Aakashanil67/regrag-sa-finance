"""Downloads the regulation PDFs listed in corpus/manifest.json into corpus/.

Idempotent: a file already present with a matching SHA-256 is left alone, so re-running after a
partial fetch only downloads what's missing. Every source is a real regulator/publisher URL that
returns a hard 200+text/html rejection when curl-style requests lack browser-shaped headers (SARB,
NCR and FSCA all sit behind a WAF that blocks bare user agents) — see `_HEADERS` below.

    python -m scripts.fetch_corpus            # fetch anything missing or mismatched
    python -m scripts.fetch_corpus --force     # redownload everything regardless of checksum
"""

import argparse
import hashlib
import json
import sys
import time

import httpx

from src.config import CORPUS_DIR, MANIFEST_PATH

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(entry: dict) -> bytes:
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/pdf,*/*",
        "Referer": entry["referer"],
    }
    last_error = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = httpx.get(entry["url"], headers=headers, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
                raise ValueError(f"response wasn't a PDF (content-type: {content_type!r})")
            return response.content
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)
    raise RuntimeError(
        f"failed to fetch {entry['id']} after {_MAX_ATTEMPTS} attempts: {last_error}"
    )


def fetch_all(force: bool = False) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    counts = {"skipped": 0, "downloaded": 0, "failed": 0}
    for entry in manifest:
        dest = CORPUS_DIR / entry["filename"]

        if not force and dest.exists() and _sha256(dest) == entry["sha256"]:
            counts["skipped"] += 1
            print(f"  ok      {entry['id']}")
            continue

        try:
            content = _download(entry)
        except RuntimeError as exc:
            counts["failed"] += 1
            print(f"  FAILED  {entry['id']}: {exc}")
            continue

        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != entry["sha256"]:
            counts["failed"] += 1
            print(
                f"  FAILED  {entry['id']}: checksum mismatch — the source document has changed "
                f"since this manifest was pinned (expected {entry['sha256'][:12]}…, "
                f"got {actual_hash[:12]}…). Not saving; update corpus/manifest.json if the new "
                "version is expected."
            )
            continue

        dest.write_bytes(content)
        counts["downloaded"] += 1
        print(f"  fetched {entry['id']} ({len(content) / 1024:.0f} KB)")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="redownload every document")
    args = parser.parse_args()

    counts = fetch_all(force=args.force)
    print(
        f"\n{counts['downloaded']} fetched, {counts['skipped']} already up to date, {counts['failed']} failed"
    )
    if counts["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
