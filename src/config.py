"""Single source of truth for paths, seeds and pipeline constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CORPUS_DIR = ROOT / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_SEED = 42

# --- ingestion / chunking ---
# tiktoken's cl100k_base isn't the tokenizer either Claude or the embedding model actually uses —
# there's no free, dependency-light tokenizer for either — but it's a stable, fast proxy for
# "roughly how big is this chunk", which is all the chunk-size budget needs.
TOKENIZER_ENCODING = "cl100k_base"
CHUNK_TARGET_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 75
MIN_CHUNK_TOKENS = 40  # trailing fragments below this get merged into the previous chunk

# a line repeated across at least this fraction of a document's pages is running header/footer
# boilerplate (address blocks, "Page X of Y"), not content
HEADER_FOOTER_REPEAT_FRACTION = 0.4
# a page where at least this fraction of lines look like "Section name .......... 12" is a table
# of contents, not prose worth chunking
TOC_DOT_LEADER_FRACTION = 0.3
