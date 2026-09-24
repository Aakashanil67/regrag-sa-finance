"""Single source of truth for paths, seeds and pipeline constants."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CORPUS_DIR = ROOT / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

EVALS_DIR = ROOT / "evals"
GOLDEN_DEV_PATH = EVALS_DIR / "golden_dev.jsonl"
GOLDEN_HOLDOUT_PATH = EVALS_DIR / "golden_holdout.jsonl"
RETRIEVAL_DEV_PATH = EVALS_DIR / "retrieval_dev.json"
RETRIEVAL_HOLDOUT_PATH = EVALS_DIR / "retrieval_holdout.json"
EVAL_PROTOCOL_PATH = EVALS_DIR / "protocol.json"
EVAL_HISTORY_CSV = REPORTS_DIR / "eval_history.csv"

CACHE_DB_PATH = ROOT / "regrag_cache.sqlite3"


# --- ingestion / chunking ---
def _env(name: str, default: str) -> str:
    return os.environ.get(f"REGRAG_{name}", default)


# tiktoken's cl100k_base isn't the tokenizer either Claude or the embedding model actually uses —
# there's no free, dependency-light tokenizer for either — but it's a stable, fast proxy for
# "roughly how big is this chunk", which is all the chunk-size budget needs.
TOKENIZER_ENCODING = "cl100k_base"
# 800 tokens plus reranking won the v1.0 chunk-size sweep (reports/archive/v1.0-audit/improvement_log.md)
CHUNK_MODE = _env("CHUNK_MODE", "tiktoken")  # tiktoken | wordpiece
CHUNK_TARGET_TOKENS = int(_env("CHUNK_TARGET", "800"))
CHUNK_OVERLAP_TOKENS = int(_env("CHUNK_OVERLAP", "75"))
EMBEDDING_MODEL_NAME = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL_REVISION = _env("EMBEDDING_REVISION", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
RETRIEVAL_STRATEGY = _env("RETRIEVAL_STRATEGY", "semantic")
RERANK = _env("RERANK", "1") == "1"
CITATION_REPAIR = _env("CITATION_REPAIR", "0") == "1"
RETRIEVAL_K = int(_env("K", "5"))
_chroma = Path(_env("CHROMA_DIR", "chroma"))
CHROMA_DIR = _chroma if _chroma.is_absolute() else ROOT / _chroma


def effective_settings() -> dict:
    """The knobs a named config can change, as this process sees them."""
    return {
        "chunk_mode": CHUNK_MODE,
        "chunk_target": CHUNK_TARGET_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP_TOKENS,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_revision": EMBEDDING_MODEL_REVISION,
        "retrieval_strategy": RETRIEVAL_STRATEGY,
        "rerank": RERANK,
        "citation_repair": CITATION_REPAIR,
        "chroma_dir": (
            CHROMA_DIR.relative_to(ROOT).as_posix()
            if CHROMA_DIR.is_relative_to(ROOT)
            else str(CHROMA_DIR)
        ),
        "k": RETRIEVAL_K,
    }


MIN_CHUNK_TOKENS = 40  # trailing fragments below this get merged into the previous chunk

# a line repeated across at least this fraction of a document's pages is running header/footer
# boilerplate (address blocks, "Page X of Y"), not content
HEADER_FOOTER_REPEAT_FRACTION = 0.4
# a page where at least this fraction of lines look like "Section name .......... 12" is a table
# of contents, not prose worth chunking
TOC_DOT_LEADER_FRACTION = 0.3

# --- vector store ---
COLLECTION_NAME = "regrag_chunks"
# The serving path remains the legacy tiktoken configuration until a retrieval variant is
# selected. These explicit values are the tokenizer-aware experiment defaults; they are kept
# separate so an audit or isolated store cannot silently change serving behaviour.
EMBEDDING_TOKENIZER_NAME = EMBEDDING_MODEL_NAME
EMBEDDING_TOKENIZER_REVISION = EMBEDDING_MODEL_REVISION
TOKENIZER_AWARE_CHUNK_TARGET_TOKENS = 240
TOKENIZER_AWARE_CHUNK_OVERLAP_TOKENS = 32
# Chroma still owns document/metadata storage, but candidate search itself is exact cosine
# similarity computed in retrieve.py, not Chroma's own approximate HNSW `.query()` — see
# retrieve.py's module docstring for why (raising hnsw:search_ef was tried and did not fix it:
# collection.modify() updates the collection's own metadata row, not the segment's, so the running
# HNSW index never actually saw the new value).

# --- retrieval / reranking ---
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
CROSS_ENCODER_MODEL_REVISION = RERANKER_MODEL_REVISION
RERANK_CANDIDATE_POOL_SIZE = 20  # how many embedding-search candidates the reranker sees

# --- LLM ---
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
RAG_MAX_ANSWER_TOKENS = 1024

# $ per 1M tokens (input, output) — approximate for OpenAI/Ollama, exact for Anthropic at time of
# writing; used only for the cost-estimate column in reports/logs, never billed against directly.
PRICING_PER_MILLION_TOKENS = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
