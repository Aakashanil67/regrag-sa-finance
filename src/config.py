"""Single source of truth for paths, seeds and pipeline constants."""

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
# tiktoken's cl100k_base isn't the tokenizer either Claude or the embedding model actually uses —
# there's no free, dependency-light tokenizer for either — but it's a stable, fast proxy for
# "roughly how big is this chunk", which is all the chunk-size budget needs.
TOKENIZER_ENCODING = "cl100k_base"
# 800, not the originally-planned 500: reports/archive/v1.0-audit/improvement_log.md swept 300/500/800 tokens x
# reranking on/off against the retrieval benchmark. 800+rerank won on every measure (hit-rate@5
# 95% vs 500's 85%, MRR 0.808 vs 0.654) and reranking improved every chunk size it was paired
# with — larger chunks give the cross-encoder more context to judge relevance against, at the
# cost of a coarser citation (a chunk's page range covers more ground). 500 was the more common
# default for a first implementation; 800 is what the eval actually rewarded.
CHUNK_TARGET_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 75
MIN_CHUNK_TOKENS = 40  # trailing fragments below this get merged into the previous chunk

# a line repeated across at least this fraction of a document's pages is running header/footer
# boilerplate (address blocks, "Page X of Y"), not content
HEADER_FOOTER_REPEAT_FRACTION = 0.4
# a page where at least this fraction of lines look like "Section name .......... 12" is a table
# of contents, not prose worth chunking
TOC_DOT_LEADER_FRACTION = 0.3

# --- vector store ---
CHROMA_DIR = ROOT / "chroma"
COLLECTION_NAME = "regrag_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Chroma still owns document/metadata storage, but candidate search itself is exact cosine
# similarity computed in retrieve.py, not Chroma's own approximate HNSW `.query()` — see
# retrieve.py's module docstring for why (raising hnsw:search_ef was tried and did not fix it:
# collection.modify() updates the collection's own metadata row, not the segment's, so the running
# HNSW index never actually saw the new value).

# --- retrieval / reranking ---
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
