# Engineering decisions log

Why the code looks the way it does. Written as I went, so the reasoning is what I actually had at
the time rather than a tidy reconstruction.

## Module map

| Module | Responsibility |
|---|---|
| `scripts/fetch_corpus.py` | Checksummed download of the 19 corpus PDFs, WAF-aware headers. |
| `src/ingest.py` | PyMuPDF extraction, boilerplate stripping, ToC detection, heading detection, de-hyphenation. |
| `src/chunking.py` | Heading-aware ~500-token chunks with overlap, oversized-paragraph splitting. |
| `src/report_chunks.py` | `reports/chunk_quality.md` and the size histogram. |
| `src/store.py` | sentence-transformers embeddings into a persistent ChromaDB collection, hash-based idempotent rebuild. |
| `src/retrieve.py` | Top-k semantic retrieval with metadata filters. |
| `src/llm.py` | Provider-agnostic chat completion (anthropic / openai / ollama). |
| `src/rag.py` | Retrieve → cited-answer prompt → citation validation → refusal detection. |
| `src/guardrails.py` | Pattern-based prompt-injection flagging (observability, not a gate). |
| `src/cache.py` | Exact-match, question-hash response cache. |
| `src/obslog.py` | SQLite query log; wires the cache and latency timing around `rag.answer_question`. |
| `api/main.py` | FastAPI `/ask`, `/health`, `/stats`. |
| `app/chat.py` / `app/ops.py` | Streamlit chat UI and observability dashboard, both calling the API over HTTP. |

## Corpus

**SARB, NCR and FSCA all reject bare `curl`/`requests` calls with a 200 that's actually an HTML
rejection page**, not a 403 — a WAF pattern that looks like success until you check the
content-type. A realistic `User-Agent` plus a same-site `Referer` header was enough to pass on all
three; no cookies or session handshake needed. `scripts/fetch_corpus.py` sends both on every
request. Confirmed by fetching one file through the actual browser first (which worked
immediately) before concluding it was a headers problem rather than a genuinely broken URL.

**Two 2004 Basel II circulars are in the corpus despite being nearly a decade out of date**,
because they're substantive regulatory text (Circular 19/2004 on hybrid capital instruments runs
several real pages; the alternative candidates I found first — G1/2023 "status of previously
issued guidance notes", G5/2018 — turned out to be thin administrative index pages or
poorly-OCR'd scans that would have produced garbage chunks and useless golden-set questions).
Corpus quality beat corpus currency as the selection criterion.

**The full IFRS 9 standard text isn't in the corpus.** The IFRS Foundation licenses that
separately; the freely-available Project Summary plus a PwC practitioner guide cover the same
concepts (classification, ECL, own-credit) without the redistribution question.

## Ingestion and chunking

**Heading detection uses numbering-pattern OR relative font size, never one alone.** Numbered
list items ("1. First requirement...") aren't headings, and some of these documents use a single
font size throughout, so neither signal alone is reliable. Accepting some false positives/negatives
here was a deliberate trade-off against building a real layout model for 19 PDFs.

**A single run-on legislative sentence can still bust the ~500-token chunk budget even after
sentence-level packing** — the NCA's registration-and-review clauses routinely go for a paragraph
with almost no terminal punctuation. First pass through the real corpus produced a 1093-token
chunk (2x+ the target) in `nca_act_34_2005`, caught by `reports/chunk_quality.md`'s max-token
column, not by a unit test — the synthetic test paragraphs I'd written didn't reproduce a sentence
that's itself oversized. Fixed with `_hard_split_by_tokens`, a token-window fallback that only
fires when sentence-level packing still isn't enough. Chunk max across the whole corpus is now 572
tokens (500 target + 75 overlap, the theoretical ceiling).

**Chroma document IDs are the chunk's own content hash**, not an incrementing counter. Re-running
`store.py --rebuild` after editing one document only re-embeds chunks whose hash actually changed;
everything else is a no-op lookup. The cost of this: a chunk that moves to a different page (same
text, page renumbered) gets treated as unchanged, which is correct — the text is what retrieval
matches against, not the page number.

## RAG core

**Citations are checked against what was actually retrieved, not trusted because the model wrote
them.** `rag._extract_citations` cross-references every `[doc_id, p.X]` the model outputs against
the (doc_id, page) pairs the retrieved chunks actually cover, and flags a mismatch as unverified.
An LLM citing a page it wasn't shown is a hallucination even when the surrounding prose is
accurate — a distinct failure mode from "the answer is wrong" that a faithfulness score alone
wouldn't isolate.

**Refusal is an exact, greppable sentence**, not "however Claude happens to phrase not knowing
something." `INSUFFICIENT_CONTEXT_PHRASE` is checked case-insensitively as a substring, which is
crude but deterministic — the eval harness and the regression gate both need to detect refusal
without parsing free text.

## Guardrails

**Prompt-injection detection flags, it doesn't block.** The domain (SA financial regulation Q&A)
is narrow enough that a legitimate question tripping a detector pattern is very unlikely, but not
zero — and refusing a real question because it contains the word "ignore" is a worse failure mode
for a demo tool than letting a flagged-but-harmless one through to the system prompt's actual
defense (rule 5 in `rag._SYSTEM_PROMPT`, which is the layer that can actually stop a followed
instruction, not the regex). See `reports/security_notes.md` for the full reasoning.

## Caching

**Cache lookups are exact-match on normalised question text, not semantic.** A semantic cache
(embed the query, check similarity against past ones) would catch more repeat traffic, but risks
serving a cached answer to a question that's subtly different from the one actually asked — wrong
for a tool whose whole premise is citation accuracy. Exact-match trades hit rate for that
guarantee; documented as a real trade-off in `reports/perf.md`, not a limitation glossed over.

## Things that fought me

**pip's resolver on `ragas==0.2.15` + `deepeval` + the rest of requirements.txt as one install
took over an hour and never finished** — not a resolver deadlock, just genuinely bandwidth-bound
(measured ~40KB/s against a PyPI mirror on this connection) pulling torch, transformers and the
langchain family as one dependency graph. Splitting the install into stages (light deps first,
then chromadb+sentence-transformers, letting torch download on its own) didn't fix the bandwidth,
but made partial progress recoverable instead of restarting a single giant resolve from zero.
`deepeval==2.5.6`'s pin of `pytest<8.0` was a separate, real conflict against this project's
`pytest==8.3.4` — moved to `deepeval==4.1.10`, whose dependency list turned out to be considerably
lighter (no `llama-index`/`instructor` pulled in) as well as resolver-compatible.

**`chroma-hnswlib` has no prebuilt Windows wheel for this Python/setup and needs a C++ compiler
to build from source** — `chromadb`'s vector index is a compiled extension, and pip's fallback to
`building wheel for chroma-hnswlib` fails outright without Microsoft's C++ Build Tools installed.
Not a chromadb bug, just an unmet system dependency this project's README now calls out explicitly
in the setup instructions, since "pip install everything from requirements.txt" silently isn't
sufficient on a fresh Windows machine.
