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

**A real bug produced 24 duplicate-content chunk groups on first embedding of the full corpus**
(1077 raw chunks, only 1015 unique — caught by diffing `chunk_corpus()`'s output against what
`store.py` actually stored, not by a test, since none of my synthetic test paragraphs happened to
put two oversized paragraphs back to back). Root cause: `flush()` always reseeds `current_parts`
with an overlap tail for the *next normal chunk*, but the oversized-paragraph branch bypasses
`current_parts` entirely (it appends pieces straight to `chunks`) without clearing that seed —
so a stale, unrelated overlap tail from *before* the oversized paragraph would resurface as its
own spurious mini-chunk the next time `flush()` ran, which in `nca_act_34_2005` (heavy with
long legislative clauses that repeatedly trigger the oversized path) happened dozens of times.
Every chunk-hash formula includes page_start too now (see below) rather than text alone, which is
what actually surfaced this as a *correctness* bug rather than a cosmetic one: two genuinely
different chunk positions were colliding onto one stored vector, silently keeping whichever
page's citation got embedded first and discarding the other. Fixed by explicitly resetting and
then reseeding `current_parts` from the oversized paragraph's own last piece; a second, narrower
case (a very short chunk's overlap tail is the *entire* chunk, which can then re-emit verbatim if
nothing new gets appended before the next flush) is closed with a direct
identical-to-previous-chunk guard in `flush()`. Full corpus: 0 duplicate chunk-hash groups now,
down from 24. Regression test: `test_two_oversized_paragraphs_in_a_row_do_not_leak_a_stale_overlap_chunk`
— verified it actually fails against the pre-fix code before trusting it.

**Chroma document IDs are `sha256(doc_id:page_start:text)`, not text alone and not an incrementing
counter.** Text alone was the first version, and it was wrong: legislative documents genuinely
repeat identical passages at different pages (a definitional clause quoted again in a later
schedule), which a text-only hash silently collapses into one stored vector — exactly the same
failure shape as the bug above, just a legitimate content repeat instead of a chunking bug.
Re-running `store.py --rebuild` after editing one document still only re-embeds chunks whose
(page, text) pair actually changed; everything else is a no-op lookup.

## Retrieval benchmark

**The first real run of `evals/retrieval_bench.py` scored hit-rate@5 = 55% — most of that was my
own labeling bug, not a retrieval problem.** Four NCR guideline PDFs and one FSCA PDF have a
title-only cover page as PDF page 1, with real content starting on page 2; I'd sourced the
`retrieval_set.json` questions from a bulk `pdftotext -f 1 -l 2` sweep that concatenated both
pages together and attributed everything to "page 1." Diagnosing the misses individually (dumping
each miss's actual top-5 results) showed the correct document landing at rank 1-2 for five of the
nine "misses," just on page 2 instead of the page 1 I'd labeled — confirmed by checking
`pdftotext -f 2 -l 2` against each PDF directly before touching the label, not assumed. Fixing
those five (and the same error in eleven `golden.jsonl` source references, same root cause)
brought the honest number to hit-rate@5 = 85%, MRR 0.654. The other four "misses" survived that
audit and are genuine: `sarb_d3_2023` (page label was independently re-verified correct; MiniLM
just doesn't connect "which earlier directive does X replace" to prose naming "Directive
5/2017"), `fsca_tcf_2011` (a 3-chunk slide-deck agenda losing to a much larger, topically
overlapping FSCA document), and the "own credit" IFRS 9 phrasing. Reported as-is in
`reports/retrieval_bench.md` rather than dropped from the question set — a benchmark that quietly
removes the questions a system gets wrong isn't measuring anything.

## RAG core

**Citations are checked against what was actually retrieved, not trusted because the model wrote
them.** `rag._extract_citations` cross-references every `[doc_id, p.X]` the model outputs against
the (doc_id, page) pairs the retrieved chunks actually cover, and flags a mismatch as unverified.
An LLM citing a page it wasn't shown is a hallucination even when the surrounding prose is
accurate — a distinct failure mode from "the answer is wrong" that a faithfulness score alone
wouldn't isolate.

**The citation checker's first live run against the real API flagged 7 of 9 genuinely correct
citations as unverified** — not a model problem, a bug in my own verification code. `covered_pages`
was a dict comprehension keyed on `doc_id`, so when a query retrieved several chunks from the same
document (the ordinary case, not an edge case — this one retrieved four from
`ifrs9_project_summary_2014`), each chunk's page range silently overwrote the previous one instead
of accumulating, leaving `covered_pages` holding only the *last* chunk's pages. Every mocked test
up to that point used exactly one chunk per document, so nothing caught it before a real question
did. Fixed by unioning page ranges into a `set` per document instead of overwriting; regression
test constructs two same-document chunks specifically to catch a return of the old behaviour.

**Refusal is an exact, greppable sentence**, not "however Claude happens to phrase not knowing
something." `INSUFFICIENT_CONTEXT_PHRASE` is checked case-insensitively as a substring, which is
crude but deterministic — the eval harness and the regression gate both need to detect refusal
without parsing free text.

**The first real smoke-test run crashed with `table queries has no column named
flagged_injection`** — `obslog.py`'s `CREATE TABLE IF NOT EXISTS` had run against a
`regrag_log.sqlite3` created before that column (and `cache_hit`) existed in the schema, and
`IF NOT EXISTS` is a no-op against a table that's already there regardless of whether its columns
match. Every insert since those columns were added would have failed the same way — a real gap,
not just a stale local file, since anyone who ran an earlier version of this code and kept their
log would hit it on upgrade. Fixed with an explicit migration (`PRAGMA table_info` diffed against
the expected column list, missing ones added via `ALTER TABLE ADD COLUMN`) instead of deleting the
file and moving on; regression test builds the old-schema table by hand and confirms logging
against it survives.

**Smoke test, 10 real questions against the live API:** 7 answered with fully-verified citations,
2 correctly refused (genuinely unanswerable — SARB repo rate, JSE listing requirements, neither in
this corpus), 1 refused that should have been answerable (`sarb_d3_2023`'s impairment
classification question) — consistent with the same document's weak retrieval already surfaced in
`reports/retrieval_bench.md`, which is the point of running both: the benchmark predicted this
exact failure before the smoke test hit it live. Three citations spot-checked word-for-word
against the source PDFs (`nca_act_34_2005` p.48, `fsca_conduct_standard_otc_derivatives_2018`
p.1, `ifrs9_project_summary_2014` p.14-15) — all matched exactly.

## Evaluation harness (RAGAS)

**ragas 0.2.15 (the version I'd originally pinned) doesn't import at all against a current
langchain install.** Its own `ragas.llms.base` unconditionally imports
`langchain_community.chat_models.vertexai.ChatVertexAI` — a module `langchain_community` removed
in its 0.4.x line, which is what pip resolves by default since ragas declares no upper bound on
any `langchain*` dependency. Ragas 0.4.3 (current) hit the identical import error for the identical
reason. Fix was pinning `langchain-community==0.3.31` (the last release before the module moved),
not chasing a different ragas version — the ragas version wasn't the variable that mattered here.

**ragas 0.4's newer `ragas.metrics.collections` API needs an `InstructorBaseRagasLLM`, and its
Anthropic support has a real gap**: the instructor adapter hardcodes
`InstructorModelArgs(temperature=0.01, top_p=0.1)` for every provider, but Claude's API rejects a
request that sets both `temperature` and `top_p` — and rejects an explicit `top_p=None` just as
strictly ("Input should be a valid number"), so there's no way to configure this away through
`llm_factory`'s public kwargs. Worked around by deleting the key from the constructed LLM's own
`model_args` dict after the fact (`del llm.model_args["top_p"]`) rather than trying to pass a
value through — confirmed live against the real API before trusting it, not just because the
types lined up.

**The default `max_tokens=1024` on that same judge truncated Faithfulness's structured-output JSON
mid-response** on an answer with many claims to verify — `InstructorRetryException: ... EOF while
parsing a list`, and it happened on item 22 of a live 45-item run, not in a quick smoke check.
Bumped to 4096. The real lesson wasn't the number, it was that a long-running scored batch job
which only saves results after every item succeeds throws away every prior item's real API spend
the moment one item fails for any reason — `run.py` now catches per-item exceptions and reports
partial results with the failures listed, not silently swallowed and not fatal to the whole run.

**A live RAGAS run is what caught a real citation-format bug, not the mocked test suite.** One
multi-document answer cited `[1, p.2]` and `[2, p.3]` — the model had copied the *numbered context
block index* `_format_context` printed for my own debugging readability, not the actual `doc_id`
the citation format asks for. A bracketed index sitting right next to a bracketed citation format
is exactly the confusion an LLM would make on a harder multi-source synthesis question; single-
source answers never triggered it because there was only one block to point at either way. The
citation verifier caught it correctly (both citations came back `verified: False`, since `"1"` and
`"2"` never match any real `doc_id`) — the safety net worked exactly as designed — but the root
cause was still worth fixing: dropped the numbered index from `_format_context` entirely, since
the model never needed it. Re-running the same question afterward produced eight `verified: True`
citations against the real document.

**11 of 45 golden items scored 0.00 across all four RAGAS metrics on the first full run — every
single one turned out to be a refusal, not a bad answer.** `run_ragas.py` didn't check
`result.refused` before handing the response to the judge, so "I don't have a source for that."
got scored against metrics built to evaluate a substantive, grounded answer. Filtering refusals
out (tracked and reported separately, not silently dropped) moved the aggregate from
faithfulness/relevancy/precision/recall of 0.771/0.615/0.538/0.778 to a materially more honest
0.791/0.851/0.673/0.912 — the first set of numbers wasn't "the system doing worse," it was the
measurement counting a correct behaviour as a failure. `record_fixtures.py` had the identical bug
for the same reason (copy-pasted before the fix existed) and got the same fix.
`test_regression.py`'s citation-presence assertion had the mirror-image version: it demanded a
citation from every non-unanswerable item, which would fail the CI gate on a legitimate refusal
instead of the actual regression the test exists to catch.

**The CI faithfulness threshold (0.5) was picked after seeing the real number, not before.** First
draft used 0.7, a round guess made before any data existed. The recorded 10-item CI subset's own
non-refused mean is 0.628 — six items, one of them a genuine 0.0 outlier that a 6-item average
can't absorb the way the full 34-item run does — so 0.7 would have failed the gate on the exact
data used to build it. 0.5 leaves room for that outlier and for ordinary run-to-run variance
(Claude doesn't run at temperature 0 here, and a borderline retrieval case has genuinely flipped
between answering and refusing across separate live runs of the same question) without the gate
firing on noise, while still catching an actual collapse in grounding.

## Eval-driven improvement (Phase 8)

**Chunk size and reranking were swept together, not tuned one at a time.** `scripts/sweep_chunk_size.py`
re-chunks and re-embeds the whole corpus at 300/500/800 tokens into a throwaway Chroma collection
(never the production one) at each size, with reranking on and off, and runs the same 20-question
retrieval benchmark against every variant:

| chunk size | rerank | chunks | hit-rate@5 | MRR |
|---|---|---|---|---|
| 300 | off | 1861 | 85% | 0.578 |
| 300 | on | 1861 | 85% | 0.717 |
| 500 | off | 1081 | 85% | 0.654 |
| 500 | on | 1081 | 90% | 0.747 |
| 800 | off | 635 | 90% | 0.752 |
| 800 | on | 635 | **95%** | **0.808** |

800+rerank won outright rather than by a coin-flip margin, and reranking improved every chunk
size it was tested against — larger chunks give the cross-encoder more surrounding context to
judge relevance from, at the direct cost of a coarser page-range citation. 500 tokens was the
original plan's number, picked before any of this data existed; 800 is what the sweep actually
rewarded, so `CHUNK_TARGET_TOKENS` moved from 500 to 800 and `rag.answer_question` now calls
`retrieve(..., rerank=True)` by default. Re-embedding the full corpus at the new size dropped the
chunk count from 1081 to 635 (fewer, larger chunks) and the retrieval benchmark against the
rebuilt *production* store confirmed the same 95%/0.808 the throwaway sweep predicted — the sweep
collection wasn't measuring something the real store then failed to reproduce.

**A live RAGAS re-run mid-flight is what caught that `evals/run_ragas.py` appends to
`reports/eval_history.csv`, not overwrites it** — I'd deleted the file before the "after" run to
get a clean single row, which also erased the pre-improvement baseline row that made "before vs.
after" a comparison instead of one number. Recovered the baseline via `git show HEAD:reports/eval_history.csv`
(it was already committed) before re-running, rather than losing the one thing the improvement
story depends on:

| | n scored | faithfulness | answer relevancy | context precision | context recall |
|---|---|---|---|---|---|
| before (500 tok, no rerank) | 34 | 0.791 | 0.851 | 0.673 | 0.912 |
| after (800 tok, rerank) | 39 | 0.829 | 0.811 | 0.790 | 0.968 |

Context precision moved the most (+0.117), which tracks — cutting retrieval noise is reranking's
whole job. Answer relevancy dipped slightly (0.851 to 0.811); left in rather than smoothed over,
since a metric moving the "wrong" direction after a change that helped on every other measure is
more informative than a report that only shows the numbers that agree with the story. (n scored
also rose, 34 to 39, mostly because reranking pulled more questions from refusal into an actual
answer — see `reports/failure_analysis.md`.)

**The same heading-misclassification bug bit twice, in two different date formats, six documents
apart.** The first instance (a bare `"08 July 2020"` press-release dateline, no punctuation)
was caught during Phase 2 chunk-quality review and fixed by excluding date-shaped lines from
`_is_heading` before the numbering check runs. `failure_analysis.md`'s investigation of golden
item `g08` found the same failure mode in `sarb_circular_19_2004`: a sentence-ending date with its
own full stop still attached (`"28 February 2005."`) didn't match the first fix's regex, which
anchored on a bare date with nothing after the year — so it still got classified as a heading, its
text still got dropped from the chunk, and the question that depended on it ("by what date were
comments due") still refused for the identical underlying reason. Widened `_DATE_LIKE` to accept
an optional trailing `.`; re-ingesting shifted 5 of 14 chunks in that document (`added 5, deleted 5,
unchanged 630` on `store.py --rebuild`), and the same question that refused before the fix now
answers "28 February 2005" with a verified citation. Two regression tests now guard both dates
independently, plus an integration test against the real PDF for each — a unit test against the
regex alone wouldn't have proven the second date's text actually survives extraction, the same gap
that let the first fix miss this one.

**RAGAS's own judge disagreed with manual inspection on at least three golden items, in both
directions of surprise.** `g16` scored answer relevancy 0.00 despite a fully correct, correctly-
cited answer ("unfair credit and credit-marketing practices," matching the reference exactly) —
most likely the judge penalising the answer's own hedge ("the context does not specify which
particular practices") as off-topic rather than as an honest scope caveat. `g41` scored context
precision 0.00 despite both retrieved-and-cited chunks being exactly the two the question needed,
ranked first and second out of five — a precision of 0 there is arithmetically implausible if the
judge had actually credited either top-ranked chunk as relevant. `g40` scored faithfulness 0.26,
the single worst score in the run, despite a specific quoted figure ("cannot exceed 9% of
applicable premiums") that a direct dump of the retrieved chunk text confirms is a verbatim, exact
match to the source PDF — not the shape of an actual hallucination. None of the three look like
retrieval or generation defects; they read as instances of RAGAS's claim-decomposition and
relevance-judging steps being noisier on longer, multi-claim, or gently-hedged answers than on
short, clean ones. Reported as judge artifacts in `reports/failure_analysis.md` rather than treated
as system bugs to chase — the fix for a noisy judge component is knowing which of its outputs to
distrust, not tuning a well-functioning RAG pipeline to please it.

**Two golden items refuse for a reason a fixed `k=5` retrieval call can't solve: the question
needs one chunk from each of two documents, and one of the two (`sarb_d10_2021_operational_resilience`,
only 3 chunks total) never makes the top 5 candidates because the other document's chunks score
higher for the query as written.** `g37` and `g44` both ask "which body do X and Y both cite" —
a single embedding of that sentence pulls hard toward whichever document's vocabulary the query
text happens to resemble more, and a 3-chunk document has fewer chances to be that closest match.
Query decomposition (retrieve once per named document, then merge context) would fix this
directly; not built this session, since Phase 8's scope was the chunk-size/reranking sweep, not a
retrieval-architecture change — tracked here as the concrete next improvement rather than folded
silently into "retrieval sometimes misses."

## Docker

**Volumes bind-mount the whole repo over the image's `/app` rather than baking corpus, chroma,
and the sqlite log/cache into the image.** All three are gitignored, locally-built artifacts
(`corpus/` and `chroma/` via `fetch_corpus.py` + `store.py --rebuild`; the sqlite files created on
first write) — a bind mount means a code change or a corpus re-ingest is visible on container
restart without a rebuild, and avoids the alternative failure mode of bind-mounting individual
files that don't exist yet on the host (Docker creates an empty *directory* at that path instead
of a file, which then breaks `sqlite3.connect()` inside the container). Mounting the whole
directory sidesteps that per-file gotcha entirely.

**`docker compose up -d` failed on port 8501 the first time, for a reason that had nothing to do
with the compose file** — a `streamlit run app/ops.py` I'd started directly on the host (via
`preview_start`, to grab an ops-dashboard screenshot for the README before Docker was verified)
was still holding that port. Not a compose bug; stopped the host process and the container came
up clean. Real trap worth naming: verifying app code locally and verifying its Docker packaging
are two different activities that can collide on shared ports if run back to back.

**Verified with an actual `docker compose up -d --build`, not just a file review**: all three
containers reached a running state, the api service's healthcheck (a Python `urllib` request
against `/health`, no `curl` in the slim base image) reported healthy before `chat`/`ops` started
via `depends_on: condition: service_healthy`, and a real `POST /ask` against the containerized API
returned a correctly-cited, verified answer — confirming the container's Python environment
resolves `chroma-hnswlib` and the rest of the compiled-extension dependency chain that needed
Windows Build Tools on the host (`python:3.12-slim`'s manylinux wheels cover it; no extra apt
packages were needed in the Dockerfile).

## Agent extension (Phase 12, stretch)

**Built to test one specific, already-diagnosed failure, not "agents are generally better."**
`reports/failure_analysis.md` found that two golden items (`g37`, `g44`) refuse because a
two-document comparison question embeds as one query, which under-retrieves whichever named
document has fewer chunks. `src/agent.py` adds exactly one capability on top of `rag.py`: after
retrieving, ask the model whether a named document is still missing, and if so issue one more
targeted retrieval call before answering (capped at 2 requeries, `MAX_STEPS=3`).

**Result on the 10 multi-doc golden items (`scripts/agent_eval.py` → `reports/agent_eval.md`):
0 refusals fixed, 0 introduced, at 2.76x mean cost and 1.77x mean latency.** Both `g37` and `g44`
did trigger the full 2 requeries each — the decision step correctly recognised the missing
document both times, which is the mechanism working as designed — but the run still refused both.

**A same-question re-run immediately afterward flipped `g44` from refused to a fully correct,
cited answer, with the exact same code and the exact same two requery steps.** Dumping the steps
log confirmed `sarb_d10_2021_operational_resilience` chunks were present in the merged context on
*both* runs — so the fix that matters (getting the missing document's chunks into context) is
working every time; what varies run-to-run is whether the final generation step commits to
synthesizing an answer from a larger, noisier merged context or falls back to refusing. This is
the same live non-determinism already documented in the RAGAS section (Claude isn't called at
temperature 0 here), just landing on a different part of the pipeline — the agent moves the
failure point from "can't find the right chunks" to "won't always commit to an answer once it has
them," which is progress on the diagnosed retrieval problem but not yet a clean fix, and isn't
being reported as one. Not pursued further this session (recalibrating for run-to-run variance
would mean re-running the batch until the numbers looked good, which is the opposite of an honest
measurement) — the negative-leaning result is reported as measured, per the plan's own instruction
that a measured non-improvement is worth more than a silently dropped feature.

## Pre-release audit

Findings from reading the repo back as an unsympathetic reviewer would, after v1.0.0 was tagged.
Four of these are defects I introduced and did not catch while building.

**The headline before/after RAGAS comparison was measuring two different things at once.** The
pre-improvement run scored 34 items, the post run 39, and I reported the mean of each side by side
as though that were a controlled comparison. It isn't: reranking pulled six previously-refused
questions into real answers and pushed one the other way, so the item set changed underneath the
average. A mean over a moving population shifts for two unrelated reasons, and only one of them is
a quality claim. Recomputing on the 33 items both runs scored (`scripts/paired_eval.py` →
`reports/paired_comparison.md`) takes faithfulness from a reported +0.038 to +0.023 with a per-item
split of 9 improved against 9 regressed — a sign test p of 1.00, which is to say nothing at all.
Context recall's +0.056 becomes +0.030 with 32 of 33 items unchanged; it was already at 0.939 and
had nowhere to go. Only context precision survives (+0.109 paired, 14 improved against 5, p=0.06),
and that is still short of conventional significance on 33 items — I keep it because it is the one
effect with a mechanism behind it rather than the one number that happened to move. What reranking
actually bought, unambiguously, is coverage: five net refusals became answers, and counting a
behaviour change needs no significance test. The README now leads with that instead of four green
arrows. Same correction applies to the retrieval benchmark: 85%→95% is 17/20 versus 19/20, two
questions, Wilson intervals 64–95% and 76–99%, and I should not have printed it as a clean 10-point
gain without saying so.

**The response cache was keyed on question text alone, so the Phase 8 config change never
invalidated it.** Adopting chunk_size=800 + reranking rebuilt the vector store and changed what
retrieval returns, but every already-cached answer kept its key and kept being served. The eval
reports would have described the new configuration while the running API returned pre-improvement
answers indefinitely, with nothing anywhere to indicate a mismatch. The evals themselves were never
affected (`run_ragas.py` calls `rag.answer_question` directly, bypassing `obslog.timed_answer` and
therefore the cache) and neither was `reports/perf.md`, which deletes the cache file before
measuring — which is exactly why this survived: every measurement path happened to route around the
bug, and only the product carried it. The key now includes a fingerprint of both models, the
collection, the chunk size, and a hash of the system prompt, so any change that alters what an
answer would be also alters the key. Old rows become unreachable rather than being deleted; that's
the intended invalidation.

**Prompt-injection flagging silently stopped working on repeat attempts.** `cache.get_cached`
reconstructed its `RAGResult` with `flagged_injection=False` hardcoded, and `timed_answer` consults
the cache *before* `rag.answer_question`, where the detector actually runs. So the first time an
injection attempt arrived it was flagged, and every identical attempt afterwards logged clean.
Repeat attempts are the entire traffic signature of someone probing a system, so the detector went
quiet on precisely the case it exists to surface — while `reports/security_notes.md` claimed in
writing that its purpose was giving an analyst visibility of injection attempts in the ops
dashboard. Reproduced end to end before fixing: same question twice, `flagged_injection` True then
False. The flag is a property of the question, not of the cached answer, so it's now recomputed on
both paths in `timed_answer` and deliberately not stored in the cache at all.

**The Docker images were 9.73GB, and 3.4GB of that was CUDA.** `torch` is an unpinned transitive
dependency of sentence-transformers, and on Linux pip's default resolution pulls the CUDA build:
2.7GB of `nvidia-*` packages plus 691MB of triton, in images whose only inference is two MiniLM
models on CPU, in a compose stack with no GPU. I verified it by measuring inside the built image
rather than assuming (`du -sh site-packages/*`) — site-packages alone was 5.9GB. Installing
CPU-only torch from PyTorch's own index ahead of `requirements.txt` fixes it for all three images:
rebuilt, `regrag-api` measured 3.55GB and `regrag-chat`/`regrag-ops` 3.2GB each, down from 9.73GB —
confirmed by re-measuring the built images, not assumed from the fix alone. Both embedding models
are now baked into the API image too, so a container starts without reaching HuggingFace; `HF_HOME`
deliberately points outside `/app`, because compose bind-mounts the repo over `/app` and would
otherwise shadow the cache and quietly undo the whole step.

**The CI faithfulness floor had drifted into being unfirable.** 0.5 was calibrated when the
recorded subset's mean was 0.628. After the config change that mean is 0.805, and a floor a third
of the way below the value it guards is not a regression gate. Raised to 0.65. The gate runs
against frozen fixtures so CI is deterministic either way — the floor only bites when someone
regenerates fixtures, which is the moment it should.

## Source authority and currency metadata

Two limitations named in the README ("the corpus mixes levels of legal authority and the system
treats them as equal," "nothing models supersession") had a straightforward fix sitting unused in
`corpus/manifest.json`: `issuer`, `title`, and (after this change) `document_type` and
`is_third_party` were already recorded per document and reached none of retrieval, the prompt, or
the citation. Two additions, both kept deliberately outside the LLM's discretion:
`_format_context` now prints each context block's type, year, issuer and title as plain,
unbracketed text (never bracketed like `[doc_id, p.X]`, so the model can't copy fields from it into
a citation the way it once copied a numbered index — see the RAG core section above); `_source_notices`
generates a fixed sentence whenever a cited source is third-party commentary or a Circular,
whether or not the model's own prose mentions it — same reasoning as `INSUFFICIENT_CONTEXT_PHRASE`
being an exact string rather than trusted free-text: a compliance disclosure shouldn't depend on
the model remembering to write it on any given call.

**Kept as a separate `RAGResult.source_notices` field, never text appended onto `answer`.** RAGAS's
faithfulness metric decomposes `answer` into claims and checks each against the retrieved chunk
text — the exact mechanism that once scored a correct refusal as 0.0 faithfulness (see the
Evaluation harness section) because the text it was given didn't match what the metric expected to
grade. A disclaimer sentence this codebase generated, not the model, would fail that same check for
the same reason: it isn't *in* the retrieved chunks, so RAGAS would score it as an unsupported
claim. Every existing consumer of `RAGResult.answer` — the cache key, citation extraction, the
refusal check, RAGAS scoring — keeps reading exactly what the model generated.

**The currency notice is scoped to one document type, not a year cutoff.** Both circulars in this
corpus are from 2004, predating the Directive/Guidance Note split the newer SARB instruments use,
and I know Basel III superseded the Basel II-era rules those circulars describe. The National
Credit Act is from 2005 and is still the current governing statute, amended in place rather than
replaced — a blanket "older than N years" rule would have flagged it as dated right alongside the
circulars, which would be wrong. The notice also doesn't name a specific successor document; this
corpus doesn't contain one, and inventing one would be a fabrication risk for a compliance tool.

**Verifying this against the live model caught a second, unrelated bug: the manifest's own document
titles were paraphrases, not the documents' real titles.** The first version of this fix labeled
`ncr_guideline_sept_2025_debt_counsellors` as "Guideline (September 2025): Debt Counsellor Contact
Information Requirements" — a reasonable-sounding description I wrote, not what the document
actually says about itself. Tested against golden item `g24` ("what is required under Guideline
004/2025"), it still refused, because "004/2025" appears nowhere in that paraphrase. Opened the
actual PDF: its own cover page reads "...004/2025 SEPTEMBER 2025." All four NCR guideline titles
were paraphrased the same way; extracted the real title text from each PDF's cover page directly
and re-verified the other 15 manifest titles against their source PDFs too, rather than assuming
only the ones I'd already found a problem with were wrong. `g24` now answers correctly and cited.

**Measured with the same paired discipline as the chunk-size change, and the RAGAS means don't move
— which is the correct result for what this fix actually is.** `scripts/paired_eval_metadata.py` →
`reports/paired_comparison_metadata.md`: every paired delta across all four metrics is inside sign-
test noise (p ranging 0.36–1.00). This isn't a retrieval or generation-quality change; it's a
disclosure change, and RAGAS's four metrics have no dimension for "did the answer correctly
attribute what kind of document this is." One item entered the scored set (`g24`, above) and one
left it: `g45` ("name an objective that appears in both the Act and the Notebook brochure") started
refusing. Checked directly against `src.retrieve.retrieve` rather than assumed: the same five
chunks come back before and after this change, and none of them is the National Credit Act's own
text — only a DTIC brochure describing it. Before this fix, the model answered anyway, treating the
brochure's summary as equivalent to the Act's own words; told explicitly that the source is a
"Regulator explainer brochure," it now correctly declines to attribute a claim to "the Act itself"
when the Act's own text was never retrieved. That's the fix working as designed, and it shows up as
a debit in a refused/answered count — exactly the kind of thing a single metric misses, which is
why this got checked by hand rather than left as an unexplained regression in a table.

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

**After installing Build Tools, torch itself still wouldn't load** — a second, unrelated system
policy: Windows Smart App Control (running in Evaluation mode) was blocking
`torch_global_deps.dll` from loading at all, `OSError: [WinError 4551] An Application Control
policy has blocked this file`. Not something to silently route around — Smart App Control is a
real security boundary, so this stayed blocked until it was explicitly turned off in Windows
Security rather than worked around some other way.

**chromadb 0.5.23 pulls in `googleapis-common-protos` (via its OpenTelemetry OTLP exporter),
which floors `protobuf>=6.33.5`, while Streamlit 1.40.2 caps `protobuf<6`.** No single version
pin resolves that — it's a genuine floor-vs-ceiling conflict between the two packages' own
dependency trees, not something `pip install` backtracking can fix by trying harder. Streamlit
1.62.0 relaxed its cap to `<8,>=5.26.1`, which resolves it — except newer Streamlit then wants
`starlette>=0.46`, which is above FastAPI 0.115.6's own `starlette<0.42` ceiling, so FastAPI had
to move to 0.141.1 too. Three packages, one dependency graph — `pip check` after every bump
until it came back clean.

**chromadb's telemetry client raises on every call** (`capture() takes 1 positional argument but
3 were given`) — a real version incompatibility, not a disabled-flag issue: chromadb 0.5.23's
`Posthog.capture()` calls the library's old three-argument signature
(`posthog.capture(distinct_id, event, properties)`), but chromadb only floors `posthog>=2.4.0`
with no ceiling, so pip resolved the latest 7.x, which replaced that with a single event-object
argument. Setting `anonymized_telemetry=False` in `store.py` didn't fix it — chromadb's own
`capture()` method calls `posthog.capture()` unconditionally regardless of that setting, so the
crash (caught internally, harmless) happened either way. Pinning `posthog==3.7.0` in
requirements.txt is the actual fix; `anonymized_telemetry=False` stayed in `store.py` anyway,
since a local research tool has no reason to phone home even with a compatible posthog version.
