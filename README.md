# regrag-sa-finance

A retrieval-augmented assistant that answers questions about South African financial regulation
(SARB prudential directives, the National Credit Act, FSCA conduct standards, IFRS 9) from a fixed
local corpus of 19 PDFs. Every factual claim in an answer carries an inline `[doc_id, p.X]`
citation checked against the pages the system actually retrieved, and the system refuses rather
than answers when the retrieved context cannot support a response. The eval harness is the point
of the project rather than a checkbox after the fact: chunking and retrieval decisions were made
by sweeping alternatives against a retrieval benchmark and a RAGAS-judged golden set, so the
numbers below are what that sweep produced, not what was assumed going in.

## Eval results

The change under test is `chunk_size=800` with cross-encoder reranking, replacing 500-token chunks
with no reranking. Both configurations were measured on the same 20-question retrieval benchmark
and the same golden set.

| | chunk size | rerank | hit-rate@5 | MRR |
|---|---|---|---|---|
| before | 500 | off | 85% (17/20) | 0.654 |
| **after** | **800** | **on** | **95% (19/20)** | **0.808** |

The hit-rate line is two questions on a 20-question benchmark, and the Wilson intervals (64 to 95
per cent, 76 to 99 per cent) overlap too heavily for that difference to count as established. MRR
is the better-powered signal in the same data, since it moves on where the right chunk ranks
rather than only on whether it cleared a cutoff.

RAGAS was run on the golden set's answerable items with `claude-haiku-4-5` as both generator and
judge. The two runs scored different item sets, since reranking pulled six previously refused
questions into real answers and pushed one the other way, so the raw means below are not directly
comparable on their own. The paired comparison on the 33 items common to both runs sits alongside
them for that reason.

| metric | before (all, n=34) | after (all, n=39) | paired Δ (n=33) | improved / regressed | sign test |
|---|---|---|---|---|---|
| faithfulness | 0.791 | 0.829 | +0.023 | 9 / 9 | p = 1.00 |
| answer relevancy | 0.851 | 0.810 | −0.041 | 10 / 11 | p = 1.00 |
| context precision | 0.673 | 0.790 | **+0.109** | **14 / 5** | p = 0.06 |
| context recall | 0.912 | 0.968 | +0.030 | 1 / 0 | p = 1.00 |

Only context precision holds up under that pairing, and even it falls short of conventional
significance. The faithfulness and recall gains in the raw means are mostly composition effects.
Once the item set is held fixed, faithfulness splits 9 to 9 across items, and recall was already
at 0.939 with 32 of 33 items unchanged, so there was little room left to move. What reranking
demonstrably bought was coverage: five net refusals became answered questions, a count of
behaviour changing rather than an estimated mean, so it needs no significance test to stand. The
precision gain earns more trust than its p-value alone would suggest, because it has a mechanism
behind it. Dropping loosely similar chunks is exactly and only what a cross-encoder does, so a
precision-shaped result is the prediction here, not a number that happened to move.

I kept the new configuration on that basis. The full working is in
[`reports/paired_comparison.md`](reports/paired_comparison.md), the sweep across 300, 500 and 800
tokens with reranking on and off is in [`reports/improvement_log.md`](reports/improvement_log.md),
and the run itself is in [`reports/eval_summary.md`](reports/eval_summary.md). The ten
worst-scoring golden items are diagnosed individually in
[`reports/failure_analysis.md`](reports/failure_analysis.md); three of them turned out to be RAGAS
judge artifacts rather than real defects, which I confirmed by dumping the retrieved chunk text
against the model's quoted claims rather than trusting the score.

A response cache, exact-match on normalised question text so a rephrased question is still a live
call, cuts median latency from 4489ms and $0.00365 per query to 1ms and $0. Measured in
[`reports/perf.md`](reports/perf.md).

## Architecture

```mermaid
flowchart LR
    subgraph offline["offline, one-time"]
        PDF[19 corpus PDFs] --> Extract["PyMuPDF extraction<br/>+ heading detection"]
        Extract --> Chunk["heading-aware chunking<br/>800 tok, 75 overlap"]
        Chunk --> Embed["MiniLM embeddings"]
        Embed --> DB[(ChromaDB<br/>635 chunks)]
    end

    subgraph online["per question"]
        Q[question] --> C{cache hit?}
        C -->|yes| Cached[cached answer]
        C -->|no| Bi["bi-encoder search<br/>top 20"]
        DB --> Bi
        Bi --> Rerank["cross-encoder rerank<br/>top 5"]
        Rerank --> LLM["Claude Haiku<br/>cited answer or refusal"]
        LLM --> Verify["citation verification<br/>against retrieved pages"]
        Verify --> Log[(SQLite: query log<br/>+ response cache)]
        Verify --> Answer[answer + citations]
    end
```

`api/main.py` is a FastAPI service (`/ask`, `/health`, `/stats`, rate-limited) sitting in front of
this pipeline. `app/chat.py` and `app/ops.py` are the Streamlit chat UI and observability
dashboard, each a separate deployable process that talks to the API over HTTP rather than
importing the RAG code directly.

## Run locally

Requires Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate  # or `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

`chroma-hnswlib` compiles a C extension on install. On Windows this needs the Microsoft C++ Build
Tools (Visual Studio Installer, "Desktop development with C++" workload). Linux and macOS wheels
are usually prebuilt, so this step is a Windows-only cost.

Copy `.env.example` to `.env`. There are three provider options, picked by `LLM_PROVIDER`.

- `anthropic` (default): set `ANTHROPIC_API_KEY`. This is what generated every number above.
- `openai`: set `OPENAI_API_KEY`. Not benchmarked here, since the eval harness assumes Claude as
  judge.
- `ollama`: no key, fully local, needs `ollama pull llama3.1:8b` and the daemon running. Quality
  will differ from the benchmarked configuration, but it runs the pipeline at zero API cost.

```bash
python -m scripts.fetch_corpus       # downloads the 19 PDFs, verifies against pinned SHA-256
python -m src.chunking               # chunks, embeds into a fresh reports/chunk_quality.md
python -m src.store --rebuild        # builds the ChromaDB collection
python -m evals.retrieval_bench      # reports/retrieval_bench.md
python -m evals.run_ragas            # reports/eval_summary.md + reports/eval_history.csv

uvicorn api.main:app --reload
streamlit run app/chat.py
streamlit run app/ops.py
```

`ruff check .`, `ruff format --check .` and `pytest -q` should all pass clean. There are 79 tests,
all run against a mocked LLM, so none of them need an API key.

### Docker

```bash
docker compose up --build
```

This brings up the API on port 8000, the chat UI on 8501, and the ops dashboard on 8502 as three
containers. The compose file bind-mounts the repo over each image's `/app`, so `corpus/`,
`chroma/`, and the SQLite log and cache files, all gitignored and built locally by the commands
above, are visible to the containers without being baked into the image. Both Dockerfiles install
CPU-only torch ahead of `requirements.txt`, since sentence-transformers' default resolution pulls
the CUDA build on Linux. That inflated the first version of these images to 9.73GB for two MiniLM
models that only ever run on CPU here. The API image now measures 3.55GB and the UI images 3.2GB
each.

## What is not built

The API and both Streamlit apps run locally and in Docker, and there is no hosted URL. Render or
Streamlit Cloud deployment was scoped out of this project on purpose, not forgotten.

Two multi-document comparison questions in the golden set still fail. A query like "what body do
directive X and directive Y both cite" embeds as one vector, and the document with fewer chunks
loses to whichever document's vocabulary the embedding happens to resemble more closely.
`src/agent.py` adds a bounded retrieve-decide-requery loop meant to fix exactly this. Tested
against the ten multi-document golden items, it fixed zero of the two target cases while tripling
cost and adding real latency, recorded in
[`reports/agent_eval.md`](reports/agent_eval.md). A same-question rerun afterward flipped one
target case from refusal to a correct answer with identical code, which points to run-to-run LLM
non-determinism rather than a fixable bug. The full result is in the Agent extension section of
[`DECISIONS.md`](DECISIONS.md). Query decomposition, a separate retrieval call per named document,
looks like the more promising fix, and it is not built.

The API has no authentication, and its rate limiting is per-process and held in memory, so a
distributed client or a shared NAT defeats it easily. Neither gap is assumed away silently: both
are written up in [`reports/security_notes.md`](reports/security_notes.md).

## Limitations

The golden set has not been reviewed by a human. I drafted the 55 items and their reference
answers myself, and Claude both answers and judges them, which is a closed loop and the weakest
foundation under every number on this page. A reference answer that is subtly wrong produces a
confidently wrong score, and nothing in the harness would catch it. The retrieval set is in better
shape, since its page labels were audited against the source PDFs after the first benchmark run
scored only 55 per cent, which is how eleven mislabelled `golden.jsonl` page references were found
and fixed. But that audit checked where the answer lives, not whether the reference answer is
correct in the first place. Independent review is the highest-value thing this project is still
missing.

The corpus mixes levels of legal authority, and until recently the system treated all of them as
equal. It contains primary legislation in the National Credit Act, subordinate instruments issued
under it in the SARB directives under section 6(6) of the Banks Act, explicitly non-binding
guidance in Guidance Note 3/2025 under section 6(5), and third-party commentary in PwC's IFRS 9
practical guide. `corpus/manifest.json` recorded issuer, title, year and category for every
document from the start, but none of it reached retrieval, the prompt, or the citation. Every
context block now carries its document's real type, year, issuer and title, extracted from each
PDF's own cover page rather than paraphrased. The first version of this fix used paraphrased
titles, and a live test against golden item `g24` caught the gap, since the paraphrase had dropped
the document's own reference number. Citing third-party commentary or a superseded instrument type
now also attaches a fixed, code-generated disclosure. This was measured the same paired way as the
chunk-size change, in
[`reports/paired_comparison_metadata.md`](reports/paired_comparison_metadata.md): no RAGAS metric
moves outside noise, because it is a disclosure fix rather than a retrieval or generation-quality
one, and `g24` flips from refusal to a correct answer for exactly the diagnosed reason. The full
story is in the Source authority and currency metadata section of
[`DECISIONS.md`](DECISIONS.md). What this does not do is rank authority. A directive is not marked
more binding than a guidance note anywhere the code enforces it, only named as one or the other.

Supersession is still not modelled, beyond flagging Circulars as a document type that this
corpus's newer instruments have superseded as a category. The corpus spans 2004 to 2026, and
Directive D3/2023 states in its own text that it replaces Directive 5/2017, a specific successor
claim this system does not verify or surface. That is deliberate: the corpus does not contain
Directive 5/2017, so confirming or naming that relationship is not something the retrieved text
can actually support.

Retrieval is measured on a small and unbalanced corpus. There are 635 chunks in total, and 333 of
them, 52 per cent, come from the National Credit Act alone. Benchmark figures from a corpus this
size and this skewed should not be read as predicting behaviour at a realistic scale.

Every RAGAS score reported here comes from Claude-haiku-4-5 judging Claude-haiku-4-5's own output.
[`reports/failure_analysis.md`](reports/failure_analysis.md) confronts that same-family judge bias
directly: three of the ten worst-scoring golden items turned out to be judge artifacts rather than
real defects, which I only found by checking them by hand rather than trusting the score.

This tool reports what a document says. It does not give legal advice, it does not tell you
whether a document is still in force, and the system prompt says so on every response.

## Further reading

[`DECISIONS.md`](DECISIONS.md) is a running log of what actually happened while building this: the
real bugs, the numbers that did not move the way I expected, the dependency conflicts and how they
were actually resolved. I have kept it as it was written at the time rather than cleaning it up
into a tidier retrospective.
