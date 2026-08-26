# Failure analysis

Ten golden-set items, picked from the post-improvement RAGAS run (`reports/eval_summary.md`,
chunk size 800 + reranking): the six items that triggered `rag.py`'s own refusal rule, plus the
four lowest-scoring items that did produce an answer. Each one below was re-run individually
against the live pipeline with the actual retrieved chunks dumped alongside the answer, rather
than classified from the RAGAS numbers alone — three of the ten turned out to be scoring
artifacts once the real output was checked, which the numbers by themselves didn't distinguish
from a genuine defect.

| id | failure mode | fixed this session? |
|---|---|---|
| g03 | retrieval miss | no — tracked below |
| g08 | chunking artifact (heading misclassification) | **yes** |
| g16 | judge artifact | n/a — not a system defect |
| g24 | retrieval mis-ranking / missing doc-number metadata | no — tracked below |
| g28 | retrieval miss (chunk granularity) | no — tracked below |
| g29 | chunking artifact (fact split across chunk boundary) | no — tracked below |
| g37 | retrieval miss (multi-doc, low-chunk-count document) | no — tracked below |
| g40 | judge artifact | n/a — not a system defect |
| g41 | judge artifact | n/a — not a system defect |
| g44 | retrieval miss (same root cause as g37) | no — tracked below |

## Retrieval misses

**g03** — *Why did Directive 5/2017 need to be updated, according to Directive D3/2023?*
Expected source: `sarb_d3_2023_accounting_provisions_ifrs9` p.2. The correct document never
appears anywhere in the top-5 retrieved candidates — the query pulled `fsca_rdr_2014` and
`nca_act_34_2005` instead, both topically unrelated. This was already a known gap going into
Phase 8: `reports/retrieval_bench.md`'s own audit flagged this exact document ("MiniLM just
doesn't connect 'which earlier directive does X replace' to prose naming 'Directive 5/2017'") as
one of four genuine retrieval misses that survived the labeling-bug cleanup. Reranking doesn't
help here because the correct chunk isn't in the bi-encoder's candidate pool to begin with —
reranking can only reorder what retrieval already surfaced. Correctly refused rather than
answering from the wrong document, which is the system doing the right thing given a genuine
retrieval failure. **Fix, not built this session:** a keyword/BM25 fallback alongside the
embedding search would likely catch this — "Directive 5/2017" is a literal string a lexical match
would find regardless of embedding distance.

**g28** — *What did Proposal J of the Retail Distribution Review address regarding intermediation
and outsourced services?* Expected source: `fsca_rdr_intermediary_segmentation_2019` p.2. That
document's p.2-3 chunk IS retrieved (rank 2 of 5), but it's a general 2018-developments summary
that doesn't name "Proposal J" specifically — the chunk containing that exact proposal is one of
this 5-chunk document's other four, none of which make the top 5 once four slots go to
`fsca_rdr_2014` (2014, a different but related document with much more total content competing
for the same query terms). Correctly refused. **Fix, not built this session:** document-level
retrieval (top-k *documents*, then top-chunk-per-document) would guarantee at least one chunk
from the smaller, more specifically relevant document instead of letting a larger document's
volume crowd it out.

**g37 / g44** — both ask "which SARB document(s) X and Y have in common" across exactly two
named documents. In both cases, `sarb_d10_2021_operational_resilience` (only 3 chunks total)
never appears in the top 5 — the other named document's chunks dominate every slot because a
single-sentence query embedding resembles whichever document's vocabulary is closer, and a
3-chunk document simply has fewer chances to be that closest match. Both correctly refused rather
than answering from only one of the two documents and guessing at the other. **Fix, not built
this session:** query decomposition — split "compare X and Y" into two retrieval calls, one per
named document, then merge the results into one context — would resolve this class of question
directly, since it stops asking one embedding to do the job of two.

## Chunking artifacts

**g08** — *By what date were comments due on the proposed amendments to hybrid capital instrument
rules in Banks Act Circular 19/2004?* This is the one fixed this session. The correct chunk
(`sarb_circular_19_2004_capital_hybrid_instruments` p.1-2) was retrieved at rank 1, but its
`section` metadata was literally `"28 February 2005."` — the exact deadline the question asks
for — because `_is_heading` misclassified that sentence-ending date (with its own full stop) as a
section heading, the same failure mode already fixed once this session for a bare, unpunctuated
date (`"08 July 2020"` on a press release, see `DECISIONS.md`). A heading's text is dropped from
the chunk body and kept only as metadata, so the model saw "comments due by not later than
[nothing]" in the paragraph text and correctly refused rather than guessing. Root cause: the first
fix's `_DATE_LIKE` regex anchored on a bare date with nothing following the year, which a
trailing `.` breaks. Fixed by widening the regex to accept an optional trailing period; re-ran
`store.py --rebuild` (5 of this document's 14 chunks changed), and the same question now answers
"28 February 2005" with a verified citation. Two unit tests and one integration test (against the
real PDF) now guard both date shapes independently — see `tests/test_ingest.py`.

**g29** — *In what year did the FSB carry out the technical work to determine which activities
intermediaries were remunerated for?* The retrieved chunk (`fsca_rdr_intermediary_segmentation_2019`
p.2-3, rank 1) contains the sentence itself almost verbatim — the model quotes it directly in its
answer — but the specific year ("during the course of 2016") isn't present in that chunk's text,
only the description of the work. Given this document has only 5 chunks total at an 800-token
budget, the year likely sits in an adjacent sentence that landed in a neighbouring chunk instead,
or in a footnote stripped during boilerplate/ToC cleanup. Correctly refused on a partial answer
rather than stating the FSB "carried out this work" without a year and implying more confidence
than the retrieved text supports. **Fix, not built this session:** would need the actual PDF page
checked directly to confirm whether the year is a chunk-boundary casualty or a footnote-stripping
casualty before proposing a specific change — flagged here rather than guessed at.

**g24** — *What is a debt counsellor required to keep up to date with the NCR under Guideline
004/2025?* The correct chunk (`ncr_guideline_sept_2025_debt_counsellors` p.2-3, containing "Every
Debt Counsellor must notify the NCR immediately upon any change in their contact details") IS
retrieved, but ranks third — behind two chunks from a different, unrelated document
(`ncr_guideline_feb_2026_clearance_certificates`) that happen to literally mention the string
"Guideline 004/2025" by name in their own text, without containing its actual content. Guideline
numbers aren't stored as chunk metadata anywhere in the pipeline, so both the reranker and the
model have to infer document identity from surface text alone — and a document that name-drops
"004/2025" while discussing something else outranks the document that actually *is* 004/2025 but
never states its own number inline. Correctly refused rather than answering from the
wrong-but-similar-sounding document. **Fix, not built this session:** extract each document's own
official number/title at ingestion time and inject it into `_format_context`'s per-chunk header
(alongside `doc_id` and page), so "this chunk is from Guideline 004/2025" is available to the
model directly instead of needing to be inferred from prose that may or may not state it.

## Judge artifacts (not system defects)

Three items scored low enough on RAGAS metrics to be flagged automatically, but manual
inspection of the actual retrieved chunk text and the actual answer shows no real defect in
either retrieval or generation:

**g16** (answer relevancy 0.00) — answer: *"the Act prohibits 'certain unfair credit and
credit-marketing practices' [nca_notebook_brochure, p.1]"* — matches the reference answer
("Unfair credit and credit-marketing practices") exactly, correctly cited, citation verified.
The likely explanation is that the judge penalised the answer's honest hedge ("the context does
not specify which particular practices are prohibited") as evidence of not addressing the
question, rather than crediting it as an accurate scope caveat given what was actually retrieved.

**g41** (context precision 0.00) — answer correctly names both Acts (Financial Sector Regulation
Act and Financial Markets Act, 2012), each with a verified citation, drawn from the top two of
five retrieved chunks — exactly the two chunks the question needs, ranked first and second. A
precision score of exactly 0 is arithmetically implausible if the judge credited either top-ranked
chunk as relevant to the reference answer; this reads as a judge-side scoring failure rather than
a retrieval-quality signal.

**g40** (faithfulness 0.26, the single worst score in the run) — the answer's specific quoted
figure, *"cannot exceed 9% of applicable premiums if all binder functions are performed,"* was
checked directly against the retrieved chunk's raw text (dumped separately from the RAGAS
pipeline) and is a verbatim match to the source PDF — not a plausible-sounding invention. The
underlying chunk text does have some rough edges from PDF extraction (a few sentences read as
fragments, missing a leading clause — a known cost of the font-size/numbering heuristic in
`ingest.py` rather than a real layout model), which may be making the judge's claim-by-claim NLI
check less reliable on this particular answer, but the answer itself is faithful to its source.

**Why report these as judge artifacts instead of quietly excluding them:** a RAG project's
headline claim is "faithfulness is measured, not assumed" — silently dropping the inconvenient
scores would undercut that same claim. The honest version is that an LLM-as-judge metric has its
own failure modes, most visible on longer or gently-hedged answers, and telling those apart from
real regressions requires spot-checking the underlying text rather than trusting the aggregate
number alone. That's also why `evals/test_regression.py`'s faithfulness floor (0.5) is deliberately
loose — a threshold in the space of scores demonstrated to include this many false negatives sets
its own bar to "won't fire on judge noise," not "matches the true faithfulness rate."
