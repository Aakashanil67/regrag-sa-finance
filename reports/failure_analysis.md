# Failure analysis

Source: `reports/runs/dev-a51410250c96.json` — the corrected-corpus development baseline
(`python -m evals.run_release --split dev --label corrected-corpus-baseline`), run after Tasks
1–7 of the hardening pass. 57 development items, retrieval `k=5`, reranking on.

**Headline numbers** (see `reports/eval_summary.md` for the full table): answerable answer rate
34/47 (72%), unanswerable refusal recall 10/10 (100%), citation-contract pass rate 44/57 (77%),
verified-citation rate 57/57 (100%). Every citation that reaches the user is verified — that's the
Task 1 contract doing its job — but 13 of 47 answerable items refused rather than answering.

## Method

For each of the 13 refused answerable items, the retrieved chunk IDs recorded in the run artefact
were cross-checked against the golden item's expected `(doc_id, page)` to separate two distinct
failure modes that a refusal count alone doesn't distinguish:

- **Retrieval miss**: the expected page never appeared among the 5 retrieved chunks. No citation
  contract could have passed — the context genuinely wasn't there.
- **Retrieved but still refused**: the expected page *was* retrieved, and the model still failed
  to produce a compliant, cited answer from it.

One accepted limitation of this analysis: `RAGResult.answer` only ever holds the validated answer
(`INSUFFICIENT_CONTEXT_PHRASE` on any refusal) — the raw model text that actually triggered
`MALFORMED_REFUSAL`, `UNCITED_LINE`, or `MISSING_CITATION` is deliberately not persisted (Task 1's
design: invalid model text must never reach a cache, log, or report). That means this analysis can
name *which* structural rule fired and *whether the right context was present*, but not the exact
wording that tripped the rule. Distinguishing "the model hedged mid-answer" from "the model wrote
a real second uncited sentence" would require a separate, explicit debug-only capture path — not
worth adding for a document already scheduled to change from write-once JSON to something bigger.

## Results

| id | type | expected source | retrieval | refusal reason |
|---|---|---|---|---|
| g01 | factual | `sarb_g3_2025_climate_disclosures` p.1 | hit | uncited_line |
| g03 | factual | `sarb_d3_2023_accounting_provisions_ifrs9` p.2 | **miss** | malformed_refusal |
| g15 | factual | `nca_act_34_2005` p.40 | hit | missing_citation |
| g16 | factual | `nca_notebook_brochure` p.1 | hit | malformed_refusal |
| g28 | factual | `fsca_rdr_intermediary_segmentation_2019` p.2 | hit | malformed_refusal |
| g29 | factual | `fsca_rdr_intermediary_segmentation_2019` p.2 | hit | uncited_line |
| g31 | factual | `fsca_tcf_2011` p.1 | hit | uncited_line |
| g34 | factual | `ifrs9_issued_2021` p.18 | **miss** | malformed_refusal |
| g35 | factual | `ifrs9_issued_2021` p.22 | hit | malformed_refusal |
| g37 | multi-doc | `sarb_g3_2025_climate_disclosures` p.1 + `sarb_d10_2021_operational_resilience` p.1 | **miss** (neither doc retrieved) | malformed_refusal |
| g38 | multi-doc | `nca_notebook_brochure` p.1 + `ncr_guideline_june_2025_credit_info` p.2 | hit (both) | uncited_line |
| g44 | multi-doc | `sarb_d8_2023_threshold_amounts` p.1 + `sarb_d10_2021_operational_resilience` p.1 | hit (both) | malformed_refusal |
| g57 | factual | `sarb_c1_2026_status_of_circulars` p.1 | hit | malformed_refusal |

**3 retrieval misses** (g03, g34, g37) — the correct page genuinely never reached the model.
**10 retrieved-but-refused** (the rest) — the correct context was in front of the model and it
still failed the citation contract.

That 10-of-13 split is the headline finding: most of this baseline's refusals are not a retrieval
problem. Fixing retrieval quality further would not move the answerable answer rate as much as
the raw count suggests.

## Retrieval misses — one concrete fix or accepted limitation each

**g03** — *Why did Directive 5/2017 need to be updated, according to Directive D3/2023?* Expected
`sarb_d3_2023_accounting_provisions_ifrs9` p.2. The query names a different, older directive
(5/2017) that isn't itself in the corpus; embedding similarity likely favours other D3/2023
passages that don't mention it. **Accepted limitation** — no chunk-size or reranking change fixes
a query about content the corpus only mentions in passing on one specific page; a query rewrite
step (expand "Directive 5/2017" mentions before embedding) is out of scope for this release.

**g34** — *Under the 2021 issued IFRS 9 text, what three categories does an entity classify
financial assets into?* Expected `ifrs9_issued_2021` p.18 (paragraph 4.1.1). Retrieved pages
1, 1–2, and 61 from the same document — close in vector space (all discuss financial-instrument
classification generally) but not the specific paragraph. **Concrete fix candidate**: this is
exactly the kind of miss reranking is supposed to catch; worth checking post-release whether
`RERANK_CANDIDATE_POOL_SIZE` (currently 20) is wide enough for a 188-page document, since the
correct page competes against many topically-similar candidates from the same source.

**g37** — *Which SARB documents were issued under section 6(5) versus section 6(6) of the Banks
Act?* A comparison question against two named documents; neither one's specific page was
retrieved. This is the exact failure mode `src/agent.py` exists to test (a comparison query
under-retrieves a named document) — plain RAG has no second retrieval pass to correct it.
**Accepted limitation for this release**: the spec explicitly defers agent tuning until the core
plain-RAG release passes (see `regrag-release-hardening-spec.md`, scope boundaries).

## Retrieved-but-refused — one observation, not eleven repeated notes

All ten share the same shape: full context present, refusal anyway, mostly `malformed_refusal`
(7 of 10) rather than `uncited_line` (3) or `missing_citation` (1). `malformed_refusal` fires when
the model's raw text contains the exact refusal sentence *plus other text* — consistent with the
model adding a hedge or caveat around a correct citation rather than returning the refusal
sentence verbatim. Whether that hedge was appropriate caution or an unnecessary reflex isn't
answerable from the structural data alone (see Method, above); it would need either a debug-only
raw-text capture or a manual side-by-side prompt test, both out of scope for this pass. **Accepted
limitation**: the citation contract is deliberately strict — a model that hedges around a correct
answer fails closed exactly the way an uncited one does, which is the intended trade-off (see
`DECISIONS.md`), not a bug to fix by loosening the contract.

## What this means for the release

None of these are citation-safety defects — every citation that did reach the user was verified
(100%), and the refusal-vs-answer split is a quality/coverage question, not a trust one. The
sealed holdout run (Task 12) will show whether this 72% answerable answer rate holds, worsens, or
improves once run against unseen questions.
