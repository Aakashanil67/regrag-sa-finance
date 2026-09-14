# Live evaluation budget — pending authorization

Status: prepared offline on 2026-09-14. No provider call, judge call, outreach, or fresh-test
payload was made or inspected. The candidate remains in development and the serving configuration
is unchanged.

## Configuration and pricing

Variant A is retained: 800/75 tiktoken chunks, semantic retrieval, reranking, `k=5`, and the
currently configured local corpus/store path. B and C did not qualify, so this plan budgets one
development configuration only; it does not manufacture a baseline-versus-candidate comparison.

The answer provider is configured as Anthropic model alias `claude-haiku-4-5`. The RAGAS judge is
also configured through Anthropic with the same alias and a 4,096-token output cap in the judge
adapter. The checked-in code's conservative estimates are 8,192 input / 1,024 output tokens per
generation and 4,096 input / 512 output tokens per judge metric subcall. Local embedding and
reranker inference is not provider-billed.

The official Anthropic page checked on 2026-09-14 lists Haiku 4.5 at $1.00 per million input
tokens and $5.00 per million output tokens: [Anthropic Haiku pricing](https://www.anthropic.com/claude/haiku).
The official model-status page lists the dated Haiku 4.5 model separately; the application still
uses the alias and has not made a live call to resolve that alias, so the answer/judge provider
identity remains an external provenance limitation: [Anthropic model status](https://docs.anthropic.com/en/docs/about-claude/model-deprecations).

## Proposed batch

Counts below are maxima. Refused items are not sent to RAGAS, so actual judge calls may be lower.
Repeated stress questions are reported as repeats, never pooled as independent samples.

| batch | generation calls | judge metric subcalls | provider requests | estimate |
|---|---:|---:|---:|---:|
| Development validation: 57 fixed `golden_dev` IDs | 57 | up to 47 × 4 = 188 | 245 | $1.902208 |
| Stress repeats: 10 preselected IDs × 2 additional runs | 20 | 0 | 20 | $0.266240 |
| New CI snapshot: 10 fixed items | 10 | up to 10 Faithfulness calls | 20 | $0.199680 |
| Fresh test: up to 60 items, 42 answerable × 4 RAGAS metrics | 60 | up to 168 | 228 | $1.117952 |
| Limited public/synthetic demo: at most 4 questions | 4 | 0 | 4 | $0.053248 |
| **Total maximum** | **151** | **366** | **517** | **$4.446208** |

The proposed authorization cap is **$6.00 USD**, a rounded cap with approximately 35% uncertainty
above the conservative estimate. This is a proposal, not authorization. Expected actual spend in
this offline preparation is **$0.00**. Before any paid run, re-check the provider price, exact
model identifier, remaining account authorization, and the cap passed to the runner.

## Stress selection

These IDs are selected before any repeated run and cover the requested failure modes:

| ID | category | reason |
|---|---|---|
| `g01` | numeric / citation-sensitive | Numeric obligation tied to an annexure and page citation. |
| `g02` | named-document / status | Explicit directive identifier and replacement relationship. |
| `g12` | ordinary factual | Simple commencement-date fact from the NCA. |
| `g22` | current-guideline / numeric | Current NCR guideline with statutory section references. |
| `g25` | authority/status | FSCA publication date where the local corpus records a press item. |
| `g32` | historical standard | IFRS 9 issue that can attract unsupported current-status claims. |
| `g36` | multi-document | Cross-source terminology alignment between SARB and IFRS material. |
| `g40` | multi-document / partial support | Relationship between two FSCA RDR documents. |
| `g46` | unanswerable | Current repo rate is outside the corpus and should refuse. |
| `g56` | current-status conflict | C1/2026 status treatment of an older Banks Act circular. |

## Preconditions and accounting

1. Resolve the answer-provider alias to a documented provider model identifier, or retain the
   unresolved status and do not freeze the candidate.
2. Obtain the external reviewer input and complete the fresh reference review before sealing any
   holdout protocol.
3. Run the provider-free dry-run with the same cap and frozen identities before constructing a
   provider client. Keep response caching disabled.
4. Record actual generation usage, judge usage, failures, ambiguous operations, and the final
   provider invoice/usage result. Estimates in this document must not be relabelled as actuals.
