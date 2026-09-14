# Fresh-test topic matrix — blank allocation

Status: `awaiting_external_review`. This matrix allocates coverage; it is not a question set and
contains no fresh holdout payload. The external author/reviewer must fill the slots after the
development candidate is fixed.

## Required item allocation

| item type | slots | required source shape | authoring constraint |
|---|---:|---|---|
| Single-document | 24 | One actual document/page or an explicit corpus-gap reference | Include source scope/date where status matters. |
| Multi-document | 18 | At least two required document/page references | Require synthesis, contrast, or a shared legal/regulatory concept. |
| Unanswerable | 18 | Empty answer source list plus a documented corpus gap/unsupported scope | Refusal must be justified by the frozen corpus, not by an obscure answer. |
| **Total** | **60** | 42 answerable + 18 unanswerable | Exact counts must be sealed after review. |

## Scenario coverage overlay

Tags may overlap; the reviewer should identify the item IDs covering each row after authoring.

| scenario family | minimum slots | item IDs supplied by reviewer |
|---|---:|---|
| Status, amendments, withdrawals, or successor conflicts | 6 | pending |
| Exceptions, conditions, scope, or qualification clauses | 6 | pending |
| Numeric obligations, dates, thresholds, or deadlines | 6 | pending |
| Partial support, ambiguous references, or corpus gaps | 6 | pending |

The remaining slots should broaden source families across SARB, FSCA, NCR/NCA, and IFRS material
without turning the matrix into a list of implementer-authored questions. A scenario can satisfy
more than one overlay row.

## Required post-authoring checks

- exact and normalised duplicate checks against development and exposed historical sets;
- reviewer examination of near-duplicates and substantive novelty;
- page-level reference verification, including OCR-derived text;
- explicit author/reviewer provenance and disagreements;
- derivation of a 42-item retrieval file from the answerable golden items, preserving exact question
  text and all required source references;
- hashes, counts, ordered IDs, candidate identity, targets, and resume rules sealed only after
  the independent review is complete.
