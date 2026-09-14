# Development answer validation — prepared, not run

The retrieval comparison retained Variant A. Because B and C did not qualify, no paid
baseline-versus-candidate answer comparison is authorized or budgeted. This report is the
offline run sheet for validating the retained serving configuration after budget authorization;
it contains no generated answers, judge scores, or fabricated reviewer outcomes.

## Planned single-configuration validation

- Configuration: Variant A, 800/75 tiktoken chunks, semantic retrieval, reranking, `k=5`.
- Development IDs: all 57 fixed items in `evals/golden_dev.jsonl`, preserving factual,
  multi-document, and unanswerable denominators.
- Generation: one uncached generation per ID, journalled through the schema-2 run machinery.
- Saved evidence: raw and served output, refusal reason, usage, timings, exact ordered context,
  formatted context, citations, source notices, validator outcome, and failures.
- Development judging: conditional RAGAS scoring only for answered factual/multi-document items;
  refusals and generation failures stay visible in denominators.
- Stress repeats: the ten IDs and rationale in `reports/live_evaluation_budget.md`, repeated twice
  after the main development pass and reported separately from the primary 57-item results.

## Decision rule after actual execution

This run cannot establish independent expert accuracy. It can establish only whether the retained
configuration produces reproducible development evidence and whether any observed refusal,
citation, source-status, or validator defect needs a bounded fix before the fresh test. A defect
must be recorded and fixed against development evidence; it must not trigger open-ended tuning or
turn a retrieval-only gain into an answer-quality claim.

Current status: **pending budget authorization, provider execution, and external review**.
