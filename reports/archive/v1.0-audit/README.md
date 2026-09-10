# Archived: pre-correction v1.0 evidence

These six files are the tuning and evaluation record from before the corpus correction (hardening
plan Task 5): `agent_eval.md`, `improvement_log.md`, `paired_comparison.md`,
`paired_comparison_metadata.md`, and `smoke_test.md` all reference golden-set item IDs (`g02`
through `g45`) from the development set as it existed before Task 5 replaced a withdrawn
consultation draft and stale third-party IFRS 9 commentary with corrected sources and added four
missing SARB documents — those IDs, and in several cases the underlying documents themselves, no
longer correspond to what is in `evals/golden_dev.jsonl` or `corpus/manifest.json` today.
`eval_history_ragas_legacy.csv` is the RAGAS-only history schema `evals/run_ragas.py` wrote before
`evals/run_release.py` replaced it with the current structural-plus-RAGAS schema (see
`DECISIONS.md`).

None of this is deleted, and none of it should be read as current evidence of this project's
behaviour. It's kept because it's real historical record of the tuning work (chunk size, reranking,
the agent's requery pass) that shaped decisions still in effect — the numbers just aren't
reproducible against today's corpus or golden set, so presenting them as current would misrepresent
what the pipeline does now. Current evidence lives in `reports/eval_summary.md`,
`reports/retrieval_bench.md`, `reports/failure_analysis.md`, and `reports/eval_history.csv`.
