# Archived: pre-correction v1.0 evidence

These six files are the tuning and evaluation record from before the corpus correction (corpus correction
step 5): `agent_eval.md`, `improvement_log.md`, `paired_comparison.md`,
`paired_comparison_metadata.md`, and `smoke_test.md` all reference golden-set item IDs (`g02`
through `g45`) from the development set as it existed before step 5 replaced a withdrawn
consultation draft and stale third-party IFRS 9 commentary with corrected sources and added four
missing SARB documents, those IDs, and in several cases the underlying documents themselves, no
longer correspond to what is in `evals/questions_dev.jsonl` or `corpus/manifest.json` today.
`eval_history_ragas_legacy.csv` is the RAGAS-only history schema `evals/run_ragas.py` wrote before
`evals/run_release.py` replaced it with the current structural-plus-RAGAS schema (see
`DECISIONS.md`).

These records describe earlier tuning work. Their numbers cannot be reproduced against the
current corpus or question set. Current evidence is in `reports/runs/test-final.md`,
`reports/retrieval_bench.md`, `reports/failure_analysis.md`, and `reports/eval_runs.csv`.
