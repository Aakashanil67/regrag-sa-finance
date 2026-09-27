# Archived: scripts that no longer run against current output

`paired_eval.py` and `paired_eval_metadata.py` both parse `reports/eval_summary.md` for a
per-item table (`| g01 | 0.79 | 0.85 | ... |`) that `evals/run_release.py` stopped producing when
the release evaluation harness was rewritten — the current summary is a single aggregate table
with confidence intervals, no per-item rows at all. Run today, `_parse_summary` matches zero
lines and both scripts produce an empty, silently meaningless comparison rather than an error.

Kept rather than deleted, same as `reports/archive/v1.0-audit/`: they're real historical record of
a paired-significance analysis method this project still uses in spirit (see
`reports/failure_analysis.md`'s Wilson-interval treatment of small samples), just against a report
format that no longer exists. Their own output — `reports/archive/v1.0-audit/paired_comparison.md`
and `paired_comparison_metadata.md` — is archived for the same reason.
