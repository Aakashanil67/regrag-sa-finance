# Release evaluation summary

**Split:** holdout | **Label:** v1.1.0-rc2 | **Run:** holdout-0cf5e0821ec7 | **Status:** complete

Fractions shown as `n/d` alongside the percentage — a rate with a small or partial denominator is not the same claim as one over the full set. The 95% CI is a Wilson interval on the observed rate, not a claim that the true rate equals the point estimate — a 30-item holdout leaves real uncertainty even at 100%.

| metric | value | 95% CI |
|---|---|---|
| Answerable answer rate | 17/24 (71%) | 51%–85% |
| Unanswerable refusal recall | 6/6 (100%) | 61%–100% |
| Citation-contract pass rate | 23/30 (77%) | 59%–88% |
| Verified-citation rate | 30/30 (100%) | 89%–100% |
| RAGAS (over 17 answered answerable items, 7 refusal(s) excluded) | faithfulness=0.866, answer_relevancy=0.681, context_precision=0.894, context_recall=1.000 | — |

