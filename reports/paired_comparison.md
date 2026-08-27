# Is the improvement real? A paired re-analysis

The headline before/after table in the README compares the mean of one RAGAS run against the mean of another. Those two runs did not score the same questions: the pre-improvement run scored 34 items and the post-improvement run scored 39, overlapping on 33. Reranking pulled 6 previously-refused questions (`g02`, `g31`, `g32`, `g36`, `g39`, `g43`) into a real answer, and pushed 1 the other way (`g29`).

That matters, because a mean over a changing item set moves for two unrelated reasons: the answers got better, or the mix of questions being averaged changed. Only the first is a quality claim. This report separates them.

## Paired comparison, on the 33 items both runs scored

| metric | before (all) | after (all) | before (paired) | after (paired) | paired Δ | improved | regressed | tied | sign test |
|---|---|---|---|---|---|---|---|---|---|
| faithfulness | 0.791 | 0.829 | 0.800 | 0.823 | +0.023 | 9 | 9 | 15 | p = 1.00 |
| answer relevancy | 0.851 | 0.810 | 0.859 | 0.818 | -0.041 | 10 | 11 | 12 | p = 1.00 |
| context precision | 0.673 | 0.790 | 0.694 | 0.802 | +0.109 | 14 | 5 | 14 | p = 0.06 |
| context recall | 0.912 | 0.968 | 0.939 | 0.970 | +0.030 | 1 | 0 | 32 | p = 1.00 |

### What survives

**Context precision is the strongest result, though not a significant one at this sample size.** It gains +0.109 on the paired set, with 14 items improving against 5 regressing — a sign test p of 0.06, which falls just outside the conventional 0.05 threshold and would not clear peer review on its own. What makes me treat it as real rather than lucky is that it is the only metric where the effect has a mechanism behind it: dropping chunks the bi-encoder surfaced on loose topical similarity is precisely and solely what a cross-encoder does, so a precision-shaped gain is the prediction, not a post-hoc reading of whichever number happened to move. Thirty-three items is too few to confirm it; a larger golden set is the fix, and it isn't built.

**Faithfulness does not survive.** The headline gain of +0.038 shrinks to +0.023 once the item set is held fixed, and the per-item split is 9 improved against 9 regressed — a coin flip (p = 1.00). Most of the apparent gain was composition: the six questions that entered the average happened to score above the old mean. I am not claiming reranking improved faithfulness.

**Context recall was already at the ceiling.** It reads +0.056 across the full runs but only +0.030 paired, with 32 of 33 items completely unchanged and exactly 1 item moving. At a paired baseline of 0.939 there was almost nothing left to win.

**Answer relevancy drifts slightly negative** (-0.041 paired, 10 improved against 11 regressed, p = 1.00). Also indistinguishable from noise, and reported rather than dropped.

## Retrieval benchmark: 20 questions is a small ruler

Hit-rate@5 went from 85% to 95%. In absolute terms that is 17/20 to 19/20 — **two questions**. Wilson score intervals:

| | hit-rate@5 | 95% CI |
|---|---|---|
| before | 17/20 (85%) | 64% – 95% |
| after | 19/20 (95%) | 76% – 99% |

The intervals overlap heavily. On this benchmark alone the hit-rate difference is not separable from sampling noise, and a 20-question set cannot resolve a two-question gap. MRR (0.654 → 0.808) is the better-powered signal in the same data, because it moves on *where* the correct chunk ranks rather than only whether it cleared a cutoff, so a question going from rank 4 to rank 1 registers instead of being scored identically.

## What I would say about this config, honestly

Reranking at 800-token chunks earns its place on one measured effect and one unambiguous one. The measured effect is context precision, directionally clear and mechanistically expected but short of significance on 33 items. The unambiguous one is coverage: five net refusals became answered questions, and that needs no significance test, because it is a count of behaviour changing, not an estimate of a mean. It did not demonstrably improve faithfulness or recall, and the retrieval hit-rate gain sits inside the noise floor of a 20-question benchmark. The costs are equally concrete — a cross-encoder pass over 20 candidates on every query, and a coarser citation, since an 800-token chunk spans more pages than a 500-token one and the citation inherits that span.

The honest summary is a coverage win with a probable precision win attached, not an across-the-board improvement. The README says that rather than the four-green-arrows version, which the raw means would have supported and the paired data does not.
