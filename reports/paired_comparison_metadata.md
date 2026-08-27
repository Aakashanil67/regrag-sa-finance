# Source-type/currency metadata: paired re-analysis

Same method as `reports/paired_comparison.md`, applied to the source-authority and currency metadata fix (document type, issuer, title, and dated-instrument/third-party notices — see the module docstring in `src/rag.py`). Before: chunk_size=800+rerank with no source metadata in the prompt. After: the same config with source-type lines added to context and `source_notices` added to the result.

1 item entered the scored set that was previously refused (`g24`), and 1 item newly refused (`g45`).

## Paired comparison

| metric | before (paired) | after (paired) | Δ | improved | regressed | tied | sign test |
|---|---|---|---|---|---|---|---|
| faithfulness | 0.830 | 0.819 | -0.011 | 7 | 12 | 19 | p = 0.36 |
| answer relevancy | 0.814 | 0.828 | +0.014 | 11 | 16 | 11 | p = 0.44 |
| context precision | 0.784 | 0.798 | +0.013 | 1 | 1 | 36 | p = 1.00 |
| context recall | 0.967 | 0.993 | +0.026 | 1 | 0 | 37 | p = 1.00 |

## What this fix actually did

Nothing here moves a RAGAS mean by more than noise — every paired delta is small, every sign test is indistinguishable from a coin flip, and that's the honest result. This fix was never a retrieval or generation-quality change; it's a disclosure change, and RAGAS's four metrics don't have a dimension for "did the answer correctly attribute what kind of document this is." The two changes that matter aren't visible in this table:

**`g24` went from refusal to a correct, cited answer.** This is the specific failure `reports/failure_analysis.md` diagnosed: the model couldn't tell which retrieved document was actually "Guideline 004/2025" because no guideline's own number reached the prompt anywhere. Adding each document's real title (extracted from the PDF's own cover page, not paraphrased — the first attempt at this fix used paraphrased titles and didn't work, which is how the paraphrasing was caught) fixed it directly.

**`g45` went the other way, and it's not a capability loss.** Retrieval for that question returns the same five chunks before and after this change — verified directly against `src.retrieve.retrieve`, not inferred from the score — and none of them is the National Credit Act's own text, only a DTIC brochure describing it. Before this fix, the model answered anyway, treating the brochure's summary as equivalent to the Act's own words. After, told explicitly that the source is a "Regulator explainer brochure," it correctly refused to attribute a claim to "the Act itself" when the Act's own text was never in front of it. That's the fix working as designed. It reads as a regression only if refusal count is the metric, which is exactly the kind of thing a metric-only readout misses.

Live-verified separately (not part of this batch, checked by hand against the model's actual output): citing the PwC IFRS 9 guide now attaches a third-party-source notice, and citing either 2004 SARB Circular attaches a superseded-instrument-type notice — both as structured `source_notices`, not text folded into the graded answer.
