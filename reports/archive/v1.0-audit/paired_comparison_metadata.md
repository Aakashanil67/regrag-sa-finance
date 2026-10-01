# Source-type/currency metadata: paired re-analysis

Same method as `paired_comparison.md`, applied to the source-authority and currency metadata fix. The context gained document type, issuer and title, and the result gained status and third-party notices (`src/rag.py`). Before: chunk_size=800+rerank with no source metadata in the prompt. After: the same config with source-type lines added to context and `source_notices` added to the result.

1 item entered the scored set that was previously refused (`g24`), and 1 item newly refused (`g45`).

## Paired comparison

| metric | before (paired) | after (paired) | Δ | improved | regressed | tied | sign test |
|---|---|---|---|---|---|---|---|
| faithfulness | 0.830 | 0.819 | -0.011 | 7 | 12 | 19 | p = 0.36 |
| answer relevancy | 0.814 | 0.828 | +0.014 | 11 | 16 | 11 | p = 0.44 |
| context precision | 0.784 | 0.798 | +0.013 | 1 | 1 | 36 | p = 1.00 |
| context recall | 0.967 | 0.993 | +0.026 | 1 | 0 | 37 | p = 1.00 |

## What this fix actually did

The paired deltas are small and none of the sign tests establishes a reliable change. The metadata labels source authority and currency. These RAGAS metrics do not directly measure whether an answer identifies the kind of source it cites.

`g24` changed from refusal to a correct, cited answer. The model had not been shown the document's title, so it could not identify Guideline 004/2025. Adding the title copied from the PDF cover fixed this case. Paraphrased titles had failed in the first attempt.

`g45` changed from an answer to a refusal. In the recorded retrieval check, `nca_act_34_2005` p.1-2 ranked 2nd before reranking, inside the top-5, but 6th after reranking, outside k=5. The reranker put four NCR guideline chunks ahead of the Act's purpose section. The retrieval code did not change with the metadata fix. Before the fix, the model answered from a brochure. After being told that the source was a regulator explainer, it refused to attribute the claim to the Act. Source labelling exposed this retrieval gap.

Separate manual checks found that the PwC IFRS 9 guide attached a third-party notice and either 2004 SARB Circular attached an instrument-status notice. Both were structured `source_notices`, separate from the graded answer. These checks used the historical source classifications, which were later corrected.
