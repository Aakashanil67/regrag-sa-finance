# Agent vs. plain RAG — multi-document comparison questions

All 10 `multi-doc`-type golden items, run through both `rag.answer_question` (single retrieval, k=5) and `agent.answer_question` (retrieve, decide, requery once if a named document is missing, then answer). Restricted to this subset rather than the full 45-item golden set on purpose — this is the failure class the agent was built to fix (`reports/failure_analysis.md`, items g37 and g44), and the other 35 single-document items already work without it, so running them again would only spend API budget re-confirming that reranked single-shot retrieval is enough for a question that doesn't name two documents.

**Refusals fixed: 0. Refusals introduced: 0.**
**Mean cost ratio (agent / plain): 2.76x. Mean latency ratio: 1.77x.**

| id | question | plain refused | agent refused | requeries | cost ratio | latency ratio |
|---|---|---|---|---|---|---|
| g36 | Both Directive D3/2023 and the IFRS 9 project summary describe a shift from an 'incurred loss' model to a forward-looking model. What is that forward-looking model called? | False | False | 0 | 1.79x | 0.19x |
| g37 | Which SARB documents were issued under section 6(5) versus section 6(6) of the Banks Act, and what is the practical difference between the two document types they represent? | True | True | 2 | 4.35x | 2.84x |
| g38 | How does the National Credit Act's stated purpose, as summarised in the Notebook brochure, connect to the NCR's June 2025 guideline on consumers under debt review? | False | False | 0 | 1.69x | 1.20x |
| g39 | What single section of the National Credit Act empowers the NCR to issue both the June 2025 and September 2025 guidelines discussed in this corpus? | False | False | 1 | 3.05x | 1.92x |
| g40 | How did the FSCA's 2019 intermediary activity segmentation update build on the 2014 Retail Distribution Review? | False | False | 0 | 1.62x | 1.13x |
| g41 | Both the Conduct Standard for Banks and the Conduct Standard for OTC Derivative Providers are FSCA conduct standards, but they rest on different underlying legislation. What are the two Acts? | False | False | 2 | 3.04x | 2.59x |
| g42 | The IFRS 9 project summary and PwC's practical guide both describe IFRS 9 as replacing an earlier standard. Which standard, and for what type of accounting? | False | False | 0 | 1.80x | 1.23x |
| g43 | Circular 6/2004 and Circular 19/2004 were both issued by the SARB in 2004 to banks. What distinct regulatory topics does each one address? | False | False | 0 | 1.60x | 1.12x |
| g44 | What common international standard-setting body do Directive D8/2023 and Directive D10/2021 both cite as the source of the principles they implement? | True | True | 2 | 5.08x | 3.03x |
| g45 | The National Credit Act itself and the NCR's Notebook brochure both describe the Act's objectives. Name one objective that appears in both sources. | False | False | 2 | 3.59x | 2.44x |
