# regrag-sa-finance

A retrieval-augmented assistant that answers questions about South African financial regulation
(SARB, IFRS 9, National Credit Act, FSCA) from a local corpus, with mandatory per-claim citations
and a refusal path when the retrieved context can't support an answer.

Under active build — this README gets rewritten properly once the eval harness has real numbers
to show. Current state:

- [ ] Ingestion + heading-aware chunking
- [ ] Vector store + retrieval benchmark
- [ ] Cited RAG core + refusal behaviour
- [ ] Evaluation harness (RAGAS + DeepEval regression gate)
- [ ] Eval-verified retrieval improvements
- [ ] API + chat UI + ops dashboard
- [ ] Docker + hardening
- [ ] Recruiter-grade writeup
