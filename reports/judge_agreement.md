# Judge agreement on the dev runs

Every answer is graded against the reference answer as `correct`, `partial` or `incorrect`, and RAG answers are also graded against the retrieved context as `supported`, `partial` or `unsupported`. The same rubric is used by the human reviewer later. Both judges run locally through Ollama, so they cost nothing and cannot be swayed by the generator: Qwen 2.5 7B and Llama 3.1 8B come from different families than GPT-5.6 Luna and from each other. Refused answers and refused unanswerable questions are not graded; they are counted from the run itself.

## Correctness counts (34 answerable dev questions)

| run | judge | correct | partial | incorrect | refused |
|---|---|---|---|---|---|
| wordpiece (RAG) | Qwen 2.5 7B | 16 | 14 | 2 | 2 |
| wordpiece (RAG) | Llama 3.1 8B | 24 | 1 | 7 | 2 |
| closed-book | Qwen 2.5 7B | 12 | 10 | 6 | 6 |
| closed-book | Llama 3.1 8B | 16 | 1 | 11 | 6 |

Cohen's kappa between the two judges, on the items both graded:

| run | n | raw agreement | kappa |
|---|---|---|---|
| wordpiece (RAG) | 32 | 0.594 | 0.320 |
| closed-book | 28 | 0.679 | 0.512 |

Support (RAG run only): both judges call all 32 graded answers `supported`, so there is nothing to compare on that axis. It says the answers stay inside the retrieved text, not that the retrieved text is the right text.

The closed-book run answered 5 of the 6 unanswerable questions, and both judges agree on that count (it comes from the run, not the judges).

## Where they disagree

They never split on the extremes. Every `correct` from Qwen is a `correct` from Llama, and every `incorrect` from Qwen is an `incorrect` from Llama. All the disagreement sits in Qwen's `partial`: 14 items on the RAG run, of which Llama calls 8 `correct` and 5 `incorrect`.

Qwen marks down for extra material. On d04 it says the answer "includes additional information not present in the reference answer", and Llama calls the same answer correct with no contradictions. The same pattern shows on d10, d12 and d15. Llama goes the other way when a detail differs: on d25 it says the answer contradicts the reference because it claims no issuance deadline, while Qwen only says the seven-day period is missing. On d31 Llama reads "10% of all issued shares" against "10% of the total nominal value" as a contradiction, and Qwen calls it a missing detail. Llama is also the one that reads d39 as a contradiction where Qwen sees a wrong deadline.

So Qwen is stricter about padding and Llama is stricter about wrong specifics. Neither is obviously right, and a couple of Llama's "contradicts" reasons (d27) are hard to follow.

## Limits

These are 7-8B models grading legal text with long references. They are weaker graders than a frontier model, and a kappa of 0.32 on the RAG run is low. I treat their labels as a cheap second opinion, not ground truth. That is why a human checks 20 answers afterwards.
