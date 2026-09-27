# Reviewer instructions

Thanks for helping. You are checking 20 answers that RegRAG gave to questions about South African financial regulation. Your labels are the human check on the two automated judges, so please don't try to guess what they said. You won't see their labels.

**Time:** about 60 to 90 minutes.

## What you have

- `reports/review_packet_v1.2.md`: for each answer, the question, the answer, the text of the cited pages, the reference answer and its evidence quotes.
- `reports/review_v1.2.csv`: the form. One row per answer, with columns `id,correctness,support,reviewer,notes`.

## The two labels

**correctness** compares the answer with the reference answer.
- `correct`: states the reference's key facts and nothing that contradicts it.
- `partial`: some key facts right, others missing or wrong.
- `incorrect`: wrong, contradicts the reference, or answers a different question.

**support** compares the answer with the cited text in the packet.
- `supported`: every sentence of the answer is stated in the cited text.
- `partial`: some sentences are, some are not.
- `unsupported`: the main claim is not in the cited text.

## Filling in the form

Use the exact label words above, in lowercase. Put your initials in the `reviewer` column on every row.

Judge against the quoted source text and your own knowledge of the rules. If the reference answer looks wrong or incomplete to you, label the answer on what the source says, then write the disagreement in `notes`. Use `notes` for anything else odd too, such as a citation that points at the wrong page.

Save the file as CSV, keep the header row, and send it back.
