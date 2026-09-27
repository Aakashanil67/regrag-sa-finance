# Failure analysis

Source: `reports/runs/test-final.json` (run id `test-final`), the sealed test split of 60 items
(48 answerable, 12 unanswerable) run once against the frozen pipeline. v1.1's analysis is kept in
[`reports/archive/v1.1/`](archive/v1.1/).

Headline: 37 of 48 answerable items answered (77%), 11 of 12 unanswerable items refused, no served
citation unverified (0 of 82). Evidence retrieved means a retrieved chunk covers the reference
evidence page. Judge columns are correctness from the llama and qwen judges.

## Answerable items that were refused (11)

Every refusal carried the reason `model_refusal` and the text "I don't have a source for that."

| id | type | evidence retrieved | diagnosis |
|---|---|---|---|
| t06 | single | no (right document, wrong page) | retrieval miss |
| t07 | single | yes | model hedged |
| t08 | single | no (right document, wrong page) | retrieval miss |
| t16 | single | no (document absent from context) | retrieval miss |
| t21 | threshold | no (right document, wrong page) | retrieval miss |
| t22 | threshold | no (right document, wrong page) | retrieval miss |
| t24 | threshold | no (document absent from context) | retrieval miss |
| t26 | threshold | no (document absent from context) | retrieval miss |
| t38 | multi | half (notebook brochure yes, conduct standard 3 no) | retrieval miss on one part |
| t41 | multi | half (affordability regulations yes, 2006 regulations no) | retrieval miss on one part |
| t42 | multi | half (2004 circular yes, D10/2021 no) | retrieval miss on one part |

## Unanswerable items that were answered (1)

| id | judges | diagnosis |
|---|---|---|
| t50 | both `answered_unanswerable` | The question asks about buy-now-pay-later. The answer cited the National Credit Act's general affordability duty (p.54), which is adjacent material, not an answer about that product. Reference ambiguous: the question sits close to covered text. |

## Items either judge marked incorrect (5)

| id | type | llama | qwen | diagnosis |
|---|---|---|---|---|
| t01 | single | incorrect | partial | Judge disagreement. The answer states the independence point correctly, then adds a Minister-directed cooperation duty from p.23 that the reference does not mention. Extra detail, not a contradiction. |
| t29 | status | incorrect | partial | Judge disagreement. The answer says IAS 39 still applies for hedge accounting, cited to `ifrs9_issued_2021` p.1, while the reference evidence is `ifrs9_project_summary_2014` p.4. Reference ambiguous. |
| t57 | false_premise | incorrect | incorrect | Model went along with the premise. The answer describes the agent's monthly statement and never says debt counsellors do not collect the money. Evidence page (p.12) not retrieved, so retrieval miss. |
| t60 | false_premise | incorrect | correct | Judge disagreement. The answer reports the review's concern about commission for selling alone and the proposal to tie fees to ongoing service. Llama read that as contradicting the reference. |

## Patterns

All 11 refusals are the model declining to answer. In 7 the evidence page was missing from the
retrieved context, in 3 (the multi-part items) one of two sources was missing, and only t07 had its
evidence and refused anyway. Retrieval is the main limit: 8 of 48 answerable items retrieved no
evidence page at all, which is the 83% any-hit rate. Refusing when the source is missing is the
intended behaviour, so these cost answer rate, not safety. The dangerous failures are the answers
that go along with a question's framing (t57, and t50 on the unanswerable side), and both rest on
neighbouring text. Three of the five "incorrect" marks come from one judge disagreeing with the
other, so the judged correctness numbers carry that noise.
