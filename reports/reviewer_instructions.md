# Independent reviewer instructions — fresh test preparation

Status: `awaiting_external_review`. This document prepares the review; it is not evidence that a
review happened. No fresh questions, answers, references, reviewer names, credentials, dates, or
scores are included here.

## Exact outside input needed

Please provide, after the development candidate is fixed:

1. A fresh 60-item question set with 24 single-document, 18 multi-document, and 18 unanswerable
   items. The implementer must not author or inspect the payload before the candidate freeze.
2. For every answerable item, a reference answer and exact source document/page references,
   including scope, date, and authority qualifications where relevant.
3. For every unanswerable item, the corpus gap or unsupported portion that makes a refusal correct.
4. A reviewer identity, relevant financial-regulation experience, review scope, review date, and
   any conflict or limitation. The reviewer may be a domain practitioner or an appropriately
   qualified regulatory researcher; the claim must match the actual experience.
5. A completed reference review that records page-level disagreements, corrections, adjudication,
   and unresolved reference issues. Do not silently edit an item after it has been run.

An AI second opinion may identify possible reference or answer issues, but it is supplementary and
must not be recorded as independent financial-regulation domain validation.

## Reference review

Before sealing the protocol, check every answerable reference against the actual extracted page
text and the official source status. Check OCR-derived text explicitly. Check that the question is
new in substance, not merely a string variant of a development or historical item. Record
duplicates, near-duplicates, source gaps, page errors, disagreements, adjudication, and unresolved
issues in the reference-review template.

## Answer review

Review all 60 saved outcomes, including refusals. Use the exact saved answer, exact formatted
context, citations, source notices, reference answer, and source pages. Do not retrieve again and
do not use a later answer to replace a weak one.

For answered items, score:

- correctness: `supported`, `partially_supported`, `unsupported`, or `not_assessable`;
- completeness against the question and reference;
- citation entailment for every cited source/page;
- authority and status handling, including withdrawn, superseded, draft, press, and historical
  materials;
- material error severity: `none`, `minor`, `material`, or `critical`;
- a concise rationale linked to item IDs and source evidence.

For refusals, distinguish a true corpus gap, a retrieval miss, a contract/validator failure, or an
unjustified refusal. Record whether the refusal was justified and why.

## Predeclared release targets

These are acceptance targets, not statistical guarantees or a legal/financial certification:

- zero observed material unsupported or misleading answered claims in human review;
- at least 34/42 answerable items answered with fully supported responses;
- at least 17/18 unanswerable items refused;
- zero structurally invalid served citations;
- required source warnings present whenever the saved evidence requires them.

If review is partial, report reviewed/total counts and keep release readiness pending. Do not score
unreviewed rows as passes. Do not fabricate a sealed protocol or independent review record.

Use `reports/fresh_test_topic_matrix.md`, `reports/fresh_item_template.json`,
`reports/reference_review.template.json`, and `reports/independent_answer_review.template.json`
as blank preparation forms only.
