# Corpus authority and currency review — 2026-09-14

This is a source-audit record for the corpus review. It is a reproducibility and retrieval-safety
review, not a legal opinion. Official landing pages, normative PDFs, and current-status notices
were checked on 2026-09-14 where the relevant public pages were available. Local PDF quality is
reported from the byte-pinned files already present in `corpus/`; no corpus PDF was downloaded,
replaced, OCRed, or otherwise modified during this review.

## Method and findings

- The 22 manifest entries were matched to local files and SHA-256 hashes. All 22 matched.
- PyMuPDF extraction produced a non-empty text layer for every file. Three documents have one
  zero-text page each, consistent with a cover or image-only page: the National Credit Act,
  the 2014 RDR discussion paper, and the 2021 issued IFRS 9 text. No document is wholly
  scanned, so local OCR was not warranted.
- Authority and stage are taken from the document and the official host. A press release,
  consultation draft, explanatory document, or historical snapshot is not treated as the
  normative instrument it describes.
- The SARB status evidence is stronger than the prior generic landing-page labels: D8/2025
  explicitly withdraws D8/2023, D4/2023 explicitly replaces D10/2021, and Circular C1/2026
  lists the effective directives and says directives remain effective until formally withdrawn.
- The FSCA consultation index still presents the April 2018 OTC document under “Documents for
  Consultation”. The final FMA Conduct Standard 2 of 2018 was not located as an official,
  directly retrievable normative PDF. The corpus's 2020 Banks item is explicitly retained as a
  press release; the final Conduct Standard 3 of 2020 is not in the corpus.
- The IFRS Foundation's current IFRS 9 page records amendments in 2024 and later standard
  history, so the 2021 issued text remains a dated historical snapshot rather than a current
  consolidated standard.

## Entry-by-entry audit

`pages/chars/words/zero/images` are local PyMuPDF counts. “Usable text” means the file has a
text layer suitable for extraction; it does not assert that every table or glyph is semantically
perfect. Page numbers in the evidence column are PDF pages, one-based.

| id | instrument / authority | snapshot / publication | status as reviewed | authoritative evidence | extraction quality | unresolved limitation |
|---|---|---|---|---|---|---|
| `sarb_g3_2025_climate_disclosures` | SARB PA; final guidance note; official non-binding guidance | 2025-01; official PDF | current as a 2025 final PA guidance note | [SARB Prudential Authority publications](https://www.resbank.co.za/en/home/publications/prudential-authority) | 9 pages; 22,853 chars; 3,075 words; 0 zero-text pages; 6 images; usable text | No dated 2026 replacement notice was found on the broad landing page; current means not superseded in the checked source, not legal advice. |
| `sarb_d3_2023_accounting_provisions_ifrs9` | SARB PA; final directive; binding instrument | 2023-01; official PDF | current; C1/2026 lists D3/2023 as effective | [C1/2026 status PDF](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-circulars/2026/C1-2026.pdf), pp. 1, 5 | 3 pages; 6,315 chars; 967 words; 0 zero-text pages; 7 images; usable text | C1/2026 is an annual status confirmation, not a consolidated directive register. |
| `sarb_d8_2023_threshold_amounts` | SARB PA; final directive; binding instrument | 2023-09; official PDF | superseded / withdrawn by D8/2025 | [D8/2025 official PDF](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-directives/2025/D8-2025%20-%20Thresholds%20Amounts%20to%20the%20revised%20Standardised%20and%20IRB%20approaches%20for%20credit%20Risk.pdf), p. 1 | 6 pages; 15,288 chars; 2,302 words; 0 zero-text pages; 7 images; usable text | The 2023 text remains useful historical evidence; do not answer current-threshold questions from it. |
| `sarb_d8_2025_threshold_amounts` | SARB PA; final directive; binding instrument | 2025-07; official PDF | current; C1/2026 lists D8/2025 as effective | [C1/2026 status PDF](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-circulars/2026/C1-2026.pdf), pp. 1, 6 | 6 pages; 15,609 chars; 2,317 words; 0 zero-text pages; 7 images; usable text | The directive's title omits some IRRBB detail present in its body; the manifest title should not be used as the sole scope description. |
| `sarb_d10_2021_operational_resilience` | SARB PA; final directive; binding instrument | 2021-12; official PDF | superseded by D4/2023 | [D4/2023 official PDF](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-directives/2023/D4-2023-%20Directive%20on%20operational%20resilience.pdf), p. 2 | 3 pages; 6,191 chars; 858 words; 0 zero-text pages; 8 images; usable text | Preserve as historical text; current compliance questions require D4/2023. |
| `sarb_d4_2023_operational_resilience` | SARB PA; final directive; binding instrument | 2023; official PDF | current; C1/2026 lists D4/2023 as effective | [C1/2026 status PDF](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-circulars/2026/C1-2026.pdf), pp. 1, 5 | 3 pages; 6,827 chars; 979 words; 0 zero-text pages; 7 images; usable text | Source PDF is visibly marked “CONFIDENTIAL”; the public corpus does not establish whether that label reflects a publication restriction or an extraction artifact. |
| `sarb_circular_19_2004_capital_hybrid_instruments` | Office of Registrar of Banks / SARB; final circular; official non-binding guidance | 2004-12; official PDF | withdrawn under C1/2026's circular rule | [C1/2026 official PDF](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-circulars/2026/C1-2026.pdf), p. 1 | 20 pages; 37,703 chars; 5,729 words; 0 zero-text pages; 0 images; usable text | C1/2026 does not name this circular individually; withdrawal follows its stated rule for older circulars not confirmed in C1. |
| `sarb_circular_6_2004_basel_ii_update` | Office of Registrar of Banks / SARB; final circular; official non-binding guidance | 2004-05; official PDF | withdrawn under C1/2026's circular rule | [C1/2026 official PDF](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-circulars/2026/C1-2026.pdf), p. 1 | 2 pages; 3,132 chars; 468 words; 0 zero-text pages; 0 images; usable text | Same indirect status evidence as Circular 19/2004; preserve only as historical retrieval material. |
| `nca_act_34_2005` | Parliament of South Africa; final Act; primary legislation; dtic-hosted copy | 2005 Act / 2006 official copy | base Act present; current consolidated currency not established by this PDF alone | [gov.za National Credit Act page](https://www.gov.za/documents/national-credit-act-34-2005) and [dtic-hosted PDF](https://www.thedtic.gov.za/wp-content/uploads/National_Credit_Act34of2005.pdf) | 112 pages; 402,068 chars; 67,496 words; 1 zero-text page; 0 images; usable text | The corpus does not include a versioned amendment/consolidation register. Current legal answers need amendment checking outside this snapshot. |
| `nca_notebook_brochure` | dtic; final explainer brochure; official explanatory material | 2006; official PDF | unknown | [dtic source site](https://www.thedtic.gov.za/) | 22 pages; 28,171 chars; 4,522 words; 0 zero-text pages; 19 images; usable text | No dated official status or supersession notice was found; it must not override the Act or regulations. |
| `ncr_guideline_june_2025_credit_info` | National Credit Regulator; final guideline; official non-binding guidance | 2025-06; official PDF | current in the NCR 2025/2026 guideline collection checked | [NCR guidelines collection](https://ncr.org.za/component/phocadownload/category/241) | 3 pages; 6,075 chars; 937 words; 0 zero-text pages; 3 images; usable text | No public NCR supersession index was found; “current” is collection membership plus dated document, not a legal conclusion. |
| `ncr_guideline_sept_2025_debt_counsellors` | National Credit Regulator; final guideline; official non-binding guidance | 2025-09; official PDF | current in the NCR collection checked | [NCR guidelines collection](https://ncr.org.za/component/phocadownload/category/241) | 4 pages; 5,513 chars; 849 words; 0 zero-text pages; 5 images; usable text | No public supersession index was found. |
| `ncr_guideline_feb_2026_clearance_certificates` | National Credit Regulator; final guideline; official non-binding guidance | 2026-02; official PDF | current in the NCR collection checked | [NCR February 2026 PDF](https://www.ncr.org.za/documents/Guidelines/Guideline%20February%202026.pdf) | 4 pages; 7,787 chars; 1,200 words; 0 zero-text pages; 4 images; usable text | The document is operational guidance; it does not establish the current text of the NCA or regulations. |
| `ncr_guideline_disputed_credit_complaints` | National Credit Regulator; final guideline; official non-binding guidance | 2024-12; official PDF | current in the NCR collection checked | [NCR guidelines collection](https://ncr.org.za/component/phocadownload/category/241) | 4 pages; 7,982 chars; 1,161 words; 0 zero-text pages; 27 images; usable text | No dated NCR supersession notice was located. |
| `fsca_press_conduct_standard_banks_2020` | FSCA; final press release; official explanatory material | 2020-07-08; official PDF | current as a press release only; not the conduct standard | [FSCA publications and resources](https://www.fsca.co.za/Publications-and-Resources/) | 1 page; 1,607 chars; 223 words; 0 zero-text pages; 1 image; usable text | Final Conduct Standard 3 of 2020 is not in the corpus. The press release cannot supply its normative text or current amendment status. |
| `fsca_conduct_standard_otc_derivatives_2018` | FSCA; consultation draft; consultation/discussion material | 2018-04; official consultation PDF | consultation / status unknown; not final | [FSCA Documents for Consultation](https://www.fsca.co.za/Document-For-Consultation/) | 13 pages; 24,162 chars; 3,728 words; 0 zero-text pages; 0 images; usable text | The official index still categorizes the April 2018 OTC document as consultation. Final FMA Conduct Standard 2 of 2018 was not located as an official normative PDF; do not substitute a press release or consultation report. |
| `fsca_rdr_intermediary_segmentation_2019` | FSCA; final regulatory update; official explanatory material | 2019-12; official PDF | unknown | [FSCA Retail Distribution Review materials](https://www.fsca.co.za/Regulatory%20Frameworks/Pages/Retail-Distribution-Review.aspx) | 9 pages; 18,817 chars; 2,669 words; 0 zero-text pages; 16 images; usable text | No current/superseded determination was found in the checked source. |
| `fsca_rdr_2014` | Financial Services Board; consultation discussion paper | 2014; official PDF | historical snapshot | [FSCA Retail Distribution Review materials](https://www.fsca.co.za/Regulatory%20Frameworks/Pages/Retail-Distribution-Review.aspx) | 97 pages; 470,439 chars; 70,690 words; 1 zero-text page; 102 images; usable text | Historical consultation content is not current conduct law. |
| `fsca_tcf_2011` | Financial Services Board; final presentation; official explanatory material | 2011-10; official archived PDF | historical snapshot | [FSCA archived TCF document](https://www2.fsca.co.za/Regulatory%20Frameworks/Archived%20Documents/2011%20-%20Treating%20Customers%20Fairly%20(TCF).pdf) | 11 pages; 4,464 chars; 622 words; 0 zero-text pages; 11 images; usable text | Archived presentation is explanatory and not a substitute for later conduct standards. |
| `ifrs9_project_summary_2014` | IFRS Foundation; final project summary; official explanatory material | 2014-07; official PDF | historical snapshot | [IFRS 9 official standard page](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/) | 32 pages; 47,923 chars; 7,601 words; 0 zero-text pages; 1 image; usable text | It predates later IFRS 9 amendments and is not a current standard text. |
| `ifrs9_issued_2021` | IFRS Foundation; issued text; binding standard text snapshot | 2021; official PDF | historical snapshot, not current consolidated text | [IFRS 9 official standard history](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/) | 188 pages; 486,676 chars; 77,702 words; 1 zero-text page; 0 images; usable text | The official page records 2024 amendments and later standard history; the current licensed consolidated text is not part of this public corpus. |
| `sarb_c1_2026_status_of_circulars` | SARB PA; final circular; official non-binding status notice | 2026-03-09; official PDF and landing page | current status notice | [C1/2026 landing page](https://www.resbank.co.za/en/home/publications/publication-detail-pages/prudential-authority/pa-deposit-takers/banks-circulars/2026/C1-2026) and [PDF](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-circulars/2026/C1-2026.pdf) | 7 pages; 13,768 chars; 1,900 words; 0 zero-text pages; 7 images; usable text | Annual status evidence must be refreshed when a later C1 is published. It does not itself replace the directive text. |

## Unresolved source gaps and refusal scenarios

1. The final normative FMA Conduct Standard 2 of 2018 is not verified in the official corpus
   source set. Retrieval may surface the April 2018 consultation draft, but an answer asking for
   the final standard must state that the final source is unavailable and refuse to treat the
   draft as final.
2. The final normative FSCA Conduct Standard 3 of 2020 for Banks is absent. Retrieval may
   surface the FSCA press release, but an answer asking for the standard's detailed obligations
   must state that the press release is explanatory only and refuse to invent the missing text.
3. The NCA file is the official base Act copy, not a fully versioned current consolidation. A
   current-law question must disclose this limitation and require an external amendment check.
4. The NCR and FSCA pages do not expose a complete public supersession register in the checked
   material. Dated items labelled `current` are safe as dated guidance snapshots, but not proof
   that no later instrument exists.
5. The IFRS 9 current consolidated text is licensed and not in this corpus. Questions about
   current IFRS 9 requirements must cite the 2021 snapshot as historical only and refuse a claim
   that it includes later amendments.

## Documents added in this version

Six documents were added after the review above: `fsr_act_9_2017` (Financial Sector Regulation
Act, from treasury.gov.za), `banks_act_94_1990`, `regs_banks_2012`, `nca_regs_2006` and
`nca_affordability_regs_2015` (all from gov.za), and `fsca_cs3_2020_banks` (Conduct Standard 3 of
2020, from the banking association's copy of Annexure A, OCR'd, see `reports/ocr_check.md`). I
added them because the question set needed the binding instruments behind the guidance notes and
the press release already in the corpus. The manifest now has 28 entries. The count of 22 and the
statement that Conduct Standard 3 of 2020 is absent apply to the audit as it was run.

## Manifest decision

No manifest or source PDF was changed. The existing metadata for the SARB successor relationships
is supported by explicit language in D8/2025 and D4/2023, while the FSCA gaps are already
represented as a consultation draft and a press release rather than being silently promoted to
final instruments. The unresolved gaps above are recorded for the next corpus-authority pass.
