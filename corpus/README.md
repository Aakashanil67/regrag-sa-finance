# Corpus

27 public regulatory and standards documents covering South African banking supervision, consumer
credit law, and IFRS 9. The PDFs themselves aren't committed (see the repo `.gitignore`) — run:

```bash
python -m scripts.fetch_corpus
python -m scripts.validate_manifest
```

This reads `manifest.json`, downloads anything missing, and verifies each file's SHA-256 against
the hash pinned when the corpus was built, so a byte-identical corpus is reproducible from a clean
clone. If a checksum fails, the source document has changed since then; the script reports it and
skips saving rather than silently ingesting a different file than the one this project's reports
and eval results were built against. `validate_manifest` enforces the authority/stage/status
contract in `manifest.schema.json` — a manifest entry can't ship with an invalid or missing
authority field.

SARB, NCR and FSCA all sit behind a WAF that rejects bare `curl`/`requests` calls with a 200 that's
actually an HTML rejection page — the fetch script sends a real browser user agent and a same-site
`Referer`, which is enough to pass.

## Authority and currency

Every entry in `manifest.json` records `authority_level` (primary legislation, binding
instrument, non-binding guidance, explanatory material, consultation/discussion, or third-party
commentary), `publication_stage` (final/consultation/draft), and `current_status` (current,
withdrawn, superseded, historical snapshot, or unknown) — see `manifest.schema.json` for the full
contract. A source with `current_status` other than `current` carries a `status_source_url` (and
often a `status_source_id`/`status_source_page` pointing at another corpus document) naming the
evidence for that status; the product renders that as a source notice separate from the model's
own answer.

A 2026 audit corrected several sources that had drifted from their stated authority:

- The active OTC-derivatives conduct standard is still the **April 2018 consultation draft**
  ("the Authority hereby publish for comments") — its own metadata was previously silent on this,
  making it read as equivalent to a final standard. The final FMA Conduct Standard 2 of 2018 is
  the intended replacement but its current live URL on the FSCA site (a JS-rendered SPA) could not
  be resolved for this release; this remains a known gap, not a resolved correction.
- **SARB Circular C1/2026** ("Status of previously issued circulars") was added. It deems every
  earlier Banks Act circular withdrawn, terminated or replaced unless confirmed in that year's
  Circular 1 — and neither 2004 circular in this corpus appears in its confirmed list. Both are
  now marked `withdrawn`, with C1/2026 as evidence.
- C1/2026's own effective-directives list also revealed that **Directive 8/2023** (threshold
  amounts) and **Directive 10/2021** (operational resilience) have been superseded by **Directive
  8/2025** and **Directive 4/2023** respectively — both same-subject, later directives that *are*
  confirmed in C1/2026. All four documents are now in the corpus with corrected status.
- The **PwC IFRS 9 guide** was removed: its metadata claimed 2017, but the PDF's own creation-date
  metadata is 2011-01-06 and its text describes the pre-2014, two-category IFRS 9 model. It is
  replaced by the **official 2021 issued IFRS 9 text**, dated as a historical snapshot — the IFRS
  Foundation's current standard page records later amendments (including 2024 amendments) that
  this dated text does not reflect. The removed file is kept, unindexed, under
  `corpus/archive-local/` as evidence of the original error.
- 2004-era issuers were corrected from "SARB Prudential Authority" (created 2018) to the
  historical Office of the Registrar of Banks/South African Reserve Bank. The National Credit
  Act's `issuing_authority` is now Parliament of South Africa, with the dtic recorded as
  `publisher` (the host of this copy, not the Act's legal author).

## Documents

### Primary legislation and regulations (5)

| Document | Status |
|---|---|
| [Financial Sector Regulation Act 9 of 2017](https://www.treasury.gov.za/legislation/acts/2017/Act%209%20of%202017%20FinanSectorRegulation.pdf) | as assented, 22 August 2017 (historical snapshot; later amended) |
| [Banks Act 94 of 1990](https://www.gov.za/sites/default/files/gcis_document/201503/act-94-1990s.pdf) | as first published, 11 July 1990, under its original title (historical snapshot; heavily amended since) |
| [Regulations relating to Banks (GN R1029, 12 December 2012)](https://www.gov.za/sites/default/files/gcis_document/201409/35950rg9872gon10291.pdf) | as published (historical snapshot; amended in 2015, 2016, 2020, 2022) |
| [National Credit Regulations, 2006 (GN R489, 31 May 2006)](https://www.gov.za/sites/default/files/gcis_document/201409/28864.pdf) | as published (historical snapshot) |
| [National Credit Regulations including Affordability Assessment Regulations (GN R202, 13 March 2015)](https://www.gov.za/sites/default/files/gcis_document/201503/38557rg10382gon202.pdf) | current |

The Banks Act copy is a scanned gazette with OCR text, so its text quality is uneven. I could not
find a consolidated, text-native copy on an official site: the SARB page links only to Sabinet,
a paid third-party database.

### SARB Prudential Authority (9)

| Document | Status |
|---|---|
| [Guidance Note 3/2025: Climate-related Disclosures for Banks](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-guidance-notes/2025/g3-2025/G3-2025-%20Guidance%20Note%20Climate%20Disclosures%20for%20banks.pdf) | current |
| [Directive 3/2023: Regulatory Treatment of Accounting Provisions (IFRS 9)](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-directives/2023/D3-2023-Regulatory%20treatment%20of%20accounting%20provisions.pdf) | current |
| [Directive 8/2023: Threshold Amounts, Revised STA/IRB Credit Risk & Liquidity Risk](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-directives/2023/D8-2023%20-%20Threshold%20amounts%20related%20to%20the%20revised%20standardised%20and%20IRB%20approaches%20for%20credit%20risk%20and%20the%20liquidity%20risk%20framework.pdf) | superseded by D8/2025 |
| [Directive 8/2025: Threshold Amounts, Revised STA/IRB Credit Risk](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-directives/2025/D8-2025%20-%20Thresholds%20Amounts%20to%20the%20revised%20Standardised%20and%20IRB%20approaches%20for%20credit%20Risk.pdf) | current |
| [Directive 10/2021: Principles for Operational Resilience](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-directives/2021/D10-2021%20-Directive%20on%20Operational%20Resilience.pdf) | superseded by D4/2023 |
| [Directive 4/2023: Directive on Operational Resilience](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-directives/2023/D4-2023%20-%20Directive%20on%20operational%20resilience.pdf) | current |
| [Banks Act Circular 19/2004: Qualifying Capital, Hybrid Debt Instruments](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-circulars/2004/3928/Banks-Act-Circular-19-of-2004.pdf) | withdrawn (per C1/2026) |
| [Banks Act Circular 6/2004: Update on Basel II](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-circulars/2004/3921/Banks-Act-Circular-6-of-2004.pdf) | withdrawn (per C1/2026) |
| [Circular C1/2026: Status of Previously Issued Circulars](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-circulars/2026/C1-2026.pdf) | current |

The two 2004 circulars are kept, correctly flagged as withdrawn, because they're real,
substantive regulatory text useful for retrieval/chunking evaluation, not because they're current
law — any answer that cites one carries a withdrawn-source notice.

### National Credit Act / NCR (6)

| Document | Status |
|---|---|
| [National Credit Act 34 of 2005](https://www.thedtic.gov.za/wp-content/uploads/National_Credit_Act34of2005.pdf) | current |
| [Notebook on the National Credit Act (plain-language explainer)](https://www.thedtic.gov.za/wp-content/uploads/NCA_Brochure.pdf) | unknown |
| [NCR Guideline (June 2025): NCA ss.69(2), 70(1), 70(2), 71(5) — credit information](https://www.ncr.org.za/documents/Guidelines/Guideline%203%20June%202025.pdf) | current |
| [NCR Guideline (September 2025): Debt Counsellor Contact Information](https://www.ncr.org.za/documents/Guidelines/Guideline%20September%202025.pdf) | current |
| [NCR Guideline (February 2026): Clearance Certificates (Form 19)](https://www.ncr.org.za/documents/Guidelines/Guideline%20February%202026.pdf) | current |
| [NCR Guidelines: Submission of Disputed Consumer Credit Information Complaints](https://www.ncr.org.za/phocadownload/GUIDELINES%20FOR%20THE%20SUBMISSION%20OF%20COMPLAINTS%20RELATING%20TO%20DISPUTED%20CONSUMER%20CREDIT%20INFORMATION1.pdf) | current |

### FSCA (5)

| Document | Status |
|---|---|
| [Press Release: Conduct Standard for Banks (8 July 2020)](https://www2.fsca.co.za/News%20Documents/FSCA%20Press%20Release%20-%20Conduct%20Standard%20for%20Banks%208%20July%202020.pdf) | current (as a press release; not a substitute for Conduct Standard 3 of 2020 itself, which this corpus does not carry — see below) |
| [Conduct Standard for Authorised OTC Derivative Providers (April 2018, consultation draft)](https://www2.fsca.co.za/Regulatory%20Frameworks/Documents%20for%20Consultation/Conduct%20Standard%20for%20authorised%20over-the-counter%20derivative%20providers%20April%202018.pdf) | consultation draft — final standard not resolved this release |
| [Retail Distribution Review: Intermediary Activity Segmentation (December 2019)](https://www2.fsca.co.za/Regulatory%20Frameworks/Regulatory%20Frameworks%20Documents/Retail%20Distribution%20Review%20Intermediary%20Activity%20Segmentation%20and%20Related%20Matters%20December%202019.pdf) | unknown |
| [Retail Distribution Review (2014, discussion paper)](https://www2.fsca.co.za/Regulatory%20Frameworks/Documents%20for%20Consultation/FSB%20Retail%20Distribution%20Review%202014.pdf) | historical snapshot |
| [Treating Customers Fairly (TCF) (2011)](https://www2.fsca.co.za/Regulatory%20Frameworks/Archived%20Documents/2011%20-%20Treating%20Customers%20Fairly%20(TCF).pdf) | historical snapshot |

**Known gap:** neither the final FMA Conduct Standard 2 of 2018 nor FSCA Conduct Standard 3 of
2020 (Banks) is in this corpus. The only obtainable copy of the latter (a Banking Association of
South Africa mirror) is a scanned image PDF with no extractable text layer — ingesting it would
have silently produced zero retrievable chunks, so it was rejected rather than added. Both remain
follow-up work.

### IFRS 9 (2)

| Document | Status |
|---|---|
| [IFRS 9 Financial Instruments: Project Summary](https://www.ifrs.org/-/media/project/fi-impairment/ifrs-standard/published-documents/project-summary-july-2014.pdf) | historical snapshot |
| [IFRS 9 Financial Instruments (issued text, 2021 edition)](https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf) | Accounting standard (binding on reporting entities through financial reporting law, not a PA instrument); historical snapshot — IFRS.org's current standard page records later amendments this text doesn't reflect |

The full current IFRS 9 standard text itself isn't included — the IFRS Foundation licenses that
separately and doesn't distribute it freely; the 2021 issued edition is a freely-available, dated
snapshot, accurate as of its own publication but not a substitute for checking the current
standard for later amendments.

## Not in the corpus

- Consolidated Banks Act 94 of 1990 with all amendments. Tried
  `https://www.resbank.co.za/en/home/publications/prudential-authority/legislation/banks-act-1990-act-no-94-of-1990`,
  which links only to a Sabinet page; the gov.za copy above is the 1990 original.
- Final FMA Conduct Standard 2 of 2018 and FSCA Conduct Standard 3 of 2020 (see the known gap
  under FSCA).

## Manual fallback

If a regulator's site restructures and a URL 404s, or the WAF workaround above stops working:
download the document by hand from the same public page, drop it in `corpus/` under the filename
`manifest.json` expects, then update that entry's `sha256` (recompute with
`python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" corpus/<file>.pdf`)
so the pin stays honest about what's actually in the corpus. Before adding any new entry, open its
first pages and check title, issuer, and instrument type against the official landing page —
`scripts/validate_manifest.py` checks the manifest's shape, not whether its claims are true.
