# Corpus

19 public regulatory and standards documents covering South African banking supervision, consumer
credit law, and IFRS 9. The PDFs themselves aren't committed (see the repo `.gitignore`) — run:

```bash
python -m scripts.fetch_corpus
```

This reads `manifest.json`, downloads anything missing, and verifies each file's SHA-256 against
the hash pinned when the corpus was built (2026-08-24), so a byte-identical corpus is
reproducible from a clean clone. If a checksum fails, the source document has changed since then;
the script reports it and skips saving rather than silently ingesting a different file than the
one this project's reports and eval results were built against.

SARB, NCR and FSCA all sit behind a WAF that rejects bare `curl`/`requests` calls with a 200 that's
actually an HTML rejection page — the fetch script sends a real browser user agent and a same-site
`Referer`, which is enough to pass.

## Documents

### SARB Prudential Authority (6)

| Document | Year |
|---|---|
| [Guidance Note 3/2025: Climate-related Disclosures for Banks](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-guidance-notes/2025/g3-2025/G3-2025-%20Guidance%20Note%20Climate%20Disclosures%20for%20banks.pdf) | 2025 |
| [Directive 3/2023: Regulatory Treatment of Accounting Provisions (IFRS 9)](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-directives/2023/D3-2023-Regulatory%20treatment%20of%20accounting%20provisions.pdf) | 2023 |
| [Directive 8/2023: Threshold Amounts, Revised STA/IRB Credit Risk & Liquidity Risk](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-directives/2023/D8-2023%20-%20Threshold%20amounts%20related%20to%20the%20revised%20standardised%20and%20IRB%20approaches%20for%20credit%20risk%20and%20the%20liquidity%20risk%20framework.pdf) | 2023 |
| [Directive 10/2021: Principles for Operational Resilience](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-directives/2021/D10-2021%20-Directive%20on%20Operational%20Resilience.pdf) | 2021 |
| [Banks Act Circular 19/2004: Qualifying Capital, Hybrid Debt Instruments](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-circulars/2004/3928/Banks-Act-Circular-19-of-2004.pdf) | 2004 |
| [Banks Act Circular 6/2004: Update on Basel II](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-circulars/2004/3921/Banks-Act-Circular-6-of-2004.pdf) | 2004 |

The two 2004 circulars are historical (Basel II implementation, superseded many times over since)
— kept deliberately because they're real, substantive regulatory text rather than a thin index
page, which matters more for a retrieval/chunking corpus than currency does.

### National Credit Act / NCR (6)

| Document | Year |
|---|---|
| [National Credit Act 34 of 2005](https://www.thedtic.gov.za/wp-content/uploads/National_Credit_Act34of2005.pdf) | 2005 |
| [Notebook on the National Credit Act (plain-language explainer)](https://www.thedtic.gov.za/wp-content/uploads/NCA_Brochure.pdf) | 2006 |
| [NCR Guideline (June 2025): NCA ss.69(2), 70(1), 70(2), 71(5) — credit information](https://www.ncr.org.za/documents/Guidelines/Guideline%203%20June%202025.pdf) | 2025 |
| [NCR Guideline (September 2025): Debt Counsellor Contact Information](https://www.ncr.org.za/documents/Guidelines/Guideline%20September%202025.pdf) | 2025 |
| [NCR Guideline (February 2026): Clearance Certificates (Form 19)](https://www.ncr.org.za/documents/Guidelines/Guideline%20February%202026.pdf) | 2026 |
| [NCR Guidelines: Submission of Disputed Consumer Credit Information Complaints](https://www.ncr.org.za/phocadownload/GUIDELINES%20FOR%20THE%20SUBMISSION%20OF%20COMPLAINTS%20RELATING%20TO%20DISPUTED%20CONSUMER%20CREDIT%20INFORMATION1.pdf) | 2024 |

### FSCA (5)

| Document | Year |
|---|---|
| [Press Release: Conduct Standard for Banks (8 July 2020)](https://www2.fsca.co.za/News%20Documents/FSCA%20Press%20Release%20-%20Conduct%20Standard%20for%20Banks%208%20July%202020.pdf) | 2020 |
| [Conduct Standard for Authorised OTC Derivative Providers (April 2018)](https://www2.fsca.co.za/Regulatory%20Frameworks/Documents%20for%20Consultation/Conduct%20Standard%20for%20authorised%20over-the-counter%20derivative%20providers%20April%202018.pdf) | 2018 |
| [Retail Distribution Review: Intermediary Activity Segmentation (December 2019)](https://www2.fsca.co.za/Regulatory%20Frameworks/Regulatory%20Frameworks%20Documents/Retail%20Distribution%20Review%20Intermediary%20Activity%20Segmentation%20and%20Related%20Matters%20December%202019.pdf) | 2019 |
| [Retail Distribution Review (2014)](https://www2.fsca.co.za/Regulatory%20Frameworks/Documents%20for%20Consultation/FSB%20Retail%20Distribution%20Review%202014.pdf) | 2014 |
| [Treating Customers Fairly (TCF) (2011)](https://www2.fsca.co.za/Regulatory%20Frameworks/Archived%20Documents/2011%20-%20Treating%20Customers%20Fairly%20(TCF).pdf) | 2011 |

### IFRS 9 (2)

| Document | Year |
|---|---|
| [IFRS 9 Financial Instruments: Project Summary](https://www.ifrs.org/-/media/project/fi-impairment/ifrs-standard/published-documents/project-summary-july-2014.pdf) | 2014 |
| [PwC: Practical Guide to IFRS 9 Financial Instruments](https://www.pwc.in/services/ifrs/ifrs-assets/practical_guide_on_financial_instrument_accounting_ifrs_9.pdf) | 2011 |

The full IFRS 9 standard text itself isn't included — the IFRS Foundation licenses that
separately and doesn't distribute it freely; the Project Summary and the PwC/Big-4 practitioner
guides are the freely-available, IFRS 9-accurate substitute.

## Manual fallback

If a regulator's site restructures and a URL 404s, or the WAF workaround above stops working:
download the document by hand from the same public page, drop it in `corpus/` under the filename
`manifest.json` expects, then update that entry's `sha256` (recompute with
`python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" corpus/<file>.pdf`)
so the pin stays honest about what's actually in the corpus.
