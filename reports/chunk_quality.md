# Chunk quality report

- Documents ingested: 28
- Total chunks: 1857
- Token count — min 1, median 572, max 875
- Empty chunks: 0
- Garbage chunks (< 40 tokens or < 50% alphanumeric): 41 (2.2% of total)

![chunk size histogram](figures/chunk_size_histogram.png)

## Per-document breakdown

| doc_id | chunks | median tokens | min | max |
|---|---|---|---|---|
| banks_act_94_1990 | 76 | 538 | 75 | 862 |
| fsca_conduct_standard_otc_derivatives_2018 | 8 | 609 | 509 | 770 |
| fsca_cs3_2020_banks | 16 | 557 | 141 | 788 |
| fsca_press_conduct_standard_banks_2020 | 1 | 306 | 306 | 306 |
| fsca_rdr_2014 | 192 | 523 | 1 | 863 |
| fsca_rdr_intermediary_segmentation_2019 | 5 | 525 | 423 | 616 |
| fsca_tcf_2011 | 2 | 493 | 241 | 493 |
| fsr_act_9_2017 | 851 | 578 | 2 | 871 |
| ifrs9_issued_2021 | 172 | 646 | 122 | 805 |
| ifrs9_project_summary_2014 | 17 | 564 | 473 | 767 |
| nca_act_34_2005 | 314 | 572 | 75 | 875 |
| nca_affordability_regs_2015 | 16 | 621 | 204 | 727 |
| nca_notebook_brochure | 9 | 639 | 536 | 778 |
| nca_regs_2006 | 88 | 504 | 85 | 762 |
| ncr_guideline_disputed_credit_complaints | 3 | 502 | 322 | 575 |
| ncr_guideline_feb_2026_clearance_certificates | 3 | 598 | 513 | 621 |
| ncr_guideline_june_2025_credit_info | 2 | 631 | 301 | 631 |
| ncr_guideline_sept_2025_debt_counsellors | 2 | 494 | 281 | 494 |
| regs_banks_2012 | 34 | 596 | 244 | 809 |
| sarb_c1_2026_status_of_circulars | 5 | 476 | 207 | 606 |
| sarb_circular_19_2004_capital_hybrid_instruments | 14 | 574 | 281 | 760 |
| sarb_circular_6_2004_basel_ii_update | 1 | 614 | 614 | 614 |
| sarb_d10_2021_operational_resilience | 2 | 801 | 448 | 801 |
| sarb_d3_2023_accounting_provisions_ifrs9 | 3 | 460 | 316 | 633 |
| sarb_d4_2023_operational_resilience | 2 | 772 | 411 | 772 |
| sarb_d8_2023_threshold_amounts | 6 | 653 | 287 | 729 |
| sarb_d8_2025_threshold_amounts | 6 | 646 | 468 | 703 |
| sarb_g3_2025_climate_disclosures | 7 | 512 | 471 | 686 |

## Garbage chunks flagged

- `fsca_rdr_2014` chunk 0 (page 3): '1'
- `fsr_act_9_2017` chunk 34 (page 29): 'No.'
- `fsr_act_9_2017` chunk 40 (page 31): 'No.'
- `fsr_act_9_2017` chunk 45 (page 33): 'No.'
- `fsr_act_9_2017` chunk 50 (page 35): 'No.'
- `fsr_act_9_2017` chunk 55 (page 37): 'No.'
- `fsr_act_9_2017` chunk 62 (page 39): 'No.'
- `fsr_act_9_2017` chunk 74 (page 43): 'No.'
- `fsr_act_9_2017` chunk 417 (page 213): 'No.'
- `nca_regs_2006` chunk 33 (page 59): ' A registrant may include its logo or letterhead on a prescribed Form, subject to subregulation (4). (4) (a) it is suffi'
- `nca_regs_2006` chunk 34 (page 63): '31 No.28864 65 1 4 APPLICATION FORM FOR REGISTRATION 2005 Generalinformadon 4 Ihe applicant must submit the completed ap'
- `nca_regs_2006` chunk 36 (page 66): ' MANAGERS OF THE APPLICANT ‘1, Par the purpose of part 3 and Part *management or control” in the Regulations. Appll*mt o'
- `nca_regs_2006` chunk 40 (page 70): ' registration form, eithc (a) explain in detail why the credit provider believes that it has adequate administrative pro'
- `nca_regs_2006` chunk 41 (page 73): ' juristic person, attach proof of authorisation. No. 28864 75 , AS A DEBT COUNSELLOR IN TERMS OF SECTION 44 OF THE NATIO'
- `nca_regs_2006` chunk 42 (page 75): ' name ............................................... ................................................. ............. ..'
- ...and 26 more
