# Chunk quality report

- Documents ingested: 19
- Total chunks: 635
- Token count — min 1, median 561, max 875
- Empty chunks: 0
- Garbage chunks (< 40 tokens or < 50% alphanumeric): 1 (0.2% of total)

![chunk size histogram](figures/chunk_size_histogram.png)

## Per-document breakdown

| doc_id | chunks | median tokens | min | max |
|---|---|---|---|---|
| fsca_conduct_standard_otc_derivatives_2018 | 8 | 609 | 509 | 770 |
| fsca_press_conduct_standard_banks_2020 | 1 | 306 | 306 | 306 |
| fsca_rdr_2014 | 191 | 523 | 1 | 863 |
| fsca_rdr_intermediary_segmentation_2019 | 5 | 525 | 423 | 616 |
| fsca_tcf_2011 | 2 | 493 | 241 | 493 |
| ifrs9_project_summary_2014 | 17 | 564 | 473 | 767 |
| nca_act_34_2005 | 333 | 572 | 75 | 875 |
| nca_notebook_brochure | 9 | 639 | 536 | 778 |
| ncr_guideline_disputed_credit_complaints | 3 | 502 | 322 | 575 |
| ncr_guideline_feb_2026_clearance_certificates | 3 | 598 | 513 | 621 |
| ncr_guideline_june_2025_credit_info | 2 | 631 | 301 | 631 |
| ncr_guideline_sept_2025_debt_counsellors | 2 | 494 | 281 | 494 |
| pwc_practical_guide_ifrs9 | 25 | 625 | 75 | 832 |
| sarb_circular_19_2004_capital_hybrid_instruments | 14 | 574 | 281 | 760 |
| sarb_circular_6_2004_basel_ii_update | 1 | 614 | 614 | 614 |
| sarb_d10_2021_operational_resilience | 3 | 448 | 364 | 524 |
| sarb_d3_2023_accounting_provisions_ifrs9 | 3 | 472 | 316 | 633 |
| sarb_d8_2023_threshold_amounts | 6 | 653 | 287 | 729 |
| sarb_g3_2025_climate_disclosures | 7 | 512 | 471 | 698 |

## Garbage chunks flagged

- `fsca_rdr_2014` chunk 0 (page 3): '1'
