# Chunk quality report

- Documents ingested: 28
- Total chunks: 6161
- Token count — min 1, median 182, max 452
- Empty chunks: 0
- Garbage chunks (< 40 tokens or < 50% alphanumeric): 825 (13.4% of total)

![chunk size histogram](figures/chunk_size_histogram.png)

## Per-document breakdown

| doc_id | chunks | median tokens | min | max |
|---|---|---|---|---|
| banks_act_94_1990 | 252 | 182 | 8 | 335 |
| fsca_conduct_standard_otc_derivatives_2018 | 31 | 186 | 33 | 237 |
| fsca_cs3_2020_banks | 60 | 184 | 1 | 320 |
| fsca_press_conduct_standard_banks_2020 | 2 | 244 | 125 | 244 |
| fsca_rdr_2014 | 543 | 197 | 1 | 295 |
| fsca_rdr_intermediary_segmentation_2019 | 18 | 163 | 44 | 234 |
| fsca_tcf_2011 | 4 | 197 | 170 | 225 |
| fsr_act_9_2017 | 2914 | 185 | 1 | 320 |
| ifrs9_issued_2021 | 648 | 193 | 14 | 270 |
| ifrs9_project_summary_2014 | 61 | 184 | 29 | 253 |
| nca_act_34_2005 | 551 | 299 | 3 | 452 |
| nca_affordability_regs_2015 | 57 | 175 | 1 | 248 |
| nca_notebook_brochure | 35 | 181 | 48 | 246 |
| nca_regs_2006 | 672 | 39 | 2 | 248 |
| ncr_guideline_disputed_credit_complaints | 10 | 155 | 105 | 207 |
| ncr_guideline_feb_2026_clearance_certificates | 10 | 216 | 30 | 232 |
| ncr_guideline_june_2025_credit_info | 7 | 148 | 129 | 174 |
| ncr_guideline_sept_2025_debt_counsellors | 5 | 153 | 104 | 211 |
| regs_banks_2012 | 119 | 175 | 6 | 369 |
| sarb_c1_2026_status_of_circulars | 15 | 154 | 13 | 242 |
| sarb_circular_19_2004_capital_hybrid_instruments | 46 | 193 | 30 | 250 |
| sarb_circular_6_2004_basel_ii_update | 4 | 175 | 139 | 237 |
| sarb_d10_2021_operational_resilience | 9 | 193 | 13 | 234 |
| sarb_d3_2023_accounting_provisions_ifrs9 | 8 | 236 | 13 | 252 |
| sarb_d4_2023_operational_resilience | 10 | 153 | 13 | 219 |
| sarb_d8_2023_threshold_amounts | 21 | 193 | 13 | 240 |
| sarb_d8_2025_threshold_amounts | 24 | 197 | 13 | 242 |
| sarb_g3_2025_climate_disclosures | 25 | 166 | 13 | 233 |

## Garbage chunks flagged

- `sarb_g3_2025_climate_disclosures` chunk 0 (page 1): 'P O Box 427 Pretoria  0001 South Africa'
- `sarb_g3_2025_climate_disclosures` chunk 3 (page 1): 'african financial sector exposed to financial and non - financial impacts through the location of assets and liabilities'
- `sarb_d3_2023_accounting_provisions_ifrs9` chunk 0 (page 1): 'P O Box 427 Pretoria  0001 South Africa'
- `sarb_d3_2023_accounting_provisions_ifrs9` chunk 3 (page 1): '##s given that ias 39 impairments were based on an incurred loss model which required a loss event to occur prior to a p'
- `sarb_d8_2023_threshold_amounts` chunk 0 (page 1): 'P O Box 427 Pretoria  0001 South Africa'
- `sarb_d8_2025_threshold_amounts` chunk 0 (page 1): 'P O Box 427 Pretoria  0001 South Africa'
- `sarb_d10_2021_operational_resilience` chunk 0 (page 1): 'P O Box 427 Pretoria  0001 South Africa'
- `sarb_d4_2023_operational_resilience` chunk 0 (page 1): 'P O Box 427 Pretoria  0001 South Africa'
- `sarb_circular_19_2004_capital_hybrid_instruments` chunk 4 (page 2): 'of capital and reserve funds may be unable to absorb unexpected losses, thereby increasing the risk of bank failure and '
- `sarb_circular_19_2004_capital_hybrid_instruments` chunk 22 (page 11): 'and supervisory requirements and developments, it has become necessary to amend the provisions of regulations 21 ( 6a ) '
- `nca_act_34_2005` chunk 8 (page 3): 'reckless credit agreement 83. declaration of reckless credit agreement 84. effect of suspension of credit agreement 85. '
- `nca_act_34_2005` chunk 11 (page 4): 'consumer may terminate agreement 123. termination of agreement by credit provider chapter 6 collection, repayment, surre'
- `nca_act_34_2005` chunk 28 (page 9): 'date on which the act of extinguishment becomes effective ; ( pending amendment : definition of “ extinguish ” to be ins'
- `nca_act_34_2005` chunk 40 (page 11): '##piry of a defined period to sell the goods and retain all the proceeds of the sale in settlement of the consumer ’ s o'
- `nca_act_34_2005` chunk 47 (page 13): '. 102 of 1996 ) ; “ south african reserve bank ” has the meaning set out in the south african reserve bank act, 1989 ( a'
- ...and 810 more
