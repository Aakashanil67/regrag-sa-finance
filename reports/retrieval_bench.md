# Retrieval benchmark

**Split: holdout.** Development-set numbers guide tuning; only a `holdout` run, executed once against a frozen pipeline, is release evidence.

**Hit-rate@k**: fraction of questions where the source document/page appears anywhere in the top k retrieved chunks — what a user actually experiences, since the RAG layer only sees the top k.

**MRR** (mean reciprocal rank): averages 1/rank of the first correct chunk across all 10 retrieved results — rewards ranking the right answer 1st over merely including it somewhere in the list.

| metric | value | 95% CI |
|---|---|---|
| hit-rate@3 | 27/30 (90%) | 74%–97% |
| hit-rate@5 | 28/30 (93%) | 79%–98% |
| hit-rate@10 | 28/30 (93%) | 79%–98% |
| MRR | 0.792 | — |

## Per-question results

| id | question | expected | first hit rank |
|---|---|---|---|
| rh01 | What BCBS framework and IFRS standard does Guidance Note 3/2025 say informed its climate-related disclosure requirements for banks? | sarb_g3_2025_climate_disclosures p.1 | 1 |
| rh02 | In what year does Directive 3/2023 say banks commenced reporting under IFRS 9's new requirements? | sarb_d3_2023_accounting_provisions_ifrs9 p.1 | 1 |
| rh03 | What two risk frameworks does Directive 8/2023 set threshold amounts for? | sarb_d8_2023_threshold_amounts p.1 | 3 |
| rh04 | What three risk frameworks does Directive 8/2025 set threshold amounts for? | sarb_d8_2025_threshold_amounts p.1 | 1 |
| rh05 | What kinds of events does Directive 10/2021 cite as having demonstrated the consequences of operational failures? | sarb_d10_2021_operational_resilience p.1 | 2 |
| rh06 | In what month and year did the Basel Committee issue the paper on principles for operational resilience referenced in Directive 4/2023? | sarb_d4_2023_operational_resilience p.1 | 1 |
| rh07 | What date is Banks Act Circular 19/2004 dated? | sarb_circular_19_2004_capital_hybrid_instruments p.1 | 3 |
| rh08 | What date is Banks Act Circular 6/2004 dated? | sarb_circular_6_2004_basel_ii_update p.1 | 2 |
| rh09 | Under which section of the Banks Act is Circular C1/2026 issued? | sarb_c1_2026_status_of_circulars p.1 | 1 |
| rh10 | On what date was the National Credit Act assented to? | nca_act_34_2005 p.1 | 1 |
| rh11 | What body does the National Credit Act establish alongside the National Credit Regulator, per its notebook explainer? | nca_notebook_brochure p.2 | 2 |
| rh12 | What section of the National Credit Act empowers the NCR to issue explanatory notices interpreting the Act, per Guideline 003/2025? | ncr_guideline_june_2025_credit_info p.2 | 1 |
| rh13 | How many business days are debt counsellors given to update their contact information on the DHS, per Guideline 004/2025? | ncr_guideline_sept_2025_debt_counsellors p.4 | 3 |
| rh14 | Which organisation has the NCR designated to provide an authentication platform for filing clearance certificates, per Guideline 001/2026? | ncr_guideline_feb_2026_clearance_certificates p.2 | 1 |
| rh15 | What form must a consumer complete to ask the NCR to investigate disputed credit information, per Guideline 005/2024? | ncr_guideline_disputed_credit_complaints p.3 | 1 |
| rh16 | In what year was the Financial Sector Regulation Act introduced, giving the FSCA its mandate over banking conduct, per this press release? | fsca_press_conduct_standard_banks_2020 p.1 | 1 |
| rh17 | Under which Act was the Conduct Standard for OTC derivative providers published for comment? | fsca_conduct_standard_otc_derivatives_2018 p.1 | 1 |
| rh18 | In what month and year did the FSB publish the original Retail Distribution Review discussed in this 2019 update? | fsca_rdr_intermediary_segmentation_2019 p.2 | 4 |
| rh19 | What Act does the 2014 Retail Distribution Review credit with raising intermediary professionalism, despite remaining concerns? | fsca_rdr_2014 p.3 | 1 |
| rh20 | Under which regulatory model does the 2011 TCF presentation say Treating Customers Fairly forms part of the market conduct 'peak'? | fsca_tcf_2011 p.3 | 1 |
| rh21 | What is the mandatory effective date of IFRS 9 according to the 2014 project summary? | ifrs9_project_summary_2014 p.3 | miss (not in top 10) |
| rh22 | How many categories does the 2021 issued IFRS 9 text classify financial assets into for subsequent measurement? | ifrs9_issued_2021 p.18 | miss (not in top 10) |
| rh23 | Which two SARB directives, one from 2021 and one from 2023, share the exact title 'Principles for operational resilience'? | sarb_d4_2023_operational_resilience p.1 | 1 |
| rh24 | Does Circular C1/2026 confirm Banks Act Circular 19/2004 as still effective? | sarb_c1_2026_status_of_circulars p.1 | 1 |
| rh25 | Which 2025 SARB directive covers the same threshold-amounts subject as Directive 8/2023? | sarb_d8_2025_threshold_amounts p.1 | 1 |
| rh26 | Which FSCA predecessor body published the original 2014 Retail Distribution Review discussed in the 2019 intermediary segmentation update? | fsca_rdr_intermediary_segmentation_2019 p.2 | 1 |
| rh27 | Which earlier standard do both the IFRS 9 project summary and the 2021 issued IFRS 9 text say IFRS 9 replaces? | ifrs9_issued_2021 p.1 | 1 |
| rh28 | What Act number do both the National Credit Act itself and its plain-language notebook cite for the National Credit Act, 2005? | nca_notebook_brochure p.1 | 1 |
| rh29 | How is a 'Primary Credit Bureau' defined in Guideline 005/2024? | ncr_guideline_disputed_credit_complaints p.2 | 1 |
| rh30 | What does Directive 3/2023 say IFRS 9 represented for banks' accounting practice? | sarb_d3_2023_accounting_provisions_ifrs9 p.1 | 1 |
