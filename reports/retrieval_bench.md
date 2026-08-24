# Retrieval benchmark

**Hit-rate@k**: fraction of questions where the source document/page appears anywhere in the top k retrieved chunks — what a user actually experiences, since the RAG layer only sees the top k.

**MRR** (mean reciprocal rank): averages 1/rank of the first correct chunk across all 10 retrieved results — rewards ranking the right answer 1st over merely including it somewhere in the list.

| metric | value |
|---|---|
| hit-rate@3 | 80% |
| hit-rate@5 | 85% |
| hit-rate@10 | 85% |
| MRR | 0.654 |

## Per-question results

| id | question | expected | first hit rank |
|---|---|---|---|
| r01 | How many climate-related disclosure templates are attached to Guidance Note 3/2025 as Annexure 1? | sarb_g3_2025_climate_disclosures p.1 | 1 |
| r02 | Which earlier directive does Directive D3/2023 on the regulatory treatment of accounting provisions replace? | sarb_d3_2023_accounting_provisions_ifrs9 p.1 | miss (not in top 10) |
| r03 | On what date did the Basel Committee issue the revised standardised and internal ratings-based approaches for credit risk referenced in Directive D8/2023? | sarb_d8_2023_threshold_amounts p.1 | 2 |
| r04 | What seven categories does the BCBS operational resilience paper organise its principles across, per Directive D10/2021? | sarb_d10_2021_operational_resilience p.1 | 1 |
| r05 | By what date were comments due on the proposed amendments to hybrid debt instrument rules in Banks Act Circular 19/2004? | sarb_circular_19_2004_capital_hybrid_instruments p.1 | 1 |
| r06 | When did the Basel Committee confirm it would publish the full text of Basel II, according to Circular 6/2004? | sarb_circular_6_2004_basel_ii_update p.1 | 1 |
| r07 | On what date did the National Credit Act come into commencement? | nca_act_34_2005 p.1 | 3 |
| r08 | How many business days' notice must a registrant give the National Credit Regulator before voluntarily cancelling its registration under section 58? | nca_act_34_2005 p.40 | 1 |
| r09 | According to the Notebook on the National Credit Act, what kind of credit-marketing practices does the Act prohibit? | nca_notebook_brochure p.1 | 1 |
| r10 | Which section of the National Credit Act gives a consumer the right to challenge the accuracy of information held about them by a credit bureau? | ncr_guideline_disputed_credit_complaints p.2 | 1 |
| r11 | How many days does a debt counsellor have to file a certified clearance certificate with credit bureaus under section 71(4) of the NCA? | ncr_guideline_feb_2026_clearance_certificates p.2 | 2 |
| r12 | Which sections of the National Credit Act does the June 2025 NCR guideline on consumers under debt review implement? | ncr_guideline_june_2025_credit_info p.2 | 2 |
| r13 | What is a debt counsellor required to keep up to date with the NCR under Guideline 004/2025? | ncr_guideline_sept_2025_debt_counsellors p.2 | 4 |
| r14 | On what date did the FSCA publish the Conduct Standard for Banks? | fsca_press_conduct_standard_banks_2020 p.1 | 1 |
| r15 | Under which Act and section was the Conduct Standard for authorised OTC derivative providers published? | fsca_conduct_standard_otc_derivatives_2018 p.1 | 2 |
| r16 | What did Proposal J of the Retail Distribution Review address regarding intermediation and outsourced services? | fsca_rdr_intermediary_segmentation_2019 p.2 | 2 |
| r17 | What regulatory approach shift does the 2014 Retail Distribution Review propose, away from a purely rules-based compliance approach? | fsca_rdr_2014 p.3 | 1 |
| r18 | What were the agenda topics of the 2011 FSB presentation on Treating Customers Fairly? | fsca_tcf_2011 p.1 | miss (not in top 10) |
| r19 | What is the 'own credit' issue that IFRS 9 addresses? | ifrs9_project_summary_2014 p.1 | miss (not in top 10) |
| r20 | How many classification categories does IFRS 9 have for financial assets, according to PwC's practical guide? | pwc_practical_guide_ifrs9 p.1 | 1 |
