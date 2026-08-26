# RAGAS evaluation summary

Scored 39/45 answerable items from `evals/golden.jsonl` (factual + multi-doc — unanswerable items are checked for refusal separately, not scored on these metrics).

**6 answerable item(s) triggered rag.py's own refusal rule** (retrieval didn't surface enough to answer) and are excluded from the means below — scoring a refusal against metrics built for a substantive answer produces noise, not signal. Listed here rather than silently dropped; see `reports/failure_analysis.md` for why each one failed to retrieve:

- `g03`: Why did Directive 5/2017 need to be updated, according to Directive D3/2023?
- `g24`: What is a debt counsellor required to keep up to date with the NCR under Guideline 004/2025?
- `g28`: What did Proposal J of the Retail Distribution Review address regarding intermediation and outsourced services?
- `g29`: In what year did the FSB (the FSCA's predecessor) carry out the technical work to determine which activities intermediaries were remunerated for, as part of the activity segmentation project?
- `g37`: Which SARB documents were issued under section 6(5) versus section 6(6) of the Banks Act, and what is the practical difference between the two document types they represent?
- `g44`: What common international standard-setting body do Directive D8/2023 and Directive D10/2021 both cite as the source of the principles they implement?

| metric | mean | what it measures |
|---|---|---|
| Faithfulness | 0.829 | Fraction of the answer's claims actually supported by retrieved context — catches hallucination, not wrongness. |
| Answer relevancy | 0.811 | Does the answer address the question asked, independent of whether it's grounded. |
| Context precision | 0.790 | Of the retrieved chunks, how many were actually relevant. |
| Context recall | 0.968 | Of what the reference answer needed, how much retrieval actually surfaced. |

## Per-item scores

| id | question | faithfulness | relevancy | precision | recall |
|---|---|---|---|---|---|
| g01 | How many climate-related disclosure templates are attached to SARB Guidance Note 3/2025 as Annexure 1? | 0.50 | 0.74 | 0.33 | 1.00 |
| g02 | Which earlier SARB directive does Directive D3/2023 on the regulatory treatment of accounting provisions replace? | 1.00 | 0.60 | 1.00 | 1.00 |
| g04 | On what date did the Basel Committee issue the revised standardised and internal ratings-based approaches for credit risk referenced in Directive D8/2023? | 1.00 | 0.85 | 1.00 | 1.00 |
| g05 | Under which section of the Banks Act was Directive D8/2023 on threshold amounts issued? | 1.00 | 0.77 | 1.00 | 1.00 |
| g06 | What seven categories does the BCBS operational resilience paper organise its principles across, per Directive D10/2021? | 0.89 | 0.92 | 1.00 | 1.00 |
| g07 | In what month and year did the Basel Committee on Banking Supervision issue its paper on principles for operational resilience? | 1.00 | 0.99 | 0.83 | 1.00 |
| g08 | By what date were comments due on the proposed amendments to hybrid capital instrument rules in Banks Act Circular 19/2004? | 0.86 | 0.00 | 0.00 | 0.00 |
| g09 | What kind of capital instruments does Banks Act Circular 19/2004 discuss, and why are they called 'hybrid'? | 1.00 | 0.86 | 1.00 | 1.00 |
| g10 | When did the Basel Committee confirm it would publish the full text of Basel II, according to Circular 6/2004? | 0.67 | 0.99 | 1.00 | 1.00 |
| g11 | According to Circular 6/2004, when would the most advanced Basel II approaches be implemented, compared to the standardised and foundation approaches? | 1.00 | 0.89 | 1.00 | 1.00 |
| g12 | On what date did the National Credit Act come into commencement? | 1.00 | 0.95 | 0.33 | 1.00 |
| g13 | Which two pieces of prior legislation did the National Credit Act repeal? | 0.67 | 0.93 | 0.95 | 1.00 |
| g14 | How many business days' notice must a registrant give the National Credit Regulator before voluntarily cancelling its registration under section 58 of the NCA? | 0.80 | 0.83 | 1.00 | 1.00 |
| g15 | Under section 58A of the NCA, what must a registrant attach to its voluntary cancellation notice besides the affidavit? | 1.00 | 0.91 | 0.50 | 1.00 |
| g16 | According to the Notebook on the National Credit Act, what kind of credit-marketing practices does the Act prohibit? | 0.50 | 0.00 | 1.00 | 1.00 |
| g17 | What two bodies did the National Credit Act establish, according to the Notebook overview? | 0.67 | 0.99 | 1.00 | 1.00 |
| g18 | Which section of the National Credit Act gives a consumer the right to challenge the accuracy of information a credit bureau holds about them, without charge? | 0.67 | 0.75 | 1.00 | 1.00 |
| g19 | According to the NCR's disputed credit complaints guideline, what three grounds can make a consumer's information 'challenged' as inaccurate? | 0.83 | 0.94 | 1.00 | 1.00 |
| g20 | How many days does a debt counsellor have to file a certified clearance certificate with credit bureaus under section 71(4) of the NCA? | 0.80 | 0.81 | 0.95 | 1.00 |
| g21 | Which organisation did the NCR designate to provide the authentication platform for filing clearance certificates? | 1.00 | 0.73 | 0.87 | 1.00 |
| g22 | Which sections of the National Credit Act does the June 2025 NCR guideline on consumers under debt review implement? | 0.50 | 0.91 | 0.33 | 1.00 |
| g23 | What problem does the June 2025 NCR guideline identify with how credit providers treat consumers under debt review? | 0.71 | 0.99 | 1.00 | 1.00 |
| g25 | On what date did the FSCA publish the Conduct Standard for Banks? | 1.00 | 0.99 | 1.00 | 1.00 |
| g26 | Which piece of legislation gave the FSCA its mandate to regulate how banks conduct themselves towards customers? | 1.00 | 0.84 | 1.00 | 1.00 |
| g27 | Under which Act and section was the Conduct Standard for authorised OTC derivative providers published? | 0.75 | 0.86 | 1.00 | 1.00 |
| g30 | What regulatory approach shift does the 2014 Retail Distribution Review propose, away from a purely rules-based compliance approach? | 1.00 | 0.92 | 1.00 | 1.00 |
| g31 | What were the agenda topics of the 2011 FSB presentation on Treating Customers Fairly? | 1.00 | 0.99 | 0.50 | 0.75 |
| g32 | What is the 'own credit' issue that IFRS 9 addresses? | 0.67 | 0.83 | 1.00 | 1.00 |
| g33 | What three phases of the IASB's project does the final version of IFRS 9 bring together? | 1.00 | 0.90 | 1.00 | 1.00 |
| g34 | How many classification categories does IFRS 9 have for financial assets, according to PwC's practical guide? | 1.00 | 0.87 | 1.00 | 1.00 |
| g35 | According to PwC's practical guide, at what value are financial instruments initially recognised under IFRS 9? | 0.83 | 0.95 | 0.50 | 1.00 |
| g36 | Both Directive D3/2023 and the IFRS 9 project summary describe a shift from an 'incurred loss' model to a forward-looking model. What is that forward-looking model called? | 1.00 | 0.54 | 1.00 | 1.00 |
| g38 | How does the National Credit Act's stated purpose, as summarised in the Notebook brochure, connect to the NCR's June 2025 guideline on consumers under debt review? | 0.57 | 0.90 | 0.00 | 1.00 |
| g39 | What single section of the National Credit Act empowers the NCR to issue both the June 2025 and September 2025 guidelines discussed in this corpus? | 0.50 | 0.82 | 0.83 | 1.00 |
| g40 | How did the FSCA's 2019 intermediary activity segmentation update build on the 2014 Retail Distribution Review? | 0.26 | 0.89 | 1.00 | 1.00 |
| g41 | Both the Conduct Standard for Banks and the Conduct Standard for OTC Derivative Providers are FSCA conduct standards, but they rest on different underlying legislation. What are the two Acts? | 1.00 | 0.77 | 0.00 | 1.00 |
| g42 | The IFRS 9 project summary and PwC's practical guide both describe IFRS 9 as replacing an earlier standard. Which standard, and for what type of accounting? | 0.90 | 0.73 | 0.89 | 1.00 |
| g43 | Circular 6/2004 and Circular 19/2004 were both issued by the SARB in 2004 to banks. What distinct regulatory topics does each one address? | 1.00 | 0.79 | 0.00 | 1.00 |
| g45 | The National Credit Act itself and the NCR's Notebook brochure both describe the Act's objectives. Name one objective that appears in both sources. | 0.78 | 0.64 | 1.00 | 1.00 |
