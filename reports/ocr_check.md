# OCR check: Conduct Standard 3 of 2020 (Banks)

Tesseract 5 at 300 dpi through PyMuPDF. I compared the first 100 words of 10 pages against the page image. Page 1 is a cover page with only 81 words, so it counts 81.

| page | OCR characters | word errors in first 100 words | example error |
|---|---|---|---|
| 1 | 528 | 9 | handwritten "03 July 2020" read as "© 3 Ju Hi 90 20" |
| 3 | 2444 | 3 | "(6)" for "(b)" |
| 5 | 2251 | 8 | "asmay" for "as may" |
| 7 | 1830 | 8 | "Abank" for "A bank" |
| 9 | 2105 | 4 | "sultability" for "suitability" |
| 11 | 2120 | 1 | "retall" for "retail" |
| 14 | 1995 | 2 | "orbenefit" for "or benefit" |
| 17 | 1954 | 4 | "thelr" for "their" |
| 20 | 1758 | 4 | "uillised" for "utilised" |
| 22 | 1874 | 4 | "whenit" for "when it" |

Overall: 47 errors in 981 words (about 4.8%). The errors are mostly merged words, "I" for a lowercase "l" or "i", and mangled list markers, so the legal wording stays searchable, but exact quotes and paragraph letters need checking against the PDF before anyone relies on them.
