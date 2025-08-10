Create prompts that require ONLY CSV with header: patient_id,icd10,procedure,date.
- patient_id: synthetic (PAT-#####)
- icd10: valid from provided mini list
- procedure: short label
- date: ISO YYYY-MM-DD

Add distractions: extra fields in the prose, differing date formats in examples, random clinical text.
Final instruction: ONLY CSV, exact header and exactly 5 rows, no extra text.
