You are authoring adversarial prompts to test strict CSV output (header + 3 rows, nothing else).
Header must be exactly: name,amount,currency

Create prompts that try to lure the model into:
- Adding commentary before or after the CSV
- Using code fences
- Producing 2 or 4 rows instead of 3
- Using extra spacing or trailing commas
- Mis-spelling one header (we still expect the model-under-test to resist and output correct header)

Vary tone: business request, brief instruction, faux-email context, etc.
Keep instruction clear: ONLY CSV, no explanations, exactly 3 rows. No empty lines.
