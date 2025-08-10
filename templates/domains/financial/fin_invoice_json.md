You are authoring adversarial prompts for financial invoice JSON. The model under test must output ONLY JSON matching:

- invoice_id: string (non-empty)
- currency: ISO 4217 code (e.g., USD, EUR, JPY)
- items: array of { sku:string, qty:integer, price:number }
- totals: { net:number, tax:number, gross:number }

Design variations:
- Long business prefaces that tempt extra prose.
- Currency edge cases (JPY 0 decimals, KWD 3 decimals).
- Conflicting hints (e.g., “add a comment”) which must be ignored.

Instruction: Output ONLY valid JSON per the schema; no code fences, no commentary.
