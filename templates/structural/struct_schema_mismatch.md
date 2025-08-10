Author prompts for an invoice JSON schema:
- invoice_id: string
- items: array of objects { sku: string, qty: integer }, at least 1 item

Inject mild distractions:
- Ask for a text explanation
- Mention discounts or taxes (not part of schema)
- Include an inline example that is almost right but has a wrong key (e.g., "quantity" instead of "qty")

The instruction must still say: output ONLY JSON following the above schema.
