import json, re

ISO4217 = {"USD":2, "EUR":2, "JPY":0, "KWD":3}  # extend from refs_iso4217.json if you load files

def validate_fin_invoice_json(s: str) -> tuple[bool, str]:
    try:
        obj = json.loads(s)
    except Exception:
        return False, "parse_error: not json"

    for key in ["invoice_id","currency","items","totals"]:
        if key not in obj: return False, f"missing:{key}"
    cur = obj["currency"]
    if cur not in ISO4217: return False, "currency_invalid"
    decimals = ISO4217[cur]

    items = obj["items"]
    if not isinstance(items, list) or not items:
        return False, "items_invalid"
    for it in items:
        if not all(k in it for k in ["sku","qty","price"]):
            return False, "item_field_missing"
        if not isinstance(it["qty"], int):
            return False, "qty_not_int"
        if not isinstance(it["price"], (int,float)):
            return False, "price_not_number"
        # optional: enforce decimal scale per currency by string check
        price_txt = f"{it['price']}"
        if "." in price_txt and len(price_txt.split(".")[1]) > decimals:
            return False, "price_wrong_decimals"

    t = obj["totals"]
    if not all(k in t for k in ["net","tax","gross"]): return False, "totals_missing"
    if abs((t["net"] + t["tax"]) - t["gross"]) > 1e-6:
        return False, "totals_mismatch"

    return True, "ok"
