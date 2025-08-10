import json
from jsonschema import validate, exceptions as js_ex

def validate_json(output_text: str, schema: dict) -> tuple[bool, str]:
    try:
        obj = json.loads(output_text)
    except Exception:
        return False, "parse_error: not json"

    try:
        validate(instance=obj, schema=schema)
        return True, "ok"
    except js_ex.SchemaError:
        # bad test config/schema — do not crash the run
        return False, "schema_invalid"
    except js_ex.ValidationError as e:
        # normal schema mismatch
        reason = getattr(e, "message", "schema_error")
        return False, f"schema_error: {reason}"
