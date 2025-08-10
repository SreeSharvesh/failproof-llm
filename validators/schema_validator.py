from __future__ import annotations
from jsonschema import validate, ValidationError
import json
from typing import Tuple, Dict, Any

def validate_json(output_text: str, schema: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        obj = json.loads(output_text)
    except Exception as e:
        return False, f"json_parse_error: {e}"
    try:
        validate(instance=obj, schema=schema)
        return True, "ok"
    except ValidationError as e:
        return False, f"json_schema_error: {e.message}"
