# adapters/openai_adapter.py
from typing import Any, Dict
import openai

class OpenAIAdapter:
    def __init__(self, model: str, **default_params):
        self.model = model
        # Never keep force_json as a default; runner will pass it per call
        default_params.pop("force_json", None)
        self.defaults = default_params or {}

    def generate(self, prompt: str, **overrides) -> Any:
        params = {**self.defaults, **overrides}
        force_json = bool(params.pop("force_json", False))

        messages = []
        if force_json:
            # Satisfy OpenAI requirement: messages must literally contain the word 'json'
            messages.append({
                "role": "system",
                "content": "You must respond with a valid JSON object. Output json only."
            })
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": params.get("temperature", 0.0),
            "max_tokens": params.get("max_tokens", 256),
        }
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = openai.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            result = type("Result", (), {})()
            result.text = text
            result.usage = getattr(resp, "usage", {}) or {}
            result.error = None
            result.latency_ms = 0
            return result
        except Exception as e:
            msg = str(e)

            # If JSON-mode tripped the policy anyway, retry once without response_format
            if force_json and "must contain the word 'json'" in msg.lower():
                try:
                    kwargs.pop("response_format", None)
                    # Keep the system message that mentions json
                    resp = openai.chat.completions.create(**kwargs)
                    text = resp.choices[0].message.content or ""
                    result = type("Result", (), {})()
                    result.text = text
                    result.usage = getattr(resp, "usage", {}) or {}
                    result.error = None
                    result.latency_ms = 0
                    return result
                except Exception as e2:
                    msg = f"{msg} | fallback_error: {e2}"

            # Return error
            result = type("Result", (), {})()
            result.text = ""
            result.usage = {}
            result.error = msg
            result.latency_ms = 0
            return result
