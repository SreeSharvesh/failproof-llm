from __future__ import annotations
import time, os, re
from typing import Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class LLMResult(BaseModel):
    text: str
    usage: Dict[str, Any]
    latency_ms: int
    error: str | None = None

class GeminiAdapter:
    def __init__(self, model: str, **params):
        self.model_name = model
        self.params = params
        self.model = genai.GenerativeModel(model)


    def generate(self, prompt: str, **overrides) -> LLMResult:
        st = time.time()
        try:
            resp = self.model.generate_content(prompt, generation_config={**self.params, **overrides})
            text = (resp.text or "") if hasattr(resp, "text") else ""
            
            usage = {}
            return LLMResult(text=text, usage=usage, latency_ms=int((time.time()-st)*1000))
        except Exception as e:
            msg = str(e)
            # crude parse for retry delay seconds
            delay = 0
            if "retry_delay" in msg:
                m = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", msg)
                if m:
                    delay = int(m.group(1))
            # Optional: sleep small backoff (don’t block event loop; do this in runner ideally)
            return LLMResult(text="", usage={}, latency_ms=int((time.time()-st)*1000), err=msg)
