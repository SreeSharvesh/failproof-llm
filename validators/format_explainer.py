from __future__ import annotations
import os, asyncio
from typing import Dict, Any
from dotenv import load_dotenv
import openai

load_dotenv()
# Uses OpenAI for simplicity; you can add Gemini later if desired.
openai.api_key = os.getenv("OPENAI_API_KEY")

_PROMPT_TMPL = """You are a precise test failure analyst.
Given:
- Family: {family}
- Expected spec (JSON): {expect_json}
- Validator reason: {reason}
- Model output (truncated to 200 chars): {output}

Explain in 1–2 sentences why it failed and the minimal fix.
Respond ONLY in JSON as:
{{"reason":"...", "fix":"..."}}"""

async def get_format_explanation(critic_cfg: Dict[str, Any], case: Dict[str, Any], output: str, reason: str) -> Dict[str, str] | None:
    try:
        prompt = _PROMPT_TMPL.format(
            family=case.get("family"),
            expect_json=str(case.get("expect", {})),
            reason=str(reason),
            output=(output or "")[:200]
        )
        # run in thread to keep runner async
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: openai.chat.completions.create(
                model=critic_cfg.get("model", "gpt-4o-mini"),
                messages=[{"role":"user","content":prompt}],
                temperature=0.0,
                max_tokens=int(critic_cfg.get("max_tokens", 120)),
                response_format={"type":"json_object"}
            )
        )
        text = resp.choices[0].message.content or "{}"
        import json as _json
        return _json.loads(text)
    except Exception:
        return None
