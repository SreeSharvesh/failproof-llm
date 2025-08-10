# adapters/hf_adapter.py
from __future__ import annotations
from typing import Any, Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

class HFAdapter:
    """
    Minimal HF adapter that matches OpenAIAdapter.generate(prompt, **overrides).
    Supports:
      - decoder-only chat models (TinyLlama, Qwen chat)
      - seq2seq (Flan-T5)
    Notes:
      - We cannot use response_format=json_object here; we rely on your validators + auto-repair.
      - We still accept 'force_json' kwarg for parity, but use it only to add a soft instruction suffix.
    """
    def __init__(self, model_name: str, **defaults):
        self.model_name = model_name
        self.defaults = defaults or {}

        # Load model
        self.is_seq2seq = any(k in model_name.lower() for k in ["t5"])
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        if self.is_seq2seq:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=torch_dtype, device_map=device_map
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype, device_map=device_map
            )
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def _build_prompt(self, prompt: str, force_json: bool) -> str:
        # Add a minimal suffix when JSON is requested (soft hint)
        if force_json:
            # keep it short—HF small models can get confused by long meta-instructions
            return prompt.rstrip() + "\n\nRespond with JSON only."
        return prompt

    def generate(self, prompt: str, **overrides) -> Any:
        params = {**self.defaults, **overrides}
        temperature = float(params.get("temperature", 0.0))
        max_tokens = int(params.get("max_tokens", 256))
        force_json = bool(params.get("force_json", False))

        # Build input text
        text = self._build_prompt(prompt, force_json=force_json)

        # Tokenize
        inputs = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096 if self.is_seq2seq else 2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generation kwargs
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            temperature=max(temperature, 1e-6) if temperature > 0.0 else None,
            top_p=0.95 if temperature > 0.0 else None,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id
        )
        # Remove None keys
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        try:
            if self.is_seq2seq:
                out_ids = self.model.generate(**inputs, **gen_kwargs)
                output = self.tok.decode(out_ids[0], skip_special_tokens=True)
            else:
                # decoder-only: concatenate prompt + completion; decode only the completion part
                input_len = inputs["input_ids"].shape[1]
                out_ids = self.model.generate(**inputs, **gen_kwargs)
                output = self.tok.decode(out_ids[0][input_len:], skip_special_tokens=True)

            result = type("Result", (), {})()
            result.text = output
            result.usage = {}
            result.error = None
            result.latency_ms = 0
            return result
        except Exception as e:
            result = type("Result", (), {})()
            result.text = ""
            result.usage = {}
            result.error = str(e)
            result.latency_ms = 0
            return result
