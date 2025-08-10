from __future__ import annotations
from typing import Any, Dict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

class HFAdapter:
    """
    Safe CPU adapter (no accelerate, no meta).
    - Forces CPU + float32 so it runs everywhere.
    - Works with TinyLlama/Qwen (decoder-only) and Flan-T5 (seq2seq).
    - 'force_json' is a soft hint; validators/repairs enforce structure.
    """

    def __init__(self, model_name: str, **defaults):
        self.model_name = model_name
        self.defaults = {k: v for k, v in (defaults or {}).items() if k != "force_json"}

        # Always CPU for stability first. (You can add a CUDA branch later if needed.)
        self.device = torch.device("cpu")
        torch_dtype = torch.float32

        # Detect seq2seq (T5) vs decoder-only
        self.is_seq2seq = "t5" in model_name.lower()

        # Load tokenizer (ensure pad token)
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tok.pad_token_id is None:
            if self.tok.eos_token_id is not None and self.tok.eos_token is not None:
                self.tok.pad_token = self.tok.eos_token
            else:
                self.tok.add_special_tokens({"pad_token": "<|pad|>"})

        # STRICT: load model without accelerate/lazy init
        from_pretrained_kwargs = dict(
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            device_map=None,  # Force all weights to load on the specified device.
            trust_remote_code=True,
        )
        model.to("cpu") 

        if self.is_seq2seq:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **from_pretrained_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **from_pretrained_kwargs)
        
        model.to('cpu') 
        # If we added a pad token, resize embeddings BEFORE moving to device
        if len(self.tok) > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tok))

        # Move to CPU and verify no meta params remain
        self.model.to("cpu")
        self.model.eval()
        for n, p in self.model.named_parameters():
            if p.is_meta:
                raise RuntimeError(f"Parameter on meta device after load: {n}")

        self.eos_id = self.tok.eos_token_id
        self.pad_id = self.tok.pad_token_id

    def _build_prompt(self, prompt: str, force_json: bool) -> str:
        # Keep hint tiny for small models
        return prompt.rstrip() + ("\n\nRespond with JSON only." if force_json else "")

    @torch.inference_mode()
    def generate(self, prompt: str, **overrides) -> Any:
        params = {**self.defaults, **overrides}
        temperature = float(params.get("temperature", 0.0))
        max_tokens = int(params.get("max_tokens", 256))
        force_json = bool(params.get("force_json", False))

        text = self._build_prompt(prompt, force_json)

        inputs = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048 if not self.is_seq2seq else 4096,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.0),
            temperature=max(temperature, 1e-6) if temperature > 0.0 else None,
            top_p=0.95 if temperature > 0.0 else None,
            eos_token_id=self.eos_id,
            pad_token_id=self.pad_id,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        try:
            out_ids = self.model.generate(**inputs, **gen_kwargs)
            if self.is_seq2seq:
                output = self.tok.decode(out_ids[0], skip_special_tokens=True)
            else:
                input_len = inputs["input_ids"].shape[1]
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
