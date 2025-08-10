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
    Minimal HF adapter compatible with your runner.
    - Supports decoder-only chat models (TinyLlama, Qwen) and seq2seq (FLAN-T5).
    - No device_map='auto' (avoids meta tensors); we pick one device and move model there.
    - Respects 'force_json' only as a soft suffix (no server-side JSON mode on HF).
    """

    def __init__(self, model_name: str, **defaults):
        self.model_name = model_name
        self.defaults = {k: v for k, v in (defaults or {}).items() if k != "force_json"}

        # Choose device and dtype safely
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # prefer bf16 if supported; else fp16
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            torch_dtype = torch.float32

        # Detect seq2seq (T5) vs decoder-only
        self.is_seq2seq = "t5" in model_name.lower()

        # Load tokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Ensure pad_token_id is set (some small chat models lack it)
        if self.tok.pad_token_id is None:
            # fall back to eos token
            if self.tok.eos_token_id is not None:
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                # as a last resort
                self.tok.add_special_tokens({"pad_token": "<|pad|>"})

        # Load model on a single device (no device_map to avoid meta tensors)
        if self.is_seq2seq:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=False,   # keep simple to avoid meta device surprises
                trust_remote_code=False,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=False,
                trust_remote_code=False,
            )

        # Resize embeddings if we added a pad token
        if len(self.tok) > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tok))

        # Move model to the chosen device and eval mode
        self.model.to(self.device)
        self.model.eval()

        # Cache EOS/PAD tokens
        self.eos_token_id = self.tok.eos_token_id
        self.pad_token_id = self.tok.pad_token_id

    def _build_prompt(self, prompt: str, force_json: bool) -> str:
        # Soft hint only for JSON; small HF models can get confused by long meta text
        if force_json:
            return prompt.rstrip() + "\n\nRespond with JSON only."
        return prompt

    @torch.inference_mode()
    def generate(self, prompt: str, **overrides) -> Any:
        params = {**self.defaults, **overrides}
        temperature = float(params.get("temperature", 0.0))
        max_tokens = int(params.get("max_tokens", 256))
        force_json = bool(params.get("force_json", False))

        text = self._build_prompt(prompt, force_json=force_json)

        # Tokenize on same device
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
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        try:
            out_ids = self.model.generate(**inputs, **gen_kwargs)

            if self.is_seq2seq:
                output = self.tok.decode(out_ids[0], skip_special_tokens=True)
            else:
                # For decoder-only, drop the prompt tokens when decoding
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
