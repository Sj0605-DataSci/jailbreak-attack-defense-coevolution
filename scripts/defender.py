"""Defender wrapper: base LFM2.5-1.2B-Instruct + an optional LoRA adapter
from a previous round, loaded cumulatively.
"""
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


class Defender:
    def __init__(self, adapter_path: Path | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(config.DEFENDER_BASE_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.DEFENDER_BASE_PATH, dtype=torch.bfloat16, device_map="cuda"
        )
        self.adapter_path = adapter_path
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(self, user_prompt: str) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=config.MAX_NEW_TOKENS_DEFENDER,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
