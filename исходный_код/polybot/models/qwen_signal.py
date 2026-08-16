"""Local configurable Qwen signal reviewer. It never submits an order."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import app_config as settings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """You are a conservative prediction-market signal classifier.
Return exactly one JSON object and nothing else, with these keys:
action: BUY_YES, BUY_NO, or HOLD
confidence: number from 0 to 1
reason: short Russian explanation
risks: array of short Russian strings
The input may contain market, oracle and news data. Never invent missing facts.
If oracle confirmation is absent, stale, conflicting, or data is insufficient, action must be HOLD.
You cannot place orders and you must not recommend bypassing risk limits."""


def extract_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model did not return a JSON object")
    signal = json.loads(text[start : end + 1])
    if signal.get("action") not in {"BUY_YES", "BUY_NO", "HOLD"}:
        raise ValueError("Invalid action from model")
    confidence = float(signal.get("confidence", 0))
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return signal


def generate_signal(context: dict[str, Any]) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        settings.QWEN_MODEL_ID,
        local_files_only=settings.QWEN_LOCAL_FILES_ONLY,
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.QWEN_MODEL_ID,
        torch_dtype=torch.float32,
        local_files_only=settings.QWEN_LOCAL_FILES_ONLY,
    ).to(settings.QWEN_DEVICE)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(settings.QWEN_DEVICE)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=settings.QWEN_MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(output[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
    return extract_json(completion)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python qwen_signal.py market_context.json")
    context_path = Path(sys.argv[1])
    context = json.loads(context_path.read_text(encoding="utf-8"))
    print(json.dumps(generate_signal(context), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync
    run_sync(__file__, main)
