"""Ленивая локальная LLM-инференция: в памяти одновременно только одна модель."""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import api_config as api
import app_config as settings

from polybot.models.model_registry import get_model

SYSTEM_PROMPT = """Ты консервативный классификатор рынка BTC Up/Down на 5 минут.
Оцени вероятность того, что ФИНАЛЬНАЯ reference_price в момент окончания события будет
выше или равна target_price (Price to Beat). Up означает финал >= target_price, Down — финал < target_price.
Цена контракта best_bid/best_ask НЕ является целевой ценой BTC и используется только для оценки сделки.
Обязательно учитывай distance_to_target, remaining_seconds и realized_volatility_60s_pct.
Оцени только переданные числовые данные. Не придумывай новости или факты.
Верни ровно JSON: {"direction":"Up|Down|Hold","confidence":0..1,"reason":"кратко"}.
Confidence — калиброванная вероятность выигрыша выбранного исхода относительно target_price.
При конфликте источников, недостатке target/reference данных или неясности выбери Hold."""

_loaded_key: str | None = None
_tokenizer: Any = None
_model: Any = None
_cache: dict[tuple[str, str, int], dict[str, Any]] = {}


def _dtype() -> Any:
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(settings.LLM_TORCH_DTYPE, torch.float16)


def release_model() -> None:
    global _loaded_key, _tokenizer, _model
    _loaded_key = None
    _tokenizer = None
    _model = None
    _cache.clear()
    gc.collect()


def _load(key: str) -> tuple[Any, Any]:
    global _loaded_key, _tokenizer, _model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _loaded_key == key and _tokenizer is not None and _model is not None:
        return _tokenizer, _model
    release_model()
    spec = get_model(key)
    if spec.kind != "llm" or spec.path is None or not (spec.path / "config.json").exists():
        raise RuntimeError(f"Локальные веса модели {key} не установлены")
    token = getattr(api, "HUGGINGFACE_API_TOKEN", "") or None
    source = str(settings.QWEN_LOCAL_PATH if key == "qwen_lora" else spec.path)
    _tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, token=token)
    _model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        token=token,
        torch_dtype=_dtype(),
        low_cpu_mem_usage=True,
    ).to(settings.QWEN_DEVICE)
    if key == "qwen_lora":
        from peft import PeftModel
        _model = PeftModel.from_pretrained(_model, str(settings.QWEN_LORA_ADAPTER_PATH), local_files_only=True)
    _model.eval()
    _loaded_key = key
    return _tokenizer, _model


def _parse(text: str) -> dict[str, Any]:
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for candidate in reversed(matches):
        try:
            value = json.loads(candidate)
            direction = str(value.get("direction", "Hold")).title()
            if direction not in {"Up", "Down", "Hold"}:
                direction = "Hold"
            confidence = max(0.0, min(1.0, float(value.get("confidence", 0.0))))
            return {"direction": direction, "confidence": confidence, "reason": str(value.get("reason", ""))[:240]}
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    reason = "LLM не вернула JSON" if not matches else "Некорректный JSON от LLM"
    return {"direction": "Hold", "confidence": 0.0, "reason": reason}


def _llama_executable() -> str:
    discovered = shutil.which("llama-cli")
    if discovered:
        return discovered
    packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    matches = list(packages.glob("ggml.llamacpp_*/*llama-cli.exe"))
    if not matches:
        matches = list(packages.rglob("llama-cli.exe")) if packages.exists() else []
    if not matches:
        raise RuntimeError("llama-cli не найден; установите llama.cpp через Winget")
    return str(matches[0])


def _infer_qwen_gguf(context: dict[str, Any]) -> tuple[dict[str, Any], str]:
    prompt = (
        "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
        "<|im_start|>user\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    command = [
        _llama_executable(), "-m", str(settings.QWEN_GGUF_PATH), "-p", prompt,
        "-n", str(settings.QWEN_MAX_NEW_TOKENS), "-c", "2048", "-t", "4",
        "--temp", "0", "--no-display-prompt", "--single-turn", "--simple-io", "--no-warmup",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"llama-cli завершился с кодом {completed.returncode}: {completed.stderr[-500:]}")
    return _parse(completed.stdout), completed.stdout


def infer(key: str, context: dict[str, Any], use_cache: bool = True) -> dict[str, Any]:
    event = str(context.get("event_slug", "offline"))
    bucket = int(time.time() // max(1, settings.LLM_DECISION_CACHE_SECONDS))
    cache_key = (key, event, bucket)
    if use_cache and cache_key in _cache:
        return _cache[cache_key]
    if key == "qwen" and settings.QWEN_GGUF_PATH.exists():
        try:
            result, completion = _infer_qwen_gguf(context)
            result["raw"] = completion[-1000:]
            _cache[cache_key] = result
            return result
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            # Повреждённый/несовместимый GGUF не должен останавливать турнир:
            # используем уже установленные transformers-веса этой же модели.
            gguf_error = f"{type(error).__name__}: {error}"
        else:
            gguf_error = ""
    tokenizer, model = _load(key)
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"))},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(settings.QWEN_DEVICE)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=settings.QWEN_MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    result = _parse(completion)
    result["raw"] = completion[:500]
    if key == "qwen" and 'gguf_error' in locals():
        result["runtime_fallback"] = "transformers_after_gguf_error"
        result["gguf_error"] = gguf_error[-300:]
    _cache[cache_key] = result
    return result
