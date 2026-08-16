"""Обучение числовых моделей и последовательная загрузка локальных LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from polybot.models.train_catboost_model import train as train_catboost
from polybot.models.train_direction_model import train as train_custom

import api_config as api
import app_config as settings


def _download(model_id: str, path: Path) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    if (path / "config.json").exists() and any(path.glob("*.safetensors")):
        return {"model_id": model_id, "path": str(path), "installed": True, "reused_local_files": True}
    snapshot_download(
        repo_id=model_id,
        local_dir=path,
        token=getattr(api, "HUGGINGFACE_API_TOKEN", "") or None,
        allow_patterns=[
            "*.json", "*.safetensors", "*.model", "*.txt",
            "tokenizer*", "vocab*", "merges*", "added_tokens*",
        ],
    )
    return {"model_id": model_id, "path": str(path), "installed": (path / "config.json").exists()}


def install_all() -> dict[str, Any]:
    if not settings.DATABASE_PATH.exists():
        raise RuntimeError(f"Не найдена база данных: {settings.DATABASE_PATH}")
    result: dict[str, Any] = {"catboost": {}, "custom": {}, "gemma": {}, "qwen": {}}
    result["custom"] = train_custom(settings.DATABASE_PATH)
    result["catboost"] = train_catboost(settings.DATABASE_PATH)
    result["qwen"] = _download(settings.QWEN_MODEL_ID, settings.QWEN_LOCAL_PATH)
    settings.QWEN_GGUF_PATH.parent.mkdir(parents=True, exist_ok=True)
    downloaded_gguf = str(settings.QWEN_GGUF_PATH)
    if not settings.QWEN_GGUF_PATH.exists():
        downloaded_gguf = hf_hub_download(
            repo_id=settings.QWEN_GGUF_REPO_ID,
            filename=settings.QWEN_GGUF_FILENAME,
            local_dir=settings.QWEN_GGUF_PATH.parent,
            token=getattr(api, "HUGGINGFACE_API_TOKEN", "") or None,
        )
    result["qwen"]["gguf_path"] = downloaded_gguf
    result["qwen"]["gguf_installed"] = settings.QWEN_GGUF_PATH.exists()
    try:
        result["gemma"] = _download(settings.GEMMA_MODEL_ID, settings.GEMMA_LOCAL_PATH)
    except Exception as exc:  # noqa: BLE001 - сохраняем понятный отчёт для любой ошибки доступа HF
        result["gemma"] = {
            "model_id": settings.GEMMA_MODEL_ID,
            "installed": False,
            "error": str(exc),
            "hint": "Примите условия Gemma на Hugging Face и повторите запуск.",
        }
    return result


def main() -> None:
    result = install_all()
    output = settings.MODEL_DIR / "installation_report.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from polybot.runtime import run_sync

    run_sync(__file__, main)
