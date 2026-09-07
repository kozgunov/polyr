"""Единый реестр торговых моделей и их локальной готовности."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import app_config as settings


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    kind: str
    description: str
    path: Path | None = None
    model_id: str | None = None

    @property
    def installed(self) -> bool:
        if self.kind in {"consensus"}:
            return True
        if self.path is None:
            return False
        if self.kind == "llm":
            if self.key == "qwen" and settings.QWEN_GGUF_PATH.exists():
                return True
            if self.key == "qwen_lora":
                return (self.path / "adapter_config.json").exists()
            return (self.path / "config.json").exists()
        return self.path.exists()

    def public(self) -> dict[str, object]:
        value = asdict(self)
        value["path"] = str(self.path) if self.path else None
        value["installed"] = self.installed
        return value


MODEL_SPECS: dict[str, ModelSpec] = {
    "catboost": ModelSpec(
        "catboost", "CatBoost", "numeric",
        "Компактная табличная модель для стакана, времени и внешних цен.",
        settings.CATBOOST_ARTIFACT_PATH,
    ),
    "custom": ModelSpec(
        "custom", "Своя дообученная", "numeric",
        "HistGradientBoosting с event-level разбиением и Platt-калибровкой.",
        settings.TRAINING_ARTIFACT_PATH,
    ),
    "custom_bidir": ModelSpec(
        "custom_bidir", "Двухсторонняя Up/Down v1", "numeric",
        "Shadow-challenger с симметричными признаками относительно выбранной стороны.",
        settings.BIDIRECTIONAL_ENTRY_CANDIDATE_PATH,
    ),
    "gemma": ModelSpec(
        "gemma", "Gemma 3 1B", "llm",
        "Локальная instruction-LLM; загружается только при выборе.",
        settings.GEMMA_LOCAL_PATH, settings.GEMMA_MODEL_ID,
    ),
    "qwen": ModelSpec(
        "qwen", "Qwen2.5 1.5B Instruct", "llm",
        "Локальная instruction-LLM для структурированного рыночного решения.",
        settings.QWEN_LOCAL_PATH, settings.QWEN_MODEL_ID,
    ),
    "qwen_lora": ModelSpec(
        "qwen_lora", "Qwen2.5 · LoRA challenger", "llm",
        "Сбалансированный adapter для Price to Beat; базовые веса не изменяются.",
        settings.QWEN_LORA_ADAPTER_PATH, settings.QWEN_MODEL_ID,
    ),
    "consensus_qwen_custom": ModelSpec(
        "consensus_qwen_custom", "Consensus · Qwen + своя", "consensus",
        "Вход только при совпадении сильных сигналов Qwen и числовой модели.",
    ),
    "consensus_gemma_custom": ModelSpec(
        "consensus_gemma_custom", "Consensus · Gemma + своя", "consensus",
        "Вход только при совпадении сильных сигналов Gemma и числовой модели.",
    ),
}


def get_model(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"Неизвестная модель: {key}") from exc


def model_is_ready(key: str) -> bool:
    spec = get_model(key)
    if key == "consensus_qwen_custom":
        return MODEL_SPECS["custom"].installed and MODEL_SPECS["qwen"].installed
    if key == "consensus_gemma_custom":
        return MODEL_SPECS["custom"].installed and MODEL_SPECS["gemma"].installed
    return spec.installed


def public_models() -> list[dict[str, object]]:
    result = []
    for key, spec in MODEL_SPECS.items():
        item = spec.public()
        item["installed"] = model_is_ready(key)
        result.append(item)
    return result
