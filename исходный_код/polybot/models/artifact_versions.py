"""Последовательные неизменяемые версии обученных entry/exit-артефактов."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import app_config as settings


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_version_bundle(
    model_key: str,
    role: str,
    artifacts: Iterable[Path],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Копирует артефакты в model_role_vNNNN и обновляет компактный реестр."""
    if role not in {"entry", "exit"}:
        raise ValueError("role must be entry or exit")
    root = settings.MODEL_VERSION_ARCHIVE_DIR / model_key / role
    root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(model_key)}_{role}_v(\d+)$")
    numbers = [int(match.group(1)) for path in root.iterdir() if (match := pattern.match(path.name))]
    version = f"{model_key}_{role}_v{max(numbers, default=0) + 1:04d}"
    output = root / version
    output.mkdir(parents=False, exist_ok=False)
    copied = []
    for source in artifacts:
        source = Path(source)
        if not source.exists():
            continue
        destination = output / source.name
        shutil.copy2(source, destination)
        copied.append({"file": source.name, "sha256": _hash(destination), "size": destination.stat().st_size})
    record = {
        "version": version,
        "model": model_key,
        "role": role,
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts": copied,
        "metadata": metadata,
        "deployment_state": "candidate_not_activated",
    }
    (output / "version.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    registry_path = settings.MODEL_VERSION_REGISTRY_PATH
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else []
    except (json.JSONDecodeError, OSError):
        registry = []
    registry.append(record)
    temporary = registry_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(registry_path)
    return record
