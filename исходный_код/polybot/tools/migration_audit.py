"""Проверяет переносимость данных и моделей перед переходом на новый компьютер."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app_config as settings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(checkpoint: bool = False) -> dict[str, Any]:
    required = [
        settings.DATABASE_PATH, settings.GPU_REQUIREMENTS_PATH, settings.GPU_TRAINING_PLAN_PATH,
        settings.GPU_MODEL_CATALOG_PATH,
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    database = {"path": str(settings.DATABASE_PATH), "exists": settings.DATABASE_PATH.exists()}
    if settings.DATABASE_PATH.exists():
        if checkpoint:
            writable = sqlite3.connect(settings.DATABASE_PATH)
            writable.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            writable.close()
        connection = sqlite3.connect(f"file:{settings.DATABASE_PATH}?mode=ro", uri=True)
        wal_path = Path(f"{settings.DATABASE_PATH}-wal")
        database.update({
            "quick_check": str(connection.execute("PRAGMA quick_check").fetchone()[0]),
            "tables": int(connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]),
            "training_examples": int(connection.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]),
            "resolved_events": int(connection.execute("SELECT COUNT(DISTINCT event_slug) FROM training_examples").fetchone()[0]),
            "bytes": settings.DATABASE_PATH.stat().st_size,
            "sha256": _sha256(settings.DATABASE_PATH),
            "wal_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
        })
        connection.close()
    artifacts = []
    for suffix in ("*.joblib", "*.cbm", "*.pt", "*.json"):
        for path in settings.MODEL_DIR.rglob(suffix):
            if path.is_file() and path.stat().st_size < 2 * 1024**3:
                artifacts.append({
                    "path": str(path.relative_to(settings.PROJECT_ROOT)), "bytes": path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                })
    manifest = {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "ready": not missing and database.get("quick_check") == "ok" and database.get("wal_bytes", 0) == 0,
        "project_root": str(settings.PROJECT_ROOT), "missing": missing, "database": database,
        "model_artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "copy": ["app_config.py", "api_config.py (передать отдельно и безопасно)", "данные", "модели", "исходный_код", "запуск", "настройка_проекта", "тесты"],
        "exclude": [".venv", ".venv-gpu", "__pycache__", ".pytest_cache", ".ruff_cache", "журналы/*.log"],
        "new_machine_command": "py -3.12 подготовить_gpu_компьютер.py --execute --build-data",
        "finalize_before_copy": ".\\.venv\\Scripts\\python.exe .\\запустить_проект.py stop; затем .\\.venv\\Scripts\\python.exe .\\подготовить_переход.py --checkpoint",
    }
    settings.MIGRATION_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def verify_existing_manifest() -> dict[str, Any]:
    if not settings.MIGRATION_MANIFEST_PATH.exists():
        return {"status": "missing", "valid": False}
    expected = json.loads(settings.MIGRATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_hash = expected.get("database", {}).get("sha256")
    actual_hash = _sha256(settings.DATABASE_PATH) if settings.DATABASE_PATH.exists() else None
    quick_check = None
    if settings.DATABASE_PATH.exists():
        connection = sqlite3.connect(f"file:{settings.DATABASE_PATH}?mode=ro", uri=True)
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        connection.close()
    return {
        "status": "ok" if expected_hash == actual_hash and quick_check == "ok" else "mismatch",
        "valid": expected_hash == actual_hash and quick_check == "ok",
        "expected_sha256": expected_hash, "actual_sha256": actual_hash, "quick_check": quick_check,
    }
