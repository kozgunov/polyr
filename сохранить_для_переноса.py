"""Создаёт воспроизводимый manifest проекта для переноса через OneDrive."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "ПЕРЕНОС_НА_НОВЫЙ_ПК"
MANIFEST = OUTPUT / "полный_manifest_sha256.json"
EXCLUDED_DIRS = {".venv", ".venv-gpu", "__pycache__", ".pytest_cache", ".ruff_cache", "ПЕРЕНОС_НА_НОВЫЙ_ПК"}
TRANSIENT_SUFFIXES = (".pyc", ".pyo", ".tmp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_files():
    for directory, names, files in os.walk(ROOT):
        names[:] = [name for name in names if name not in EXCLUDED_DIRS]
        base = Path(directory)
        for name in files:
            path = base / name
            if path.name in {"market_data.sqlite3-wal", "market_data.sqlite3-shm"}:
                continue
            if path.suffix.lower() in TRANSIENT_SUFFIXES:
                continue
            yield path


def database_snapshot() -> dict:
    path = ROOT / "данные" / "market_data.sqlite3"
    connection = sqlite3.connect(path, timeout=60)
    checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
    tables = int(connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0])
    examples = int(connection.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0])
    events = int(connection.execute("SELECT COUNT(DISTINCT event_slug) FROM training_examples").fetchone()[0])
    controls = {str(row[0]): str(row[1]) for row in connection.execute("SELECT control_key,control_value FROM runtime_controls")}
    session = connection.execute(
        "SELECT session_id,started_at,status,strategy_version,model_name,run_label,initial_balance_usdc,cash_balance_usdc,realized_pnl_usdc FROM paper_sessions ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    open_paper = int(connection.execute("SELECT COUNT(*) FROM paper_positions WHERE status='open'").fetchone()[0])
    open_live = int(connection.execute("SELECT COUNT(*) FROM live_positions WHERE status='open'").fetchone()[0])
    connection.close()
    wal = Path(f"{path}-wal")
    return {
        "relative_path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
        "sha256": sha256(path), "quick_check": quick_check, "wal_checkpoint": list(checkpoint),
        "wal_bytes": wal.stat().st_size if wal.exists() else 0, "tables": tables,
        "training_examples": examples, "resolved_events": events,
        "open_paper_positions": open_paper, "open_live_positions": open_live,
        "runtime_controls": controls,
        "latest_session": dict(zip(
            ("session_id", "started_at", "status", "strategy_version", "model_name", "run_label", "initial_balance_usdc", "cash_balance_usdc", "realized_pnl_usdc"),
            session,
        )) if session else None,
    }


def code_archive() -> dict:
    path = OUTPUT / "код_конфигурация_тесты.zip"
    included = []
    excluded_roots = {"данные", "модели", "журналы", "ПЕРЕНОС_НА_НОВЫЙ_ПК", ".venv", ".venv-gpu"}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file in project_files():
            relative = file.relative_to(ROOT)
            if relative.parts[0] in excluded_roots or relative.name == "api_config.py":
                continue
            archive.write(file, relative.as_posix())
            included.append(relative.as_posix())
    return {"relative_path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path), "files": len(included)}


def write_restore_files() -> None:
    readme = """# Перенос Polybot на новый Windows-компьютер

Папка `poly_project` целиком синхронизируется через OneDrive. Не копируйте `.venv`: она намеренно исключена и создаётся заново.

1. Дождитесь, пока OneDrive полностью скачает папки `данные` и `модели`.
2. Установите Python 3.12 x64 и свежий драйвер NVIDIA.
3. Откройте PowerShell в корне `poly_project`.
4. Выполните: `powershell -ExecutionPolicy Bypass -File .\\ВОССТАНОВИТЬ_НА_НОВОМ_ПК.ps1`.
5. Для полной проверки всех 20+ ГБ: `.\\.venv\\Scripts\\python.exe .\\проверить_целостность_переноса.py --full`.
6. Первый безопасный запуск: `.\\.venv\\Scripts\\python.exe .\\запустить_проект.py --mode paper`.

`api_config.py` содержит реальные секреты. Он включён в OneDrive-папку и checksum-manifest, но намеренно не дублируется в ZIP кода.
LIVE после переноса включайте только вручную после PAPER-проверки API, стакана и исполнителя заявок.
"""
    (ROOT / "README_ПЕРЕНОС.md").write_text(readme, encoding="utf-8")
    restore = r'''$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python.exe -m pip install -r .\настройка_проекта\requirements.txt
& .\.venv\Scripts\python.exe .\проверить_целостность_переноса.py
Write-Host "Среда восстановлена. Сначала запустите PAPER:"
Write-Host ".\.venv\Scripts\python.exe .\запустить_проект.py --mode paper"
'''
    (ROOT / "ВОССТАНОВИТЬ_НА_НОВОМ_ПК.ps1").write_text(restore, encoding="utf-8-sig")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    database = database_snapshot()
    if database["quick_check"] != "ok" or database["wal_bytes"] != 0:
        raise RuntimeError(f"База не готова к переносу: {database}")
    if database["open_paper_positions"] or database["open_live_positions"]:
        raise RuntimeError("Нельзя фиксировать перенос при открытой позиции")
    write_restore_files()
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    (OUTPUT / "pip_freeze_исходного_устройства.txt").write_text(freeze, encoding="utf-8")
    archive = code_archive()
    entries = []
    total = 0
    for index, path in enumerate(project_files(), 1):
        stat = path.stat()
        total += stat.st_size
        entries.append({
            "path": path.relative_to(ROOT).as_posix(), "bytes": stat.st_size,
            "sha256": sha256(path), "modified_ns": stat.st_mtime_ns,
            "sensitive": path.name == "api_config.py",
        })
        if index % 50 == 0:
            print(f"HASH_PROGRESS files={index} gb={total / 1024**3:.2f}", flush=True)
    payload = {
        "schema_version": 2, "created_at": datetime.now(UTC).isoformat(),
        "source": "OneDrive project folder; .venv and caches intentionally excluded",
        "project_directory_name": ROOT.name, "files": entries, "file_count": len(entries),
        "total_bytes": total, "database": database, "code_archive": archive,
        "restore": "ВОССТАНОВИТЬ_НА_НОВОМ_ПК.ps1",
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "ГОТОВО_К_ПЕРЕНОСУ.txt").write_text(
        f"created_at={payload['created_at']}\nfiles={len(entries)}\nbytes={total}\ndatabase_sha256={database['sha256']}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "ready", "manifest": str(MANIFEST), "files": len(entries),
        "total_gb": round(total / 1024**3, 3), "database": database,
        "code_archive": archive,
    }, ensure_ascii=False, indent=2))
    print(f"all_done{Path(__file__).name}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(f"error_in_{Path(__file__).name}")
        raise
