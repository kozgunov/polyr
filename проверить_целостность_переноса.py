"""Проверяет перенесённые файлы по manifest без чтения или вывода секретов."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "ПЕРЕНОС_НА_НОВЫЙ_ПК" / "полный_manifest_sha256.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="проверить SHA-256 всех данных и весов")
    args = parser.parse_args()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    critical_prefixes = ("app_config.py", "api_config.py", "данные/market_data.sqlite3", "модели/", "исходный_код/", "настройка_проекта/")
    selected = payload["files"] if args.full else [row for row in payload["files"] if row["path"].startswith(critical_prefixes)]
    missing = []
    mismatched = []
    for index, row in enumerate(selected, 1):
        path = ROOT / Path(row["path"])
        if not path.exists():
            missing.append(row["path"])
        elif path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            mismatched.append(row["path"])
        if index % 50 == 0:
            print(f"VERIFY_PROGRESS {index}/{len(selected)}", flush=True)
    database = ROOT / "данные" / "market_data.sqlite3"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
    connection.close()
    report = {"valid": not missing and not mismatched and quick_check == "ok", "checked": len(selected), "missing": missing, "mismatched": mismatched, "database_quick_check": quick_check}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)
    print(f"all_done{Path(__file__).name}")


if __name__ == "__main__":
    main()
