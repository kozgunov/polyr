"""Обучает challenger-модели из подготовленного bundle, не заменяя рабочие модели.

Запуск последнего пакета: ``python дообучить_модели.py``.
Конкретный пакет: ``python дообучить_модели.py --bundle ПУТЬ``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app_config as settings
from polybot.models.train_catboost_model import train as train_catboost
from polybot.models.train_direction_model import train as train_custom
from polybot.models.train_exit_sequence import train as train_exit
from polybot.models.train_pnl_model import train as train_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    bundle = args.bundle or Path((settings.PREPARED_TRAINING_DIR / "latest.txt").read_text(encoding="utf-8").strip())
    if not (bundle / "preparation_report.json").exists():
        raise RuntimeError(f"Некорректный training bundle: {bundle}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = bundle / "candidates" / f"cpu_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    snapshot = bundle / "training_snapshot.sqlite3"
    reports = {
        "custom_entry": train_custom(
            snapshot, artifact_path=output / "custom_entry.joblib",
            history_windows=settings.EVENT_HISTORY_LIVE_WINDOWS,
        ),
        "catboost_entry": train_catboost(
            snapshot, output=output / "catboost_entry",
            history_windows=settings.EVENT_HISTORY_LIVE_WINDOWS,
        ),
        "entry_value": train_value(
            snapshot, artifact_path=output / "entry_value.joblib",
            report_path=output / "entry_value_report.json",
        ),
        "exit_value": train_exit(
            bundle / "exit_sequences.parquet", output / "exit_value",
        ),
    }
    result = {
        "status": "candidate_only", "created_at": datetime.now(UTC).isoformat(),
        "bundle": str(bundle), "output": str(output), "reports": reports,
        "production_replaced": False,
    }
    (output / "training_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("all_doneдообучить_модели.py")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("error_in_дообучить_модели.py")
        raise
