"""Готовит event-level, direction-balanced датасет QLoRA без утечки событий."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app_config as settings


def action_of(row: dict) -> str:
    try:
        return str(json.loads(row["messages"][-1]["content"])["action"]).upper()
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return "INVALID"


def balanced(rows: list[dict], seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        action = action_of(row)
        if action in {"BUY_UP", "BUY_DOWN", "WAIT"}:
            groups[action].append(row)
    rng = random.Random(seed)
    target = min(len(groups["BUY_UP"]), len(groups["BUY_DOWN"]), len(groups["WAIT"]))
    selected = []
    for action in ("BUY_UP", "BUY_DOWN", "WAIT"):
        selected.extend(rng.sample(groups[action], target))
    rng.shuffle(selected)
    return selected


def main() -> None:
    reports = sorted(
        settings.MODEL_CANDIDATE_DIR.glob("cycle_*/training_report.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not reports:
        raise RuntimeError("Нет challenger-датасета для QLoRA")
    source = reports[-1].parent / "qwen" / "supervised_reward_examples.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    events = sorted({str(row["event_slug"]) for row in rows})
    split = max(1, int(len(events) * 0.8))
    train_events, validation_events = set(events[:split]), set(events[split:])
    train = balanced([row for row in rows if str(row["event_slug"]) in train_events], 42)
    validation = balanced([row for row in rows if str(row["event_slug"]) in validation_events], 43)
    output = settings.MODEL_DIR / "qwen_qlora_experiment_v1"
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("train.jsonl", train), ("validation.jsonl", validation)):
        (output / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8",
        )
    config = {
        "created_at": datetime.now(UTC).isoformat(), "base_model": str(settings.QWEN_LOCAL_PATH),
        "source": str(source), "event_split": "chronological 80/20; no event overlap",
        "train_events": len(train_events), "validation_events": len(validation_events),
        "train_rows": len(train), "validation_rows": len(validation),
        "train_actions": dict(Counter(action_of(row) for row in train)),
        "validation_actions": dict(Counter(action_of(row) for row in validation)),
        "method": "QLoRA NF4 when CUDA is available; LoRA adapter never overwrites base Qwen",
        "hyperparameters": {"rank": 8, "alpha": 16, "dropout": 0.05, "epochs": 1, "learning_rate": 0.0002, "max_length": 512},
        "acceptance": {"direction_share_max": 0.65, "validation_roc_auc_min": 0.60, "brier_max": 0.24, "no_direction_collapse": True},
    }
    (output / "experiment.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print("all_done_prepare_qlora_experiment.py")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("error_in_prepare_qlora_experiment.py")
        raise
