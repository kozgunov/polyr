"""Сравнивает более активные пороги только в офлайн walk-forward replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import app_config as settings
from polybot.analytics.walk_forward_backtest import run


CONFIGS = [
    {"name": "baseline", "confidence": 0.72, "edge": 0.04, "max_price": 0.75, "min_elapsed": 60},
    {"name": "wider_price", "confidence": 0.72, "edge": 0.04, "max_price": 0.90, "min_elapsed": 60},
    {"name": "edge_003", "confidence": 0.72, "edge": 0.03, "max_price": 0.90, "min_elapsed": 60},
    {"name": "edge_002", "confidence": 0.72, "edge": 0.02, "max_price": 0.90, "min_elapsed": 60},
    {"name": "active_070", "confidence": 0.70, "edge": 0.02, "max_price": 0.90, "min_elapsed": 45},
]


def main() -> None:
    original = {
        "confidence": settings.MIN_ENTRY_CONFIDENCE, "edge": settings.PAPER_MIN_ENTRY_NET_EDGE,
        "max_price": settings.PAPER_MAX_ENTRY_PRICE, "min_elapsed": settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN,
    }
    reports = []
    try:
        for config in CONFIGS:
            settings.MIN_ENTRY_CONFIDENCE = config["confidence"]
            settings.PAPER_MIN_ENTRY_NET_EDGE = config["edge"]
            settings.PAPER_MAX_ENTRY_PRICE = config["max_price"]
            settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN = config["min_elapsed"]
            result = run(settings.DATABASE_PATH)
            reports.append({"config": config, "metrics": {key: result[key] for key in (
                "out_of_sample_events", "events_with_trades", "entries", "exits", "net_pnl_usdc",
                "fees_usdc", "win_rate_on_traded_events", "profit_factor", "average_event_pnl",
            )}})
    finally:
        settings.MIN_ENTRY_CONFIDENCE = original["confidence"]
        settings.PAPER_MIN_ENTRY_NET_EDGE = original["edge"]
        settings.PAPER_MAX_ENTRY_PRICE = original["max_price"]
        settings.PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN = original["min_elapsed"]
    output = settings.MODEL_DIR / "trading_threshold_sweep.json"
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
        print(f"all_done{Path(__file__).name}")
    except Exception:
        print(f"error_in_{Path(__file__).name}")
        raise
