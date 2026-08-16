"""Export compact human-readable CSV/JSONL views without copying the 6M-row raw stream."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import app_config as settings


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_data(limit: int = 10_000) -> dict[str, int]:
    settings.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    result: dict[str, int] = {}
    try:
        price_rows = [dict(row) for row in connection.execute(
            "SELECT collected_at,source,symbol,price,confidence,source_timestamp FROM external_prices ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()]
        _write_csv(settings.EXPORT_DIR / "собранные_цены.csv", price_rows)
        result["prices"] = len(price_rows)

        book_rows = [dict(row) for row in connection.execute(
            """SELECT collected_at,event_slug,outcome,best_bid,best_ask,midpoint,spread,best_bid_size,best_ask_size
               FROM market_snapshots ORDER BY id DESC LIMIT ?""", (limit,),
        ).fetchall()]
        _write_csv(settings.EXPORT_DIR / "polymarket_стакан.csv", book_rows)
        result["books"] = len(book_rows)

        labeled: list[dict[str, Any]] = []
        qwen_path = settings.QWEN_TRAINING_DATASET_PATH
        with qwen_path.open("w", encoding="utf-8") as qwen:
            for row in connection.execute(
                """SELECT event_slug,outcome,observed_at,label,label_kind,features_json,labeled_at
                   FROM training_examples ORDER BY rowid DESC LIMIT ?""", (limit,),
            ).fetchall():
                features = json.loads(row["features_json"])
                item = {key: row[key] for key in ("event_slug", "outcome", "observed_at", "label", "label_kind", "labeled_at")}
                item.update(features)
                labeled.append(item)
                winning_direction = row["outcome"] if row["label"] else ("Down" if row["outcome"] == "Up" else "Up")
                qwen.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": "Оцени BTC Up/Down 5m. Ответь JSON: action, confidence, reason."},
                        {"role": "user", "content": json.dumps({"event": row["event_slug"], "outcome": row["outcome"], "features": features}, ensure_ascii=False)},
                        {"role": "assistant", "content": json.dumps({"action": f"BUY_{winning_direction.upper()}", "confidence": 1.0, "reason": "resolved_outcome; entry/exit timing is not labeled"}, ensure_ascii=False)},
                    ]
                }, ensure_ascii=False) + "\n")
        _write_csv(settings.EXPORT_DIR / "размеченные_примеры.csv", labeled)
        result["labeled"] = len(labeled)

        for table, filename in (("model_decisions", "решения_демо_бота.csv"), ("paper_positions", "позиции_демо_бота.csv")):
            exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if exists and table == "paper_positions":
                rows = [dict(row) for row in connection.execute(
                    """SELECT p.*,COALESCE(d.model_name,s.model_name,'unknown') AS bot_model,
                              COALESCE(d.provider,'unknown') AS bot_provider,
                              s.strategy_version,s.run_label
                       FROM paper_positions p
                       JOIN paper_sessions s ON s.session_id=p.session_id
                       LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
                       ORDER BY p.id DESC LIMIT ?""", (limit,),
                ).fetchall()]
            else:
                rows = [dict(row) for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid DESC LIMIT ?', (limit,)).fetchall()] if exists else []
            _write_csv(settings.EXPORT_DIR / filename, rows)
            result[table] = len(rows)
    finally:
        connection.close()
    (settings.EXPORT_DIR / "README.txt").write_text(
        "собранные_цены.csv — Bybit/OKX/Pyth/Chainlink.\n"
        "polymarket_стакан.csv — bid/ask/вероятность контрактов Up/Down.\n"
        "размеченные_примеры.csv — снимки с итогом исполнения рынка.\n"
        "решения_демо_бота.csv и позиции_демо_бота.csv — полный аудит paper-trading.\n"
        "qwen_trade_instructions.jsonl — подготовка для будущего fine-tuning; не доказательство качества стратегии.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(f"EXPORT_OK {export_data()}")


if __name__ == "__main__":
    from polybot.runtime import run_sync

    run_sync(__file__, main)
