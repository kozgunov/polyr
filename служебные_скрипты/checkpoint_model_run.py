"""Create an immutable, auditable checkpoint of trades opened by one model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist, mean, median, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "исходный_код"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app_config as settings
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from polybot.trading.fees import total_fee_usdc
from polybot.trading.live_guard import wilson_lower_bound


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reward(pnl: float, notional: float) -> tuple[float, str]:
    roi = pnl / max(notional, 0.01)
    score = max(-1.0, min(3.0, roi))
    if roi <= -0.5:
        tier = "catastrophic_loss"
    elif roi < 0:
        tier = "loss"
    elif roi < 0.25:
        tier = "small_profit"
    elif roi < 0.75:
        tier = "strong_profit"
    else:
        tier = "exceptional_profit"
    return score, tier


def create_checkpoint(model_key: str, label: str | None = None) -> Path:
    db = sqlite3.connect(settings.DATABASE_PATH)
    db.row_factory = sqlite3.Row
    positions = [dict(row) for row in db.execute(
        """SELECT p.*,d.model_name AS entry_model,d.provider AS entry_provider,d.confidence,
                  d.predicted_up_probability,d.predicted_down_probability,d.expected_net_edge,
                  d.reason AS entry_reason,d.tags_json,d.market_state_json,s.strategy_version,s.run_label
           FROM paper_positions p JOIN model_decisions d ON d.id=p.entry_decision_id
           JOIN paper_sessions s ON s.session_id=p.session_id
           WHERE d.model_name=? ORDER BY p.opened_at""", (model_key,),
    )]
    if not positions:
        raise RuntimeError(f"No positions for model={model_key}")
    completed = [row for row in positions if row["status"] in {"closed", "resolved"}]
    checkpoint_label = label or f"{datetime.now(UTC):%Y-%m-%d}_v6_{model_key}_{len(positions)}_entries"
    output = settings.MODEL_DIR / "checkpoints" / model_key / checkpoint_label
    if output.exists():
        raise RuntimeError(f"Checkpoint already exists: {output}")
    output.mkdir(parents=True)

    pnls = [float(row["realized_pnl_usdc"] or 0) for row in completed]
    wins = sum(value > 0 for value in pnls)
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = abs(sum(value for value in pnls if value < 0))
    pnl_mean = mean(pnls) if pnls else 0.0
    pnl_se = stdev(pnls) / math.sqrt(len(pnls)) if len(pnls) > 1 else 0.0
    z_score = pnl_mean / pnl_se if pnl_se > 0 else 0.0
    p_value_positive = 1.0 - NormalDist().cdf(z_score) if pnl_se > 0 else 1.0
    equity, peak, max_drawdown = 300.0, 300.0, 0.0
    equity_curve = []
    for index, row in enumerate(completed, 1):
        equity += float(row["realized_pnl_usdc"] or 0)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
        equity_curve.append({"index": index, "closed_at": row["closed_at"], "event_slug": row["event_slug"], "equity": equity})

    labels: list[int] = []
    probabilities: list[float] = []
    hold_pnl = 0.0
    inverse_hold_pnl = 0.0
    hold_comparable = 0
    enriched = []
    for row in positions:
        order = db.execute(
            "SELECT * FROM paper_orders WHERE decision_id=? AND action IN ('BUY_UP','BUY_DOWN') ORDER BY id LIMIT 1",
            (row["entry_decision_id"],),
        ).fetchone()
        label_row = db.execute(
            "SELECT label FROM training_examples WHERE event_slug=? AND outcome=? LIMIT 1",
            (row["event_slug"], row["outcome"]),
        ).fetchone()
        item = dict(row)
        if order and label_row:
            outcome_label = int(label_row[0])
            held_probability = (
                float(row["predicted_up_probability"] or 0.5)
                if row["outcome"] == "Up" else float(row["predicted_down_probability"] or 0.5)
            )
            labels.append(outcome_label)
            probabilities.append(max(1e-6, min(1 - 1e-6, held_probability)))
            entry_notional = float(order["notional_usdc"] or 0)
            entry_shares = float(order["shares"] or 0)
            entry_fee = float(order["fee_usdc"] or 0)
            hypothetical_hold = entry_shares * outcome_label - entry_notional - entry_fee
            hold_pnl += hypothetical_hold
            state = json.loads(str(row["market_state_json"] or "{}"))
            opposite = "Down" if row["outcome"] == "Up" else "Up"
            opposite_ask = (state.get("book_json") or {}).get(opposite, {}).get("best_ask")
            inverse = None
            if opposite_ask and 0 < float(opposite_ask) < 1:
                inverse_shares = entry_notional / float(opposite_ask)
                inverse_fee = total_fee_usdc(inverse_shares, float(opposite_ask))
                inverse = inverse_shares * (1 - outcome_label) - entry_notional - inverse_fee
                inverse_hold_pnl += inverse
            hold_comparable += 1
            reward_score, reward_tier = _reward(float(row["realized_pnl_usdc"] or 0), entry_notional)
            item.update({
                "resolved_label": outcome_label, "held_probability_at_entry": held_probability,
                "hold_to_resolution_pnl": hypothetical_hold, "inverse_hold_pnl": inverse,
                "reward_score": reward_score, "reward_tier": reward_tier,
            })
        enriched.append(item)

    calibration: dict[str, Any] = {"examples": len(labels)}
    if labels and len(set(labels)) == 2:
        calibration.update({
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "brier": float(brier_score_loss(labels, probabilities)),
            "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
            "mean_predicted_probability": mean(probabilities),
            "actual_success_rate": mean(labels),
        })

    direction = Counter(str(row["outcome"]) for row in completed)
    direction_pnl: dict[str, float] = defaultdict(float)
    close_reason = Counter()
    stage = Counter()
    for row in completed:
        direction_pnl[str(row["outcome"])] += float(row["realized_pnl_usdc"] or 0)
        close_reason[str(row["close_reason"] or row["status"])] += 1
        stage[str(int(row["exit_stage"] or 0))] += 1

    total_wagered = sum(float(db.execute(
        "SELECT COALESCE(SUM(notional_usdc),0) FROM paper_orders WHERE decision_id=? AND action IN ('BUY_UP','BUY_DOWN')",
        (row["entry_decision_id"],),
    ).fetchone()[0]) for row in positions)
    metrics = {
        "model": model_key, "entries": len(positions), "completed": len(completed),
        "open": len(positions) - len(completed), "independent_events": len({row["event_slug"] for row in positions}),
        "wins": wins, "losses": sum(value < 0 for value in pnls),
        "win_rate": wins / len(pnls) if pnls else 0.0,
        "wilson_lower_95": wilson_lower_bound(wins, len(pnls)),
        "net_pnl_usdc": sum(pnls), "total_wagered_usdc": total_wagered,
        "roi_on_wagered": sum(pnls) / total_wagered if total_wagered else 0.0,
        "fees_usdc": sum(float(row["fees_usdc"] or 0) for row in positions),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "expectancy_usdc": pnl_mean, "median_pnl_usdc": median(pnls) if pnls else 0.0,
        "mean_pnl_ci95": [pnl_mean - 1.96 * pnl_se, pnl_mean + 1.96 * pnl_se],
        "one_sided_p_value_mean_gt_zero": p_value_positive,
        "max_drawdown_usdc": max_drawdown,
        "hold_comparable": hold_comparable, "hold_to_resolution_pnl": hold_pnl,
        "inverse_hold_pnl": inverse_hold_pnl,
        "directions": dict(direction), "direction_pnl": dict(direction_pnl),
        "close_reasons": dict(close_reason), "final_exit_stages": dict(stage),
        "calibration": calibration,
        "production_ready": False,
        "production_blockers": [
            "fewer than configured 300 independent live-gate events",
            "paper limit fills do not yet model queue/non-fill probability",
            "single short market regime",
            "live executor is not implemented",
        ],
    }

    with (output / "positions_enriched.jsonl").open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    decisions = [dict(row) for row in db.execute(
        "SELECT * FROM model_decisions WHERE model_name=? ORDER BY id", (model_key,),
    )]
    with (output / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    db.close()

    model_root = settings.QWEN_LOCAL_PATH.parent if model_key == "qwen" else settings.MODEL_DIR / model_key
    manifest = []
    for path in sorted(item for item in model_root.rglob("*") if item.is_file()):
        manifest.append({"path": str(path.relative_to(model_root)), "size": path.stat().st_size, "sha256": _hash(path)})
    for relative in ("model.json", "weights/config.json", "weights/generation_config.json", "weights/tokenizer_config.json"):
        source = model_root / relative
        if source.exists():
            destination = output / "model_metadata" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    config_keys = [
        "MIN_ENTRY_CONFIDENCE", "PAPER_MIN_ENTRY_NET_EDGE", "PAPER_MIN_ENTRY_PRICE", "PAPER_MAX_ENTRY_PRICE",
        "PAPER_MIN_ENTRY_SECONDS_AFTER_OPEN", "PAPER_LAST_ENTRY_SECONDS_BEFORE_CLOSE", "PAPER_ENTRY_NOTIONAL_USDC",
        "PAPER_MAX_EVENT_EXPOSURE_USDC", "FIVE_STAGE_EXIT_ENABLED", "EXIT_STAGE_REMAINING_FRACTIONS",
        "EXIT_STAGE_PROFIT_RETURN_PCT", "EXIT_STAGE_MAX_HELD_PROBABILITY", "EXIT_RISK_SIGNAL_CONFIRMATIONS",
        "EXIT_RISK_CONFIRMATION_MIN_SECONDS", "ENTRY_ORDER_TYPE", "EXIT_ORDER_TYPE", "TAKE_PROFIT_ORDER_TYPE",
    ]
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "equity_curve.json").write_text(json.dumps(equity_curve, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "weights_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "config_snapshot.json").write_text(
        json.dumps({key: getattr(settings, key) for key in config_keys}, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "README.md").write_text(
        f"# {checkpoint_label}\n\nCheckpoint поведения `{model_key}`: {len(positions)} входов, "
        f"{len(completed)} завершённых. Базовые веса не дублируются (≈4 GB); их SHA-256 находятся "
        "в `weights_manifest.json`. Торговая конфигурация, решения и результаты сохранены отдельно.\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--label")
    args = parser.parse_args()
    print(create_checkpoint(args.model, args.label))


if __name__ == "__main__":
    from polybot.runtime import run_sync

    run_sync(__file__, main)
