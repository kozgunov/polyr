"""Отдельная exit-модель: P(удерживаемый контракт победит) + денежное сравнение с CLOSE."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import app_config as settings
import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from polybot.models.artifact_versions import save_version_bundle
from polybot.models.train_exit_sequence import FEATURES


def train(dataset_path: Path, output_dir: Path | None = None) -> dict:
    rows = [row for row in pq.read_table(dataset_path).to_pylist()
            if int(float(row["seconds_in_position"])) % 5 == 0]
    events = list(dict.fromkeys(str(row["event_slug"]) for row in rows))
    boundaries = (int(len(events) * .55), int(len(events) * .70), int(len(events) * .85))
    event_split = {
        event: ("train" if index < boundaries[0] else "calibration" if index < boundaries[1]
                else "validation" if index < boundaries[2] else "test")
        for index, event in enumerate(events)
    }
    x = np.asarray([[float(row.get(name) or 0.0) for name in FEATURES] for row in rows], dtype=float)
    y = np.asarray([int(row["resolved_label"]) for row in rows], dtype=int)
    names = np.asarray([event_split[str(row["event_slug"])] for row in rows])
    masks = {name: names == name for name in ("train", "calibration", "validation", "test")}
    model = HistGradientBoostingClassifier(
        max_iter=400, max_leaf_nodes=15, learning_rate=.035, l2_regularization=10,
        min_samples_leaf=30, random_state=47,
    ).fit(x[masks["train"]], y[masks["train"]])
    raw_cal = np.clip(model.predict_proba(x[masks["calibration"]])[:, 1], 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(C=.20, random_state=47).fit(
        np.log(raw_cal / (1 - raw_cal)).reshape(-1, 1), y[masks["calibration"]],
    )

    def probabilities(mask: np.ndarray) -> np.ndarray:
        raw = np.clip(model.predict_proba(x[mask])[:, 1], 1e-6, 1 - 1e-6)
        return calibrator.predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]

    def policy_metrics(name: str, margin: float) -> dict:
        indices = np.flatnonzero(masks[name])
        predicted = probabilities(masks[name])
        grouped: dict[tuple[str, int], list[tuple[int, float]]] = {}
        for local, index in enumerate(indices.tolist()):
            row = rows[index]
            grouped.setdefault((str(row.get("source", "paper")), int(row["position_id"])), []).append(
                (index, float(predicted[local])))
        policy_pnl: list[float] = []; hold_pnl: list[float] = []; advantages: list[float] = []; outcomes: list[str] = []
        for candidates in grouped.values():
            candidates.sort(key=lambda item: str(rows[item[0]]["observed_at"]))
            baseline = float(rows[candidates[-1][0]]["hold_pnl_usdc"])
            selected = None
            for index, p_held in candidates:
                row = rows[index]
                predicted_hold = float(row["shares"]) * p_held - float(row["original_cost_usdc"])
                predicted_advantage = float(row["close_now_pnl_usdc"]) - predicted_hold
                if predicted_advantage >= margin:
                    selected = index
                    break
            realised = baseline if selected is None else float(rows[selected]["close_now_pnl_usdc"])
            policy_pnl.append(realised); hold_pnl.append(baseline)
            if selected is not None:
                advantages.append(realised - baseline); outcomes.append(str(rows[selected]["outcome"]))
        return {
            "positions": len(grouped), "closes": len(advantages),
            "close_precision": float(np.mean(np.asarray(advantages) > 0)) if advantages else 0.0,
            "policy_pnl_usdc": float(np.sum(policy_pnl)), "hold_pnl_usdc": float(np.sum(hold_pnl)),
            "advantage_vs_hold_usdc": float(np.sum(policy_pnl) - np.sum(hold_pnl)),
            "selected_advantage_mean_usdc": float(np.mean(advantages)) if advantages else 0.0,
            "up_closes": outcomes.count("Up"), "down_closes": outcomes.count("Down"),
        }

    grid = [{"margin_usdc": float(margin), **policy_metrics("validation", float(margin))}
            for margin in np.arange(-.25, 2.01, .05)]
    # Для PnL важнее сумма спасённого капитала, чем доля удачных закрытий:
    # несколько предотвращённых полных проигрышей могут окупить малые защитные
    # выходы. Precision сохраняем как диагностическую, но не оптимизируем её вместо денег.
    eligible = [item for item in grid if item["closes"] >= 20
                and item["advantage_vs_hold_usdc"] > 0 and item["selected_advantage_mean_usdc"] > 0
                and min(item["up_closes"], item["down_closes"]) >= 5]
    selected = max(eligible or grid, key=lambda item: (item["advantage_vs_hold_usdc"], item["close_precision"]))
    if not eligible:
        selected = {**selected, "margin_usdc": float("inf")}

    def metrics(name: str) -> dict:
        truth = y[masks[name]]; probability = probabilities(masks[name])
        return {
            "rows": int(masks[name].sum()), "events": len({events for events, split in event_split.items() if split == name}),
            "roc_auc": float(roc_auc_score(truth, probability)),
            "pr_auc": float(average_precision_score(truth, probability)),
            "brier": float(brier_score_loss(truth, probability)),
            "log_loss": float(log_loss(truth, probability)),
            **policy_metrics(name, float(selected["margin_usdc"])),
        }

    report = {
        "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "model": "exit_hold_probability_v19", "target": "P(held contract resolves to 1)",
        "features": FEATURES, "rows": len(rows), "events": len(events),
        "selected_validation_policy": selected,
        "train": metrics("train"), "calibration": metrics("calibration"),
        "validation": metrics("validation"), "test": metrics("test"),
    }
    test = report["test"]
    report["promotion_gate"] = {
        "passed": bool(test["events"] >= 30 and test["closes"] >= 20
                       and test["advantage_vs_hold_usdc"] > 0 and test["selected_advantage_mean_usdc"] > 0
                       and min(test["up_closes"], test["down_closes"]) >= 5),
        "candidate_only": True,
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = output_dir or settings.MODEL_CANDIDATE_DIR / f"exit_hold_probability_v19_{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    artifact_path = directory / "exit_hold_probability_v19.joblib"
    report_path = directory / "report.json"
    joblib.dump({
        "held_probability_model": model, "held_probability_calibrator": calibrator,
        "features": FEATURES, "advantage_threshold": float(selected["margin_usdc"]), "report": report,
    }, artifact_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if output_dir is None:
        report["artifact_version"] = save_version_bundle(
            "exit_hold_probability_v19", "exit", [artifact_path, report_path], report,
        )["version"]
    report.update({"artifact_path": str(artifact_path), "report_path": str(report_path)})
    return report
