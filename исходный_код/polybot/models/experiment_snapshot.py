"""Неизменяемый снимок модели, конфигурации и событий PAPER/LIVE."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app_config as settings
import joblib


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


def _effective_exit_implementation(selected_exit_model: str | None) -> str:
    prefix = f"{selected_exit_model or settings.DEFAULT_EXIT_MODEL} held-outcome probability"
    path = Path(settings.EXIT_MODEL_ARTIFACT_PATH)
    if path.exists():
        try:
            artifact = joblib.load(path)
            gate = artifact.get("report", {}).get("promotion_gate", {}) if isinstance(artifact, dict) else {}
            if bool(gate.get("passed")):
                return f"{prefix} + promoted learned CLOSE-vs-HOLD artifact"
        except (OSError, ValueError, TypeError):
            pass
    return f"{prefix} + analytical CLOSE-vs-HOLD fallback"


def create_snapshot(database: Path = settings.DATABASE_PATH, version: str = "v13_control") -> dict[str, Any]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = settings.MODEL_VERSION_ARCHIVE_DIR / "experiments" / f"{version}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    paper = [dict(row) for row in connection.execute(
        """SELECT p.*,COALESCE(d.model_name,s.model_name,'unknown') entry_model,d.confidence,
                  d.reason,d.tags_json,s.strategy_version,s.run_label
           FROM paper_positions p JOIN paper_sessions s ON s.session_id=p.session_id
           LEFT JOIN model_decisions d ON d.id=p.entry_decision_id ORDER BY p.opened_at"""
    )]
    live = [dict(row) for row in connection.execute(
        """SELECT p.*,COALESCE(d.model_name,'unknown') entry_model,d.confidence,d.reason,d.tags_json
           FROM live_positions p LEFT JOIN model_decisions d ON d.id=p.entry_decision_id ORDER BY p.opened_at"""
    )]
    live_orders = [dict(row) for row in connection.execute("SELECT * FROM live_orders ORDER BY id")]
    controls = {str(row[0]): str(row[1]) for row in connection.execute(
        "SELECT control_key,control_value FROM runtime_controls"
    )}
    connection.close()
    _jsonl(output / "paper_events.jsonl", paper)
    _jsonl(output / "live_events.jsonl", live)
    _jsonl(output / "live_orders.jsonl", live_orders)
    artifacts = []
    artifact_sources = {
        "custom_entry": settings.TRAINING_ARTIFACT_PATH,
        "catboost_entry": settings.CATBOOST_ARTIFACT_PATH,
        "catboost_bundle": settings.CATBOOST_METADATA_PATH,
        "exit_value": settings.EXIT_MODEL_ARTIFACT_PATH,
    }
    for key, source in artifact_sources.items():
        source = Path(source)
        if not source.exists():
            artifacts.append({"key": key, "source": str(source), "missing": True})
            continue
        destination = output / "artifacts" / f"{key}{source.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        artifacts.append({"key": key, "file": str(destination.relative_to(output)), "sha256": _sha256(destination), "size": destination.stat().st_size})

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        completed = [row for row in rows if row.get("status") in {"closed", "resolved"}]
        pnl = [float(row.get("realized_pnl_usdc") or 0) for row in completed]
        return {
            "positions": len(rows), "completed": len(completed), "net_pnl_usdc": sum(pnl),
            "wins": sum(value > 0 for value in pnl), "losses": sum(value < 0 for value in pnl),
            "up": sum(row.get("outcome") == "Up" for row in rows),
            "down": sum(row.get("outcome") == "Down" for row in rows),
            "early_exits": sum(bool(row.get("had_early_exit")) for row in rows),
            "first_event": min((str(row.get("opened_at")) for row in rows), default=None),
            "last_event": max((str(row.get("closed_at") or row.get("opened_at")) for row in rows), default=None),
        }
    record = {
        "version": version, "created_at": datetime.now(UTC).isoformat(),
        "selected_entry_model": controls.get("selected_entry_model"),
        "selected_exit_model": controls.get("selected_exit_model"),
        "effective_exit_implementation": _effective_exit_implementation(controls.get("selected_exit_model")),
        "strategy_version": settings.STRATEGY_VERSION, "run_label": settings.STRATEGY_RUN_LABEL,
        "paper": summary(paper), "live": summary(live), "artifacts": artifacts,
        "config": {
            "ml_notionals_usdc": settings.ML_POLICY_NOTIONALS_USDC,
            "position_size_tiers": settings.POSITION_SIZE_EDGE_TIERS,
            "partial_exit_enabled": settings.PARTIAL_EXIT_ENABLED,
            "exit_value_enabled": settings.EXIT_VALUE_ENABLED,
        },
    }
    (output / "snapshot.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**record, "path": str(output)}
