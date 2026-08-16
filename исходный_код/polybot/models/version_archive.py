"""Версионирование прибыльных моделей без их постоянного развёртывания в памяти."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app_config as settings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifacts(model: str) -> list[Path]:
    if model == "custom":
        return [settings.TRAINING_ARTIFACT_PATH]
    if model == "catboost":
        return [settings.CATBOOST_ARTIFACT_PATH, settings.CATBOOST_METADATA_PATH]
    if model == "qwen_lora" and settings.QWEN_LORA_ADAPTER_PATH.exists():
        return [path for path in settings.QWEN_LORA_ADAPTER_PATH.rglob("*") if path.is_file()]
    return []


def archive_profitable_session(connection: sqlite3.Connection, session_id: str) -> list[Path]:
    """Сохраняет отдельную версию каждого прибыльного entry-моделя с результатами сессии."""
    connection.row_factory = sqlite3.Row
    session = connection.execute("SELECT * FROM paper_sessions WHERE session_id=?", (session_id,)).fetchone()
    if not session:
        return []
    rows = connection.execute(
        """SELECT COALESCE(d.model_name,s.model_name,'unknown') model,
                  COUNT(DISTINCT p.event_slug) events,
                  SUM(CASE WHEN p.status IN ('closed','resolved') THEN 1 ELSE 0 END) completed,
                  COALESCE(SUM(CASE WHEN p.status IN ('closed','resolved') THEN p.realized_pnl_usdc ELSE 0 END),0) pnl,
                  SUM(CASE WHEN p.outcome='Up' THEN 1 ELSE 0 END) up,
                  SUM(CASE WHEN p.outcome='Down' THEN 1 ELSE 0 END) down
           FROM paper_positions p JOIN paper_sessions s ON s.session_id=p.session_id
           LEFT JOIN model_decisions d ON d.id=p.entry_decision_id
           WHERE p.session_id=? GROUP BY COALESCE(d.model_name,s.model_name,'unknown')""",
        (session_id,),
    ).fetchall()
    created: list[Path] = []
    for row in rows:
        model, pnl, completed = str(row["model"]), float(row["pnl"] or 0), int(row["completed"] or 0)
        if pnl <= 0 or completed <= 0:
            continue
        started = str(session["started_at"] or datetime.now(UTC).isoformat()).replace(":", "-")[:19]
        version = f"{model}_v_{started}_{session_id[:8]}"
        output = settings.MODEL_VERSION_ARCHIVE_DIR / model / version
        output.mkdir(parents=True, exist_ok=True)
        copied = []
        for source in _artifacts(model):
            if not source.exists():
                continue
            destination = output / "artifacts" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append({
                "name": source.name, "size": destination.stat().st_size, "sha256": _sha256(destination),
            })
        result: dict[str, Any] = {
            "version": version, "model": model, "session_id": session_id,
            "strategy_version": session["strategy_version"], "run_label": session["run_label"],
            "events": int(row["events"] or 0), "completed": completed,
            "net_pnl_usdc": pnl, "up": int(row["up"] or 0), "down": int(row["down"] or 0),
            "statistically_confirmed": completed >= 100,
            "deployment_state": "archived_not_loaded",
            "created_at": datetime.now(UTC).isoformat(), "artifacts": copied,
        }
        (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        created.append(output)
    return created

