"""Multi-task обучение временных моделей; запускается только в CUDA-окружении."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import app_config as settings
from polybot.models.gpu_networks import build_network
from polybot.models.sequence_dataset import FEATURES
from polybot.models.temporal_validation import purged_expanding_folds


def _dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(model_name: str, epochs: int, selected_fold: int | None = None) -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if not torch.cuda.is_available():
        raise RuntimeError("Обучение остановлено: CUDA недоступна.")
    random.seed(settings.GPU_TRAINING_RANDOM_SEED)
    np.random.seed(settings.GPU_TRAINING_RANDOM_SEED)
    torch.manual_seed(settings.GPU_TRAINING_RANDOM_SEED)
    table = pq.read_table(settings.SEQUENCE_DATASET_PATH)
    frame = table.to_pandas().sort_values("observed_at").reset_index(drop=True)
    events = frame["event_slug"].drop_duplicates().tolist()
    plan = json.loads(Path(settings.GPU_TRAINING_PLAN_PATH).read_text(encoding="utf-8"))
    validation = plan["validation"]
    folds = purged_expanding_folds(events, folds=validation["minimum_folds"], purge=validation["purge_events"], embargo=validation["embargo_events"])
    if selected_fold is not None:
        folds = [folds[selected_fold]]
    feature_values = np.stack([
        np.stack([np.asarray(value, dtype=np.float32) for value in frame[name]]) for name in FEATURES
    ], axis=-1)
    direction = frame["resolution_label"].to_numpy(np.float32)
    utility = np.stack([
        frame[name].fillna(np.nan).to_numpy(np.float32)
        for name in ("utility_hold", "utility_up", "utility_down")
    ], axis=-1)
    fill = frame["best_action_filled"].fillna(np.nan).to_numpy(np.float32)
    output_root = Path(settings.MODEL_DIR) / "gpu_challengers" / f"{model_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=False)
    reports = []
    device = torch.device("cuda")
    for fold_index, fold in enumerate(folds):
        train_mask = frame["event_slug"].isin(fold.train_events).to_numpy()
        val_mask = frame["event_slug"].isin(fold.validation_events).to_numpy()
        mean = feature_values[train_mask].mean(axis=(0, 1), keepdims=True)
        std = feature_values[train_mask].std(axis=(0, 1), keepdims=True).clip(1e-5)
        normalized = (feature_values - mean) / std

        def loader(mask, shuffle):
            tensors = [torch.from_numpy(values[mask]) for values in (normalized, direction, utility, fill)]
            return DataLoader(TensorDataset(*tensors), batch_size=settings.GPU_TRAINING_BATCH_SIZE, shuffle=shuffle,
                              num_workers=settings.GPU_NUM_WORKERS, pin_memory=True)

        network = build_network(model_name, len(FEATURES)).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4, weight_decay=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=settings.GPU_MIXED_PRECISION)
        best_loss, patience, best_state = float("inf"), 0, None
        for epoch in range(epochs):
            network.train()
            for x, y_direction, y_utility, y_fill in loader(train_mask, True):
                x, y_direction, y_utility, y_fill = [value.to(device, non_blocking=True) for value in (x, y_direction, y_utility, y_fill)]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=settings.GPU_MIXED_PRECISION):
                    prediction = network(x)
                    loss = nn.functional.binary_cross_entropy_with_logits(prediction["direction_logit"], y_direction)
                    utility_mask = torch.isfinite(y_utility)
                    if utility_mask.any():
                        loss = loss + 0.35 * nn.functional.smooth_l1_loss(prediction["action_utility"][utility_mask], y_utility[utility_mask])
                    fill_mask = torch.isfinite(y_fill)
                    if fill_mask.any():
                        loss = loss + 0.20 * nn.functional.binary_cross_entropy_with_logits(prediction["fill_logit"][fill_mask], y_fill[fill_mask])
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            network.eval(); losses = []
            with torch.no_grad():
                for x, y_direction, *_ in loader(val_mask, False):
                    prediction = network(x.to(device, non_blocking=True))
                    losses.append(float(nn.functional.binary_cross_entropy_with_logits(prediction["direction_logit"], y_direction.to(device)).cpu()))
            validation_loss = float(np.mean(losses))
            if validation_loss < best_loss - 1e-5:
                best_loss, patience = validation_loss, 0
                best_state = {key: value.detach().cpu() for key, value in network.state_dict().items()}
            else:
                patience += 1
                if patience >= settings.GPU_TRAINING_EARLY_STOPPING_PATIENCE:
                    break
        torch.save({"state_dict": best_state, "mean": mean, "std": std, "features": FEATURES}, output_root / f"fold_{fold_index}.pt")
        reports.append({"fold": fold_index, "validation_loss": best_loss, "train_events": len(fold.train_events), "test_events": len(fold.test_events)})
    report = {"model": model_name, "created_at": datetime.now(UTC).isoformat(), "dataset_sha256": _dataset_hash(settings.SEQUENCE_DATASET_PATH), "folds": reports}
    (output_root / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("tcn", "gru", "tiny_transformer"), default="tcn")
    parser.add_argument("--epochs", type=int, default=settings.GPU_TRAINING_MAX_EPOCHS)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    print(json.dumps(train(args.model, args.epochs, args.fold), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
