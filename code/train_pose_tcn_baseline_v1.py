from __future__ import annotations

import csv
import json
import math
import platform
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path("/home/data/yoloA27/features/caucafall_pose_windows_v1")
OUTPUT_DIR = Path("/home/data/yoloA27/experiments/pose_tcn_baseline_v1_seed42")

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
CHANNELS = (64, 64, 128)
KERNEL_SIZE = 3
ALARM_CONSECUTIVE_FRAMES = 5
CAUCAFALL_FPS = 20.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return 0.0, 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y_true[order]
    sorted_scores = scores[order]
    true_positives = np.cumsum(sorted_y)
    false_positives = np.cumsum(1 - sorted_y)
    threshold_ends = np.r_[
        np.where(np.diff(sorted_scores) != 0)[0], len(sorted_scores) - 1
    ]

    tpr = np.r_[0.0, true_positives[threshold_ends] / positives]
    fpr = np.r_[0.0, false_positives[threshold_ends] / negatives]
    auroc = float(np.trapz(tpr, fpr))

    precision = true_positives / np.arange(1, len(y_true) + 1)
    recall = true_positives / positives
    recall_points = recall[threshold_ends]
    precision_points = precision[threshold_ends]
    auprc = float(
        np.sum(np.diff(np.r_[0.0, recall_points]) * precision_points)
    )
    return auroc, auprc


def best_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = binary_metrics(y_true, probabilities >= best_threshold)
    for threshold in np.linspace(0.0, 1.0, 1001):
        metrics = binary_metrics(y_true, probabilities >= threshold)
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, dilation: int = 1, kernel_size: int = 3, **kwargs):
        super().__init__(
            *args,
            dilation=dilation,
            kernel_size=kernel_size,
            padding=0,
            **kwargs,
        )
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.bn2(self.conv2(x)))
        return F.relu(x + residual)


class PoseTCN(nn.Module):
    def __init__(self, input_features: int) -> None:
        super().__init__()
        blocks = []
        in_channels = input_features
        for index, out_channels in enumerate(CHANNELS):
            blocks.append(
                TemporalResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=2**index,
                    kernel_size=KERNEL_SIZE,
                    dropout=DROPOUT,
                )
            )
            in_channels = out_channels
        self.temporal = nn.Sequential(*blocks)
        self.classifier = nn.Linear(CHANNELS[-1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.temporal(x)
        return self.classifier(x[:, :, -1])


def load_data():
    train_npz = np.load(DATA_DIR / "train_windows.npz")
    val_npz = np.load(DATA_DIR / "val_windows.npz")
    x_train = train_npz["x"].astype(np.float32)
    y_train = train_npz["y"].astype(np.int64)
    x_val = val_npz["x"].astype(np.float32)
    y_val = val_npz["y"].astype(np.int64)
    train_names = train_npz["feature_names"].astype(str)
    val_names = val_npz["feature_names"].astype(str)

    if not np.array_equal(train_names, val_names):
        raise ValueError("Train and val feature names differ")
    if x_train.shape != (2513, 32, 103):
        raise ValueError(f"Unexpected train shape: {x_train.shape}")
    if x_val.shape != (3450, 32, 103):
        raise ValueError(f"Unexpected val shape: {x_val.shape}")

    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    return x_train, y_train, x_val, y_val, train_names, mean, std


def make_loaders(x_train, y_train, x_val, y_val):
    generator = torch.Generator().manual_seed(SEED)
    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses = []
    labels = []
    probabilities = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        losses.append(float(loss.item()) * len(y_batch))
        labels.append(y_batch.cpu().numpy())
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    y_true = np.concatenate(labels)
    scores = np.concatenate(probabilities)
    predictions = (scores >= 0.5).astype(int)
    metrics = binary_metrics(y_true, predictions)
    auroc, auprc = ranking_metrics(y_true, scores)
    metrics.update(
        {
            "loss": sum(losses) / len(y_true),
            "auroc": auroc,
            "auprc": auprc,
        }
    )
    return metrics, y_true, scores


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(y_batch)
        total_examples += len(y_batch)
    return total_loss / total_examples


def alarm_triggers(rows: list[dict]) -> list[dict]:
    triggers = []
    consecutive = 0
    active = False
    for row in rows:
        if int(row["prediction"]) == 1:
            consecutive += 1
            if consecutive >= ALARM_CONSECUTIVE_FRAMES and not active:
                triggers.append(row)
                active = True
        else:
            consecutive = 0
            active = False
    return triggers


def event_evaluation(metadata: list[dict[str, str]], probabilities, threshold):
    enriched = []
    for index, row in enumerate(metadata):
        item = dict(row)
        item["probability"] = float(probabilities[index])
        item["prediction"] = int(probabilities[index] >= threshold)
        enriched.append(item)

    groups: dict[str, list[dict]] = {}
    for row in enriched:
        groups.setdefault(row["sequence_id"], []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["target_frame_index"]))

    event_rows = []
    false_alarms = 0
    duplicate_alarms = 0
    delays = []
    for sequence_id in sorted(groups):
        rows = groups[sequence_id]
        triggers = alarm_triggers(rows)
        if rows[0]["category"] == "fall":
            first_positive = next(
                (row for row in rows if int(row["target_label"]) == 1), None
            )
            if first_positive is None:
                raise ValueError(f"Fall sequence has no positive label: {sequence_id}")
            onset = (
                int(first_positive["target_frame_index"])
                - int(first_positive["positive_frames_in_window"])
                + 1
            )
            valid = [
                row for row in triggers if int(row["target_frame_index"]) >= onset
            ]
            early = [
                row for row in triggers if int(row["target_frame_index"]) < onset
            ]
            false_alarms += len(early)
            duplicate_alarms += max(0, len(valid) - 1)
            if valid:
                alarm_frame = int(valid[0]["target_frame_index"])
                delay = alarm_frame - onset
                delays.append(delay)
                detected = 1
            else:
                alarm_frame = ""
                delay = ""
                detected = 0
            event_rows.append(
                {
                    "sequence_id": sequence_id,
                    "onset_frame": onset,
                    "detected": detected,
                    "first_alarm_frame": alarm_frame,
                    "detection_delay_frames": delay,
                    "detection_delay_seconds": (
                        delay / CAUCAFALL_FPS if delay != "" else ""
                    ),
                    "early_false_alarms": len(early),
                    "alarms_after_onset": len(valid),
                }
            )
        else:
            false_alarms += len(triggers)

    detected_events = sum(int(row["detected"]) for row in event_rows)
    negative_frames = sum(int(row["target_label"]) == 0 for row in enriched)
    negative_hours = negative_frames / CAUCAFALL_FPS / 3600.0
    summary = {
        "fall_events": len(event_rows),
        "detected_events": detected_events,
        "missed_events": len(event_rows) - detected_events,
        "event_recall": safe_div(detected_events, len(event_rows)),
        "false_alarm_count": false_alarms,
        "negative_exposure_hours": negative_hours,
        "false_alarms_per_hour": safe_div(false_alarms, negative_hours),
        "duplicate_alarms_after_onset": duplicate_alarms,
        "mean_delay_frames": float(np.mean(delays)) if delays else None,
        "median_delay_frames": float(np.median(delays)) if delays else None,
        "mean_delay_seconds": (
            float(np.mean(delays) / CAUCAFALL_FPS) if delays else None
        ),
        "median_delay_seconds": (
            float(np.median(delays) / CAUCAFALL_FPS) if delays else None
        ),
    }
    return event_rows, summary, enriched


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(history: list[dict]) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_f1_at_0.5"] for row in history])
    axes[1].set_title("Validation F1 at 0.5")
    axes[2].plot(epochs, [row["val_auprc"] for row in history], label="AUPRC")
    axes[2].plot(epochs, [row["val_auroc"] for row in history], label="AUROC")
    axes[2].set_title("Ranking metrics")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_confusion(metrics: dict) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
    axis.set_xticks([0, 1], ["non_fall", "fall"])
    axis.set_yticks([0, 1], ["non_fall", "fall"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Validation confusion matrix")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    x_train, y_train, x_val, y_val, names, mean, std = load_data()
    train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val)

    model = PoseTCN(input_features=x_train.shape[-1]).to(DEVICE)
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2.0 * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )

    print(f"Device: {DEVICE}", flush=True)
    print(f"Train windows: {len(y_train)}, val windows: {len(y_val)}", flush=True)
    print(f"Class weights: {class_weights.tolist()}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}", flush=True)

    history = []
    best_auprc = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    state_path = OUTPUT_DIR / "best_model_state.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_metrics, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_precision_at_0.5": val_metrics["precision"],
            "val_recall_at_0.5": val_metrics["recall_sensitivity"],
            "val_f1_at_0.5": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"F1={val_metrics['f1']:.4f} AUPRC={val_metrics['auprc']:.4f} "
            f"AUROC={val_metrics['auroc']:.4f}",
            flush=True,
        )

        if val_metrics["auprc"] > best_auprc + 1e-6:
            best_auprc = val_metrics["auprc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), state_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    save_csv(OUTPUT_DIR / "history.csv", history)
    plot_history(history)
    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    val_metrics_05, y_true, probabilities = evaluate(model, val_loader, criterion)
    threshold, threshold_metrics = best_f1_threshold(y_true, probabilities)
    auroc, auprc = ranking_metrics(y_true, probabilities)
    threshold_metrics.update({"auroc": auroc, "auprc": auprc})

    metadata = read_csv(DATA_DIR / "val_window_metadata.csv")
    if len(metadata) != len(probabilities):
        raise ValueError("Validation metadata and predictions have different lengths")
    event_rows, event_summary, enriched = event_evaluation(
        metadata, probabilities, threshold
    )
    save_csv(OUTPUT_DIR / "val_predictions.csv", enriched)
    save_csv(OUTPUT_DIR / "val_event_results.csv", event_rows)
    plot_confusion(threshold_metrics)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_mean": torch.from_numpy(mean.squeeze()).float(),
        "input_std": torch.from_numpy(std.squeeze()).float(),
        "feature_names": names.tolist(),
        "threshold": threshold,
        "config": {
            "window_length": 32,
            "input_features": int(x_train.shape[-1]),
            "channels": list(CHANNELS),
            "kernel_size": KERNEL_SIZE,
            "dropout": DROPOUT,
            "alarm_consecutive_frames": ALARM_CONSECUTIVE_FRAMES,
        },
    }
    torch.save(checkpoint, OUTPUT_DIR / "pose_tcn_baseline_best.pt")

    summary = {
        "protocol": {
            "model": "three-block causal Pose-TCN baseline",
            "seed": SEED,
            "train_subjects": [1, 3, 6, 7, 8, 9],
            "val_subjects": [4, 5],
            "test_used": False,
            "early_stopping_metric": "validation AUPRC",
            "threshold_selection": "maximum validation frame-level F1",
            "standardization": "train statistics only",
            "window_length": 32,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_auprc_during_training": best_auprc,
        "selected_threshold": threshold,
        "validation_metrics_at_0.5": val_metrics_05,
        "validation_metrics_at_selected_threshold": threshold_metrics,
        "validation_event_metrics": event_summary,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Completed. Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
