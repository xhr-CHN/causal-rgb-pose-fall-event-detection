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
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path("/home/data/yoloA27/features/caucafall_pose_windows_v1")
OUTPUT_ROOT = Path("/home/data/yoloA27/experiments/pose_mlp_ablation_v1_seed42")

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
ALARM_CONSECUTIVE_FRAMES = 5
CAUCAFALL_FPS = 20.0

EXPERIMENTS = {
    "pose_mlp_static65": {
        "input_dim": 65,
        "description": "single current frame: pose, bbox and quality features",
    },
    "pose_mlp_1step103": {
        "input_dim": 103,
        "description": "current frame plus one-step bbox/keypoint deltas",
    },
}


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


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    ends = np.r_[np.where(np.diff(sorted_scores) != 0)[0], len(scores) - 1]
    tpr = np.r_[0.0, tp[ends] / positives]
    fpr = np.r_[0.0, fp[ends] / negatives]
    auroc = float(np.trapz(tpr, fpr))
    precision = tp / np.arange(1, len(y_true) + 1)
    recall = tp / positives
    auprc = float(np.sum(np.diff(np.r_[0.0, recall[ends]]) * precision[ends]))
    return auroc, auprc


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    best_threshold = 0.5
    best = binary_metrics(y_true, scores >= best_threshold)
    for threshold in np.linspace(0.0, 1.0, 1001):
        metrics = binary_metrics(y_true, scores >= threshold)
        if metrics["f1"] > best["f1"]:
            best_threshold = float(threshold)
            best = metrics
    return best_threshold, best


class PoseMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_base_data():
    train_npz = np.load(DATA_DIR / "train_windows.npz")
    val_npz = np.load(DATA_DIR / "val_windows.npz")
    x_train = train_npz["x"].astype(np.float32)
    y_train = train_npz["y"].astype(np.int64)
    x_val = val_npz["x"].astype(np.float32)
    y_val = val_npz["y"].astype(np.int64)
    names = train_npz["feature_names"].astype(str)
    if not np.array_equal(names, val_npz["feature_names"].astype(str)):
        raise ValueError("Train and val feature names differ")
    if x_train.shape != (2513, 32, 103) or x_val.shape != (3450, 32, 103):
        raise ValueError(f"Unexpected shapes: train={x_train.shape}, val={x_val.shape}")
    return x_train, y_train, x_val, y_val, names


def prepare_input(x_train, x_val, input_dim: int):
    train = x_train[:, -1, :input_dim].copy()
    val = x_val[:, -1, :input_dim].copy()
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (val - mean) / std, mean, std


def make_loaders(x_train, y_train, x_val, y_val):
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses, labels, probabilities = [], [], []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)
        logits = model(x_batch)
        losses.append(float(criterion(logits, y_batch).item()) * len(y_batch))
        labels.append(y_batch.cpu().numpy())
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    y_true = np.concatenate(labels)
    scores = np.concatenate(probabilities)
    metrics = binary_metrics(y_true, scores >= 0.5)
    auroc, auprc = ranking_metrics(y_true, scores)
    metrics.update({"loss": sum(losses) / len(y_true), "auroc": auroc, "auprc": auprc})
    return metrics, y_true, scores


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE, non_blocking=True)
        y_batch = y_batch.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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


def event_evaluation(metadata, probabilities, threshold):
    enriched = []
    for index, row in enumerate(metadata):
        item = dict(row)
        item["probability"] = float(probabilities[index])
        item["prediction"] = int(probabilities[index] >= threshold)
        enriched.append(item)
    groups = {}
    for row in enriched:
        groups.setdefault(row["sequence_id"], []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["target_frame_index"]))

    event_rows, delays = [], []
    false_alarms = 0
    duplicate_alarms = 0
    for sequence_id in sorted(groups):
        rows = groups[sequence_id]
        triggers = alarm_triggers(rows)
        if rows[0]["category"] == "fall":
            first_positive = next(row for row in rows if int(row["target_label"]) == 1)
            onset = int(first_positive["target_frame_index"]) - int(first_positive["positive_frames_in_window"]) + 1
            valid = [row for row in triggers if int(row["target_frame_index"]) >= onset]
            early = [row for row in triggers if int(row["target_frame_index"]) < onset]
            false_alarms += len(early)
            duplicate_alarms += max(0, len(valid) - 1)
            if valid:
                alarm_frame = int(valid[0]["target_frame_index"])
                delay = alarm_frame - onset
                delays.append(delay)
                detected = 1
            else:
                alarm_frame, delay, detected = "", "", 0
            event_rows.append({
                "sequence_id": sequence_id,
                "onset_frame": onset,
                "detected": detected,
                "first_alarm_frame": alarm_frame,
                "detection_delay_frames": delay,
                "detection_delay_seconds": delay / CAUCAFALL_FPS if delay != "" else "",
                "early_false_alarms": len(early),
                "alarms_after_onset": len(valid),
            })
        else:
            false_alarms += len(triggers)

    detected = sum(int(row["detected"]) for row in event_rows)
    negative_frames = sum(int(row["target_label"]) == 0 for row in enriched)
    negative_hours = negative_frames / CAUCAFALL_FPS / 3600.0
    summary = {
        "fall_events": len(event_rows),
        "detected_events": detected,
        "missed_events": len(event_rows) - detected,
        "event_recall": safe_div(detected, len(event_rows)),
        "false_alarm_count": false_alarms,
        "negative_exposure_hours": negative_hours,
        "false_alarms_per_hour": safe_div(false_alarms, negative_hours),
        "duplicate_alarms_after_onset": duplicate_alarms,
        "mean_delay_frames": float(np.mean(delays)) if delays else None,
        "median_delay_frames": float(np.median(delays)) if delays else None,
        "mean_delay_seconds": float(np.mean(delays) / CAUCAFALL_FPS) if delays else None,
        "median_delay_seconds": float(np.median(delays) / CAUCAFALL_FPS) if delays else None,
    }
    return event_rows, summary, enriched


def plot_history(history, output_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_f1_at_0.5"] for r in history])
    axes[1].set_title("Validation F1 at 0.5")
    axes[2].plot(epochs, [r["val_auprc"] for r in history], label="AUPRC")
    axes[2].plot(epochs, [r["val_auroc"] for r in history], label="AUROC")
    axes[2].set_title("Ranking metrics")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_confusion(metrics, output_dir: Path) -> None:
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
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def run_experiment(name, config, base_data, metadata):
    set_seed(SEED)
    output_dir = OUTPUT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=False)
    x_train_all, y_train, x_val_all, y_val, feature_names = base_data
    input_dim = int(config["input_dim"])
    x_train, x_val, mean, std = prepare_input(x_train_all, x_val_all, input_dim)
    train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val)

    model = PoseMLP(input_dim).to(DEVICE)
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2.0 * class_counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    print(f"\n=== {name} ===", flush=True)
    print(f"Input dim: {input_dim}; parameters: {sum(p.numel() for p in model.parameters())}", flush=True)
    history = []
    best_auprc = -math.inf
    best_epoch = 0
    stale = 0
    state_path = output_dir / "best_model_state.pt"
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
        print(f"Epoch {epoch:03d} loss={train_loss:.4f}/{val_metrics['loss']:.4f} F1={val_metrics['f1']:.4f} AUPRC={val_metrics['auprc']:.4f}", flush=True)
        if val_metrics["auprc"] > best_auprc + 1e-6:
            best_auprc = val_metrics["auprc"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), state_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    save_csv(output_dir / "history.csv", history)
    plot_history(history, output_dir)
    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    metrics_05, y_true, probabilities = evaluate(model, val_loader, criterion)
    threshold, selected_metrics = best_f1_threshold(y_true, probabilities)
    auroc, auprc = ranking_metrics(y_true, probabilities)
    selected_metrics.update({"auroc": auroc, "auprc": auprc})
    event_rows, event_summary, enriched = event_evaluation(metadata, probabilities, threshold)
    save_csv(output_dir / "val_predictions.csv", enriched)
    save_csv(output_dir / "val_event_results.csv", event_rows)
    plot_confusion(selected_metrics, output_dir)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_mean": torch.from_numpy(mean.squeeze()).float(),
        "input_std": torch.from_numpy(std.squeeze()).float(),
        "feature_names": feature_names[:input_dim].tolist(),
        "threshold": threshold,
        "config": {"model": name, "input_dim": input_dim, "hidden_dims": [256, 128], "dropout": DROPOUT, "alarm_consecutive_frames": ALARM_CONSECUTIVE_FRAMES},
    }
    torch.save(checkpoint, output_dir / f"{name}_best.pt")
    summary = {
        "protocol": {
            "model": name,
            "description": config["description"],
            "seed": SEED,
            "train_subjects": [1, 3, 6, 7, 8, 9],
            "val_subjects": [4, 5],
            "test_used": False,
            "early_stopping_metric": "validation AUPRC",
            "threshold_selection": "maximum validation frame-level F1",
            "standardization": "train target-frame statistics only",
            "alarm_consecutive_frames": ALARM_CONSECUTIVE_FRAMES,
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_auprc_during_training": best_auprc,
        "selected_threshold": threshold,
        "validation_metrics_at_0.5": metrics_05,
        "validation_metrics_at_selected_threshold": selected_metrics,
        "validation_event_metrics": event_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    base_data = load_base_data()
    metadata = read_csv(DATA_DIR / "val_window_metadata.csv")
    if len(metadata) != len(base_data[3]):
        raise ValueError("Validation metadata and labels have different lengths")
    summaries = {}
    for name, config in EXPERIMENTS.items():
        summaries[name] = run_experiment(name, config, base_data, metadata)
    (OUTPUT_ROOT / "comparison_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nCompleted both ablations. Outputs: {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
