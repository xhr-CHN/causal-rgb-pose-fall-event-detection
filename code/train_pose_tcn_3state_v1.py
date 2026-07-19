from __future__ import annotations

import csv
import json
import math
import platform
import random
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_pose_mlp_ablation_v1 import ranking_metrics
from train_pose_tcn_baseline_v1 import PoseTCN


DATA_DIR = Path("/home/data/yoloA27/features/caucafall_pose_windows_3state_v1")
OUTPUT_DIR = Path("/home/data/yoloA27/experiments/pose_tcn_3state_v1_seed42")

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CAUCAFALL_FPS = 20.0

STATE_NAMES = ("ADL", "Falling", "Fallen")

# Source-domain state-machine policy. No target-domain data is used here.
FALLING_REQUIRED = 2
FALLING_LOOKBACK_FRAMES = 20
ARM_MEMORY_FRAMES = 30
FALLEN_CONSECUTIVE_FRAMES = 5
ADL_RESET_FRAMES = 10


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
        raise ValueError(f"Cannot save empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def multiclass_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_pred = np.argmax(probabilities, axis=1)
    matrix = np.zeros((3, 3), dtype=int)
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth), int(prediction)] += 1

    per_class = {}
    f1_values = []
    recalls = []
    supports = []
    auprc_values = []
    auroc_values = []
    for class_id, class_name in enumerate(STATE_NAMES):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        support = int(matrix[class_id, :].sum())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        binary_true = (y_true == class_id).astype(int)
        auroc, auprc = ranking_metrics(binary_true, probabilities[:, class_id])
        per_class[class_name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc_ovr": auroc,
            "auprc_ovr": auprc,
        }
        f1_values.append(f1)
        recalls.append(recall)
        supports.append(support)
        auprc_values.append(auprc)
        auroc_values.append(auroc)

    supports_array = np.asarray(supports, dtype=float)
    f1_array = np.asarray(f1_values, dtype=float)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1_array)),
        "weighted_f1": float(np.sum(f1_array * supports_array) / np.sum(supports_array)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_auprc_ovr": float(np.mean(auprc_values)),
        "macro_auroc_ovr": float(np.mean(auroc_values)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def load_data():
    train_npz = np.load(DATA_DIR / "train_windows.npz")
    val_npz = np.load(DATA_DIR / "val_windows.npz")
    x_train = train_npz["x"].astype(np.float32)
    y_train = train_npz["y"].astype(np.int64)
    x_val = val_npz["x"].astype(np.float32)
    y_val = val_npz["y"].astype(np.int64)
    train_names = train_npz["feature_names"].astype(str)
    val_names = val_npz["feature_names"].astype(str)
    state_names = train_npz["state_names"].astype(str)

    if not np.array_equal(train_names, val_names):
        raise ValueError("Train/validation feature names differ")
    if tuple(state_names.tolist()) != STATE_NAMES:
        raise ValueError(f"Unexpected state names: {state_names}")
    if x_train.shape != (2513, 32, 103):
        raise ValueError(f"Unexpected train shape: {x_train.shape}")
    if x_val.shape != (3450, 32, 103):
        raise ValueError(f"Unexpected validation shape: {x_val.shape}")
    if set(np.unique(y_train)) != {0, 1, 2} or set(np.unique(y_val)) != {0, 1, 2}:
        raise ValueError("All three classes must occur in train and validation")

    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    return x_train, y_train, x_val, y_val, train_names, mean, std


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


def make_model() -> PoseTCN:
    model = PoseTCN(input_features=103)
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    return model.to(DEVICE)


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
        losses.append(float(criterion(logits, y_batch).item()) * len(y_batch))
        labels.append(y_batch.cpu().numpy())
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    y_true = np.concatenate(labels)
    scores = np.concatenate(probabilities)
    metrics = multiclass_metrics(y_true, scores)
    metrics["loss"] = sum(losses) / len(y_true)
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(y_batch)
        total_examples += len(y_batch)
    return total_loss / total_examples


def state_machine_triggers(rows: list[dict]) -> list[dict]:
    triggers = []
    recent_falling: deque[int] = deque()
    armed_until = -1
    fallen_consecutive = 0
    alarm_active = False
    adl_consecutive = 0

    for index, row in enumerate(rows):
        state = int(row["predicted_state_id"])
        while recent_falling and recent_falling[0] < index - FALLING_LOOKBACK_FRAMES + 1:
            recent_falling.popleft()
        if state == 1:
            recent_falling.append(index)
            if len(recent_falling) >= FALLING_REQUIRED:
                armed_until = index + ARM_MEMORY_FRAMES

        fallen_consecutive = fallen_consecutive + 1 if state == 2 else 0
        adl_consecutive = adl_consecutive + 1 if state == 0 else 0
        if adl_consecutive >= ADL_RESET_FRAMES:
            alarm_active = False

        if (
            fallen_consecutive >= FALLEN_CONSECUTIVE_FRAMES
            and index <= armed_until
            and not alarm_active
        ):
            triggers.append(row)
            alarm_active = True
    return triggers


def event_evaluation(metadata: list[dict[str, str]], probabilities: np.ndarray):
    enriched = []
    predicted = np.argmax(probabilities, axis=1)
    for index, row in enumerate(metadata):
        item = dict(row)
        item["prob_adl"] = float(probabilities[index, 0])
        item["prob_falling"] = float(probabilities[index, 1])
        item["prob_fallen"] = float(probabilities[index, 2])
        item["predicted_state_id"] = int(predicted[index])
        item["predicted_state"] = STATE_NAMES[int(predicted[index])]
        enriched.append(item)

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in enriched:
        groups[row["sequence_id"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["target_frame_index"]))

    event_rows = []
    false_alarms = 0
    duplicate_alarms = 0
    delays = []
    for sequence_id in sorted(groups):
        rows = groups[sequence_id]
        triggers = state_machine_triggers(rows)
        if rows[0]["category"] == "fall":
            onset = int(rows[0]["onset_frame"])
            stable = int(rows[0]["stable_fallen_frame"])
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
                    "stable_fallen_frame": stable,
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
    negative_windows = sum(int(row["target_state_id"]) == 0 for row in enriched)
    negative_hours = negative_windows / CAUCAFALL_FPS / 3600.0
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


def plot_history(history: list[dict]) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history])
    axes[1].set_title("Validation macro F1")
    axes[2].plot(epochs, [row["val_macro_auprc"] for row in history], label="AUPRC")
    axes[2].plot(epochs, [row["val_macro_auroc"] for row in history], label="AUROC")
    axes[2].set_title("Macro one-vs-rest metrics")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_confusion(matrix: list[list[int]]) -> None:
    array = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(array, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axis.text(j, i, str(array[i, j]), ha="center", va="center")
    axis.set_xticks(range(3), STATE_NAMES)
    axis.set_yticks(range(3), STATE_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Validation confusion matrix")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    x_train, y_train, x_val, y_val, names, mean, std = load_data()
    train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val)
    model = make_model()

    class_counts = np.bincount(y_train, minlength=3)
    class_weights = len(y_train) / (3.0 * class_counts)
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
    print(f"Class counts: {class_counts.tolist()}", flush=True)
    print(f"Class weights: {class_weights.tolist()}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}", flush=True)

    history = []
    best_auprc = -math.inf
    best_epoch = 0
    stale = 0
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
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_auprc": val_metrics["macro_auprc_ovr"],
            "val_macro_auroc": val_metrics["macro_auroc_ovr"],
            "val_falling_f1": val_metrics["per_class"]["Falling"]["f1"],
            "val_fallen_f1": val_metrics["per_class"]["Fallen"]["f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} loss={train_loss:.4f}/{val_metrics['loss']:.4f} "
            f"macroF1={val_metrics['macro_f1']:.4f} "
            f"FallingF1={row['val_falling_f1']:.4f} "
            f"AUPRC={val_metrics['macro_auprc_ovr']:.4f}",
            flush=True,
        )
        if val_metrics["macro_auprc_ovr"] > best_auprc + 1e-6:
            best_auprc = val_metrics["macro_auprc_ovr"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), state_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    save_csv(OUTPUT_DIR / "history.csv", history)
    plot_history(history)
    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    validation_metrics, y_true, probabilities = evaluate(model, val_loader, criterion)
    metadata = read_csv(DATA_DIR / "val_window_metadata.csv")
    if len(metadata) != len(probabilities):
        raise RuntimeError("Validation metadata and predictions differ in length")
    if not np.array_equal(
        y_true, np.asarray([int(row["target_state_id"]) for row in metadata])
    ):
        raise RuntimeError("Validation labels do not match metadata")

    event_rows, event_summary, predictions = event_evaluation(metadata, probabilities)
    save_csv(OUTPUT_DIR / "val_predictions.csv", predictions)
    save_csv(OUTPUT_DIR / "val_event_results.csv", event_rows)
    plot_confusion(validation_metrics["confusion_matrix"])

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_mean": torch.from_numpy(mean.squeeze()).float(),
        "input_std": torch.from_numpy(std.squeeze()).float(),
        "feature_names": names.tolist(),
        "state_names": list(STATE_NAMES),
        "config": {
            "window_length": 32,
            "input_features": 103,
            "state_machine": {
                "falling_required": FALLING_REQUIRED,
                "falling_lookback_frames": FALLING_LOOKBACK_FRAMES,
                "arm_memory_frames": ARM_MEMORY_FRAMES,
                "fallen_consecutive_frames": FALLEN_CONSECUTIVE_FRAMES,
                "adl_reset_frames": ADL_RESET_FRAMES,
            },
        },
    }
    torch.save(checkpoint, OUTPUT_DIR / "pose_tcn_3state_best.pt")

    summary = {
        "protocol": {
            "model": "three-state causal Pose-TCN",
            "states": list(STATE_NAMES),
            "seed": SEED,
            "train_subjects": [1, 3, 6, 7, 8, 9],
            "val_subjects": [4, 5],
            "test_used": False,
            "early_stopping_metric": "validation macro one-vs-rest AUPRC",
            "standardization": "train statistics only",
            "window_length": 32,
            "state_machine": checkpoint["config"]["state_machine"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "class_counts": class_counts.tolist(),
        "class_weights": class_weights.tolist(),
        "best_epoch": best_epoch,
        "best_validation_macro_auprc": best_auprc,
        "validation_frame_metrics": validation_metrics,
        "validation_event_metrics": event_summary,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Completed. Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
