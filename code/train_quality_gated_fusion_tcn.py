#!/usr/bin/env python3
"""Train a source-only quality-gated RGB/Pose causal TCN on CAUCAFall.

The script never reads CAUCAFall test data or URFD. Model selection, feature
standardization, and the alarm policy are all fixed from the source train/val
split. The saved checkpoint can later be evaluated zero-shot on URFD.
"""

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEED = 42
ROOT = Path("/home/data/yoloA27")
POSE_DIR = ROOT / "features/caucafall_pose_windows_3state_kinematic_v2"
RGB_DIR = ROOT / "features/caucafall_rgb_roi_windows_3state_v1"
VAL_METADATA = (
    ROOT
    / "experiments/pose_tcn_3state_kinematic_v2_seed42/val_predictions.csv"
)
OUT_DIR = ROOT / "experiments/quality_gated_rgb_pose_tcn_v1_seed42"

BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
HIDDEN_DIM = 64
AUXILIARY_LOSS_WEIGHT = 0.15
NUM_WORKERS = 4

STATE_NAMES = ["ADL", "Falling", "Fallen"]
QUALITY_NAMES = [
    "pose_found",
    "person_conf",
    "visible_keypoint_ratio",
    "mean_keypoint_conf",
    "torso_keypoint_conf",
]

STATE_MACHINE = {
    "falling_required": 2,
    "falling_lookback_frames": 20,
    "arm_memory_frames": 30,
    "fallen_consecutive_frames": 5,
    "adl_reset_frames": 10,
}


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_split(split):
    pose = np.load(POSE_DIR / f"{split}_windows.npz", allow_pickle=True)
    rgb = np.load(RGB_DIR / f"{split}_windows.npz", allow_pickle=True)

    xp = pose["x"].astype(np.float32)
    xr = rgb["x"].astype(np.float32)
    yp = pose["y"].astype(np.int64)
    yr = rgb["y"].astype(np.int64)

    if xp.shape[:2] != xr.shape[:2]:
        raise ValueError(f"{split}: Pose/RGB window shapes are not aligned")
    if not np.array_equal(yp, yr):
        raise ValueError(f"{split}: Pose/RGB labels are not aligned")
    if int(pose["window_length"]) != int(rgb["window_length"]):
        raise ValueError(f"{split}: window lengths differ")
    if not np.array_equal(pose["state_names"], rgb["state_names"]):
        raise ValueError(f"{split}: state names differ")

    pose_names = pose["feature_names"].astype(str).tolist()
    quality_indices = []
    for name in QUALITY_NAMES:
        if name not in pose_names:
            raise ValueError(f"Required quality feature not found: {name}")
        quality_indices.append(pose_names.index(name))

    quality = np.clip(xp[:, :, quality_indices], 0.0, 1.0).astype(np.float32)
    return xp, xr, quality, yp, pose_names, rgb["feature_names"].astype(str).tolist()


def compute_stats(x):
    mean = x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = x.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


class FusionDataset(Dataset):
    def __init__(self, pose, rgb, quality, labels):
        self.pose = torch.from_numpy(pose)
        self.rgb = torch.from_numpy(rgb)
        self.quality = torch.from_numpy(quality)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.pose[index],
            self.rgb[index],
            self.quality[index],
            self.labels[index],
        )


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.left_padding,
            dilation=dilation,
        )

    def forward(self, x):
        y = self.conv(x)
        if self.left_padding:
            y = y[:, :, :-self.left_padding]
        return y


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, 3, dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, 3, dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))


class CausalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            TemporalBlock(hidden_dim, 1, dropout),
            TemporalBlock(hidden_dim, 2, dropout),
            TemporalBlock(hidden_dim, 4, dropout),
        )

    def forward(self, x):
        # Input is B,T,F; Conv1d expects B,F,T.
        x = x.transpose(1, 2)
        return self.blocks(self.input_projection(x))


class QualityGatedFusionTCN(nn.Module):
    def __init__(self, pose_dim, rgb_dim, quality_dim, hidden_dim, dropout):
        super().__init__()
        self.pose_encoder = CausalEncoder(pose_dim, hidden_dim, dropout)
        self.rgb_encoder = CausalEncoder(rgb_dim, hidden_dim, dropout)

        # The sigmoid output is the Pose weight. Low-quality pose should allow
        # the network to rely more strongly on RGB appearance information.
        self.quality_gate = nn.Sequential(
            nn.Linear(quality_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.fusion_block = TemporalBlock(hidden_dim, 1, dropout)
        self.fused_classifier = nn.Linear(hidden_dim, 3)
        self.pose_classifier = nn.Linear(hidden_dim, 3)
        self.rgb_classifier = nn.Linear(hidden_dim, 3)

    def forward(self, pose, rgb, quality):
        pose_encoded = self.pose_encoder(pose)
        rgb_encoded = self.rgb_encoder(rgb)

        pose_weight = self.quality_gate(quality).transpose(1, 2)
        fused = pose_weight * pose_encoded + (1.0 - pose_weight) * rgb_encoded
        fused = self.fusion_block(fused)

        return {
            "fused": self.fused_classifier(fused[:, :, -1]),
            "pose": self.pose_classifier(pose_encoded[:, :, -1]),
            "rgb": self.rgb_classifier(rgb_encoded[:, :, -1]),
            "pose_weight": pose_weight.squeeze(1),
        }


def binary_average_precision(targets, scores):
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(targets.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = targets[order]
    true_positives = np.cumsum(ranked)
    precision = true_positives / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def binary_roc_auc(targets, scores):
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int(targets.sum())
    negative_count = int(len(targets) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = ranks[targets == 1].sum()
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


def make_confusion_matrix(labels, predictions, classes=3):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for true_value, predicted_value in zip(labels, predictions):
        matrix[int(true_value), int(predicted_value)] += 1
    return matrix


def multiclass_metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(3, dtype=np.int64)[labels]
    matrix = make_confusion_matrix(labels, predictions)
    per_class = {}
    precisions = []
    recalls = []
    f1_values = []
    supports = []
    aps = []
    aucs = []
    for class_id, name in enumerate(STATE_NAMES):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        support = int(matrix[class_id, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ap = binary_average_precision(one_hot[:, class_id], probabilities[:, class_id])
        auc = binary_roc_auc(one_hot[:, class_id], probabilities[:, class_id])
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
        supports.append(support)
        aps.append(ap)
        aucs.append(auc)
        per_class[name] = {
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auprc_ovr": float(ap),
            "auroc_ovr": float(auc),
        }
    supports = np.asarray(supports, dtype=np.float64)
    f1_values = np.asarray(f1_values, dtype=np.float64)
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1_values.mean()),
        "weighted_f1": float((f1_values * supports).sum() / supports.sum()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_auprc_ovr": float(np.mean(aps)),
        "macro_auroc_ovr": float(np.mean(aucs)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probabilities = []
    all_labels = []
    all_pose_weights = []
    for pose, rgb, quality, labels in loader:
        pose = pose.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True)
        quality = quality.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(pose, rgb, quality)
        loss = criterion(output["fused"], labels)
        total_loss += loss.item() * len(labels)
        all_probabilities.append(torch.softmax(output["fused"], 1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_pose_weights.append(output["pose_weight"].mean(1).cpu().numpy())
    probabilities = np.concatenate(all_probabilities)
    labels = np.concatenate(all_labels)
    pose_weights = np.concatenate(all_pose_weights)
    metrics = multiclass_metrics(labels, probabilities)
    metrics["loss"] = total_loss / len(labels)
    return metrics, probabilities, labels, pose_weights


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_state_machine(metadata, predictions):
    data = [dict(row) for row in metadata]
    for row, prediction in zip(data, predictions):
        row["predicted_state_id"] = int(prediction)
    alarm_rows = []

    grouped = defaultdict(list)
    for row in data:
        grouped[row["sequence_id"]].append(row)

    for sequence_id, group in grouped.items():
        group = sorted(group, key=lambda row: int(float(row["target_frame_index"])))
        recent_falling = []
        armed_until = -1
        fallen_streak = 0
        adl_streak = 0
        first_alarm = None

        for row in group:
            frame = int(float(row["target_frame_index"]))
            state = int(row["predicted_state_id"])
            recent_falling = [
                value
                for value in recent_falling
                if frame - value < STATE_MACHINE["falling_lookback_frames"]
            ]
            if state == 1:
                recent_falling.append(frame)
            if len(recent_falling) >= STATE_MACHINE["falling_required"]:
                armed_until = max(
                    armed_until, frame + STATE_MACHINE["arm_memory_frames"]
                )

            fallen_streak = fallen_streak + 1 if state == 2 else 0
            adl_streak = adl_streak + 1 if state == 0 else 0
            if adl_streak >= STATE_MACHINE["adl_reset_frames"]:
                recent_falling = []
                armed_until = -1

            if (
                first_alarm is None
                and frame <= armed_until
                and fallen_streak >= STATE_MACHINE["fallen_consecutive_frames"]
            ):
                first_alarm = frame

        category = str(group[0]["category"])
        onset_text = group[0].get("onset_frame", "")
        onset = float(onset_text) if str(onset_text).strip() else float("nan")
        is_fall = category == "fall"
        detected = int(is_fall and first_alarm is not None and first_alarm >= onset)
        false_alarm = int((not is_fall) and first_alarm is not None)
        delay_frames = first_alarm - onset if detected else np.nan
        alarm_rows.append(
            {
                "sequence_id": sequence_id,
                "category": category,
                "onset_frame": onset,
                "detected": detected,
                "false_alarm": false_alarm,
                "first_alarm_frame": first_alarm,
                "detection_delay_frames": delay_frames,
                "detection_delay_seconds": delay_frames / 20.0 if detected else np.nan,
            }
        )

    fall_events = [row for row in alarm_rows if row["category"] == "fall"]
    detected_events = [row for row in fall_events if row["detected"] == 1]
    false_alarm_count = int(sum(row["false_alarm"] for row in alarm_rows))

    # Exposure follows the existing CAUCAFall event evaluation protocol.
    adl_frames = sum(int(float(row["target_state_id"])) == 0 for row in data)
    negative_exposure_hours = adl_frames / 20.0 / 3600.0
    delay_values = [row["detection_delay_seconds"] for row in detected_events]
    metrics = {
        "fall_events": int(len(fall_events)),
        "detected_events": int(len(detected_events)),
        "missed_events": int(len(fall_events) - len(detected_events)),
        "event_recall": float(len(detected_events) / len(fall_events)),
        "false_alarm_count": false_alarm_count,
        "negative_exposure_hours": negative_exposure_hours,
        "false_alarms_per_hour": (
            float(false_alarm_count / negative_exposure_hours)
            if negative_exposure_hours > 0
            else 0.0
        ),
        "mean_delay_seconds": (
            float(np.mean(delay_values))
            if delay_values
            else None
        ),
        "median_delay_seconds": (
            float(np.median(delay_values))
            if delay_values
            else None
        ),
    }
    return metrics, alarm_rows


def save_tabular_outputs(history, labels, probabilities):
    write_csv_rows(OUT_DIR / "history.csv", history)
    matrix = make_confusion_matrix(labels, probabilities.argmax(1))
    matrix_rows = []
    for class_id, name in enumerate(STATE_NAMES):
        matrix_rows.append(
            {
                "true_state": name,
                "pred_adl": int(matrix[class_id, 0]),
                "pred_falling": int(matrix[class_id, 1]),
                "pred_fallen": int(matrix[class_id, 2]),
            }
        )
    write_csv_rows(OUT_DIR / "confusion_matrix.csv", matrix_rows)


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_pose, train_rgb, train_quality, train_y, pose_names, rgb_names = load_split("train")
    val_pose, val_rgb, val_quality, val_y, _, _ = load_split("val")

    pose_mean, pose_std = compute_stats(train_pose)
    rgb_mean, rgb_std = compute_stats(train_rgb)
    train_pose = (train_pose - pose_mean) / pose_std
    val_pose = (val_pose - pose_mean) / pose_std
    train_rgb = (train_rgb - rgb_mean) / rgb_std
    val_rgb = (val_rgb - rgb_mean) / rgb_std

    train_loader = DataLoader(
        FusionDataset(train_pose, train_rgb, train_quality, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        FusionDataset(val_pose, val_rgb, val_quality, val_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    counts = np.bincount(train_y, minlength=3)
    class_weights = len(train_y) / (3.0 * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    model = QualityGatedFusionTCN(
        pose_dim=train_pose.shape[2],
        rgb_dim=train_rgb.shape[2],
        quality_dim=train_quality.shape[2],
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_score = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train windows: {len(train_y)} | Val windows: {len(val_y)}")
    print(f"Class counts: {counts.tolist()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for pose, rgb, quality, labels in train_loader:
            pose = pose.to(device, non_blocking=True)
            rgb = rgb.to(device, non_blocking=True)
            quality = quality.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(pose, rgb, quality)
                fused_loss = criterion(output["fused"], labels)
                pose_loss = criterion(output["pose"], labels)
                rgb_loss = criterion(output["rgb"], labels)
                loss = fused_loss + AUXILIARY_LOSS_WEIGHT * (pose_loss + rgb_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * len(labels)

        train_loss = running_loss / len(train_y)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        score = val_metrics["macro_auprc_ovr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_macro_auprc": score,
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"macro_AUPRC={score:.5f} macro_F1={val_metrics['macro_f1']:.5f}",
            flush=True,
        )

        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "pose_mean": pose_mean.squeeze().astype(np.float32),
                    "pose_std": pose_std.squeeze().astype(np.float32),
                    "rgb_mean": rgb_mean.squeeze().astype(np.float32),
                    "rgb_std": rgb_std.squeeze().astype(np.float32),
                    "pose_feature_names": pose_names,
                    "rgb_feature_names": rgb_names,
                    "quality_feature_names": QUALITY_NAMES,
                    "quality_feature_indices": list(range(5)),
                    "state_names": STATE_NAMES,
                    "state_machine": STATE_MACHINE,
                    "model_config": {
                        "pose_dim": train_pose.shape[2],
                        "rgb_dim": train_rgb.shape[2],
                        "quality_dim": train_quality.shape[2],
                        "hidden_dim": HIDDEN_DIM,
                        "dropout": DROPOUT,
                        "window_length": train_pose.shape[1],
                    },
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_validation_macro_auprc": best_score,
                },
                OUT_DIR / "quality_gated_fusion_best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(OUT_DIR / "quality_gated_fusion_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, probabilities, labels, pose_weights = evaluate(
        model, val_loader, criterion, device
    )

    metadata = read_csv_rows(VAL_METADATA)
    if len(metadata) != len(labels):
        raise ValueError("Validation metadata row count does not match predictions")
    metadata_labels = np.asarray(
        [int(float(row["target_state_id"])) for row in metadata], dtype=np.int64
    )
    if not np.array_equal(metadata_labels, labels):
        raise ValueError("Validation metadata labels do not match NPZ labels")

    predicted_states = probabilities.argmax(1)
    output_metadata = []
    for index, source_row in enumerate(metadata):
        row = dict(source_row)
        row["prob_adl_fusion"] = float(probabilities[index, 0])
        row["prob_falling_fusion"] = float(probabilities[index, 1])
        row["prob_fallen_fusion"] = float(probabilities[index, 2])
        row["predicted_state_id_fusion"] = int(predicted_states[index])
        row["predicted_state_fusion"] = STATE_NAMES[predicted_states[index]]
        row["mean_pose_gate_weight"] = float(pose_weights[index])
        output_metadata.append(row)
    write_csv_rows(OUT_DIR / "val_predictions.csv", output_metadata)

    event_metrics, event_results = run_state_machine(metadata, predicted_states)
    write_csv_rows(OUT_DIR / "val_event_results.csv", event_results)

    gate_stats = {}
    for class_id, name in enumerate(STATE_NAMES):
        values = pose_weights[labels == class_id]
        gate_stats[name] = {
            "count": int(len(values)),
            "mean_pose_weight": float(values.mean()),
            "std_pose_weight": float(values.std()),
        }

    summary = {
        "protocol": {
            "model": "quality-gated dual-stream RGB/Pose causal TCN",
            "source_dataset": "CAUCAFall",
            "test_used": False,
            "external_data_used": False,
            "seed": SEED,
            "early_stopping_metric": "validation macro one-vs-rest AUPRC",
            "window_length": 32,
            "pose_features": 95,
            "rgb_features": 256,
            "quality_features": QUALITY_NAMES,
            "state_machine": STATE_MACHINE,
        },
        "environment": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "class_counts": counts.tolist(),
        "class_weights": class_weights.tolist(),
        "best_epoch": best_epoch,
        "best_validation_macro_auprc": best_score,
        "validation_frame_metrics": val_metrics,
        "validation_event_metrics": event_metrics,
        "gate_statistics": gate_stats,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(json_ready(summary), file, ensure_ascii=False, indent=2)

    save_tabular_outputs(history, labels, probabilities)
    print("\nTraining completed.")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
