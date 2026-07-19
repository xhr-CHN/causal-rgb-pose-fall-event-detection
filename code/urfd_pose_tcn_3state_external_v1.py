from __future__ import annotations

import json
import platform
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import urfd_pose_tcn_external_v1 as urfd
from train_pose_mlp_ablation_v1 import binary_metrics, ranking_metrics, safe_div
from train_pose_tcn_baseline_v1 import PoseTCN


ROOT = Path("/home/data/yoloA27/URFD")
FRAME_LABELS = ROOT / "metadata/rgb_frame_labels.csv"
MANUAL_EVENTS = ROOT / "metadata/manual_event_annotations.csv"
POSE_CSV = Path("/home/data/yoloA27/features/urfd_pose_v1/urfd_pose_features.csv")
CHECKPOINT = Path(
    "/home/data/yoloA27/experiments/pose_tcn_3state_v1_seed42/"
    "pose_tcn_3state_best.pt"
)
OUTPUT_DIR = Path(
    "/home/data/yoloA27/experiments/urfd_pose_tcn_3state_external_v1"
)

STATE_NAMES = ("ADL", "Falling", "Fallen")
STATE_TO_ID = {name: index for index, name in enumerate(STATE_NAMES)}
TARGET_FPS = 20.0
TCN_BATCH_SIZE = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def multiclass_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = np.argmax(probabilities, axis=1)
    matrix = np.zeros((3, 3), dtype=int)
    for truth, prediction in zip(y_true, predictions):
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
        f1 = safe_div(2.0 * precision * recall, precision + recall)
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
        "accuracy": float(np.mean(y_true == predictions)),
        "macro_f1": float(np.mean(f1_array)),
        "weighted_f1": float(
            np.sum(f1_array * supports_array) / np.sum(supports_array)
        ),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_auprc_ovr": float(np.mean(auprc_values)),
        "macro_auroc_ovr": float(np.mean(auroc_values)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(3, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def top_label_ece(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correctness = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(mask):
            value += float(np.mean(mask)) * abs(
                float(np.mean(confidence[mask]))
                - float(np.mean(correctness[mask]))
            )
    return float(value)


def make_three_state_model() -> PoseTCN:
    model = PoseTCN(input_features=103)
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    return model


@torch.no_grad()
def predict_windows(
    x: np.ndarray, checkpoint: dict
) -> np.ndarray:
    mean = checkpoint["input_mean"].numpy().reshape(1, 1, -1)
    std = checkpoint["input_std"].numpy().reshape(1, 1, -1)
    standardized = (x - mean) / std
    if not np.isfinite(standardized).all():
        raise ValueError("Non-finite standardized URFD features")

    model = make_three_state_model().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(standardized.astype(np.float32))),
        batch_size=TCN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    outputs = []
    for batch_index, (batch,) in enumerate(loader, start=1):
        logits = model(batch.to(DEVICE, non_blocking=True))
        outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        if batch_index % 10 == 0 or batch_index == len(loader):
            print(f"TCN inference: {batch_index}/{len(loader)} batches", flush=True)
    return np.concatenate(outputs)


def state_machine_triggers(rows: list[dict], policy: dict) -> list[dict]:
    falling_required = int(policy["falling_required"])
    lookback = int(policy["falling_lookback_frames"])
    arm_memory = int(policy["arm_memory_frames"])
    fallen_required = int(policy["fallen_consecutive_frames"])
    adl_reset = int(policy["adl_reset_frames"])

    triggers = []
    recent_falling: deque[int] = deque()
    armed_until = -1
    fallen_consecutive = 0
    alarm_active = False
    adl_consecutive = 0
    for index, row in enumerate(rows):
        state = int(row["predicted_state_id"])
        while recent_falling and recent_falling[0] < index - lookback + 1:
            recent_falling.popleft()
        if state == 1:
            recent_falling.append(index)
            if len(recent_falling) >= falling_required:
                armed_until = index + arm_memory

        fallen_consecutive = fallen_consecutive + 1 if state == 2 else 0
        adl_consecutive = adl_consecutive + 1 if state == 0 else 0
        if adl_consecutive >= adl_reset:
            alarm_active = False
        if (
            fallen_consecutive >= fallen_required
            and index <= armed_until
            and not alarm_active
        ):
            triggers.append(row)
            alarm_active = True
    return triggers


def event_evaluation(
    prediction_rows: list[dict],
    manual_rows: list[dict[str, str]],
    original_frame_rows: list[dict[str, str]],
    policy: dict,
):
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in prediction_rows:
        groups[row["sequence_id"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["sample_index"]))

    timestamp_lookup = {
        (row["sequence_id"], int(row["frame_number"])): int(row["timestamp_ms"])
        for row in original_frame_rows
    }
    annotations = {row["sequence_id"]: row for row in manual_rows}
    event_rows = []
    delays_ms = []
    false_alarms = 0
    duplicate_alarms = 0

    for sequence_id in sorted(groups):
        rows = groups[sequence_id]
        triggers = state_machine_triggers(rows, policy)
        if sequence_id.startswith("fall-"):
            annotation = annotations[sequence_id]
            onset_frame = int(annotation["onset_frame"])
            stable_frame = int(annotation["stable_fallen_frame"])
            onset_ms = timestamp_lookup[(sequence_id, onset_frame)]
            stable_ms = timestamp_lookup[(sequence_id, stable_frame)]
            valid = [
                row for row in triggers
                if int(row["sample_timestamp_ms"]) >= onset_ms
            ]
            early = [
                row for row in triggers
                if int(row["sample_timestamp_ms"]) < onset_ms
            ]
            false_alarms += len(early)
            duplicate_alarms += max(0, len(valid) - 1)
            if valid:
                first = valid[0]
                alarm_ms = int(first["sample_timestamp_ms"])
                alarm_frame = int(first["source_frame_number"])
                delay_ms = alarm_ms - onset_ms
                detected = 1
                delays_ms.append(delay_ms)
            else:
                alarm_ms = ""
                alarm_frame = ""
                delay_ms = ""
                detected = 0
            event_rows.append(
                {
                    "sequence_id": sequence_id,
                    "onset_frame": onset_frame,
                    "onset_timestamp_ms": onset_ms,
                    "stable_fallen_frame": stable_frame,
                    "stable_fallen_timestamp_ms": stable_ms,
                    "detected": detected,
                    "first_alarm_source_frame": alarm_frame,
                    "first_alarm_timestamp_ms": alarm_ms,
                    "detection_delay_ms": delay_ms,
                    "early_false_alarms": len(early),
                    "alarms_after_onset": len(valid),
                }
            )
        else:
            false_alarms += len(triggers)

    detected = sum(int(row["detected"]) for row in event_rows)
    negative_windows = sum(
        int(row["target_state_id"]) == 0 for row in prediction_rows
    )
    negative_hours = negative_windows / TARGET_FPS / 3600.0
    summary = {
        "total_fall_events": len(event_rows),
        "detected_fall_events": detected,
        "missed_fall_events": len(event_rows) - detected,
        "event_recall": safe_div(detected, len(event_rows)),
        "false_alarm_count": false_alarms,
        "negative_exposure_hours": negative_hours,
        "false_alarms_per_hour": safe_div(false_alarms, negative_hours),
        "duplicate_alarms_after_onset": duplicate_alarms,
        "mean_detection_delay_ms": (
            float(np.mean(delays_ms)) if delays_ms else None
        ),
        "median_detection_delay_ms": (
            float(np.median(delays_ms)) if delays_ms else None
        ),
        "p95_detection_delay_ms": (
            float(np.percentile(delays_ms, 95)) if delays_ms else None
        ),
    }
    return event_rows, summary


def main() -> None:
    for path in (FRAME_LABELS, MANUAL_EVENTS, POSE_CSV, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    urfd.OUTPUT_DIR = OUTPUT_DIR
    urfd.POSE_CSV = POSE_CSV

    frame_rows = urfd.read_csv(FRAME_LABELS)
    manual_rows = urfd.read_csv(MANUAL_EVENTS)
    pose_rows = urfd.read_existing_pose_rows()
    if len(frame_rows) != 11936 or len(pose_rows) != 11936:
        raise RuntimeError(
            f"Expected 11936 frame/pose rows, found {len(frame_rows)}/{len(pose_rows)}"
        )
    if len(manual_rows) != 30:
        raise RuntimeError(f"Expected 30 manual events, found {len(manual_rows)}")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    if tuple(checkpoint["state_names"]) != STATE_NAMES:
        raise ValueError(f"Unexpected checkpoint states: {checkpoint['state_names']}")
    if list(checkpoint["feature_names"]) != urfd.feature_names():
        raise ValueError("Checkpoint and external feature names differ")
    policy = dict(checkpoint["config"]["state_machine"])

    print(f"Python: {platform.python_version()}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("Protocol: locked source-only three-state zero-shot evaluation", flush=True)
    print(f"Locked state machine: {json.dumps(policy)}", flush=True)
    print("Using cached URFD pose features; no pose extraction", flush=True)

    sampled_rows = urfd.resample_to_20fps(pose_rows)
    x, metadata = urfd.build_windows(sampled_rows)
    probabilities = predict_windows(x, checkpoint)
    if len(probabilities) != len(metadata):
        raise RuntimeError("Prediction and metadata lengths differ")

    prediction_rows = []
    for item, probability in zip(metadata, probabilities):
        event_label = str(item["event_label"])
        if event_label not in STATE_TO_ID:
            raise ValueError(f"Unknown URFD state: {event_label}")
        predicted_id = int(np.argmax(probability))
        row = dict(item)
        row["target_state_id"] = STATE_TO_ID[event_label]
        row["target_state"] = event_label
        row["prob_adl"] = float(probability[0])
        row["prob_falling"] = float(probability[1])
        row["prob_fallen"] = float(probability[2])
        row["predicted_state_id"] = predicted_id
        row["predicted_state"] = STATE_NAMES[predicted_id]
        prediction_rows.append(row)

    y_true = np.asarray(
        [int(row["target_state_id"]) for row in prediction_rows], dtype=int
    )
    y_pred = np.asarray(
        [int(row["predicted_state_id"]) for row in prediction_rows], dtype=int
    )
    metrics = multiclass_metrics(y_true, probabilities)
    metrics["multiclass_brier_score"] = multiclass_brier(y_true, probabilities)
    metrics["top_label_ece_10_bins"] = top_label_ece(y_true, probabilities)

    binary_true = (y_true != 0).astype(int)
    binary_pred = (y_pred != 0).astype(int)
    fall_score = 1.0 - probabilities[:, 0]
    collapsed = binary_metrics(binary_true, binary_pred)
    collapsed_auroc, collapsed_auprc = ranking_metrics(binary_true, fall_score)
    collapsed["auroc"] = collapsed_auroc
    collapsed["auprc"] = collapsed_auprc

    event_rows, event_summary = event_evaluation(
        prediction_rows, manual_rows, frame_rows, policy
    )
    urfd.save_csv(OUTPUT_DIR / "frame_predictions.csv", prediction_rows)
    urfd.save_csv(OUTPUT_DIR / "event_results.csv", event_rows)

    summary = {
        "protocol": {
            "source_training_dataset": "CAUCAFall",
            "external_test_dataset": "URFD Camera 0 RGB",
            "model": "three-state causal Pose-TCN",
            "states": list(STATE_NAMES),
            "fine_tuning": False,
            "target_data_used_for_model_or_policy_selection": False,
            "source_locked_state_machine": policy,
            "source_fps": 20.0,
            "external_original_fps": "approximately 30",
            "external_resampled_fps": TARGET_FPS,
            "window_length": 32,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(DEVICE),
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "three_state_frame_level": metrics,
        "collapsed_fall_vs_adl_frame_level": collapsed,
        "event_level_locked_state_machine": event_summary,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Completed. Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
