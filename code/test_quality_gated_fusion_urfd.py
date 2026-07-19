#!/usr/bin/env python3
"""Zero-shot URFD evaluation for the CAUCAFall quality-gated fusion TCN.

No URFD label, probability, or event result is used for model selection,
normalization, threshold tuning, or state-machine tuning.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from train_quality_gated_fusion_tcn import (
    QualityGatedFusionTCN,
    STATE_NAMES,
    binary_average_precision,
    binary_roc_auc,
    json_ready,
    make_confusion_matrix,
    multiclass_metrics,
    write_csv_rows,
)


ROOT = Path("/home/data/yoloA27")
MODEL_PATH = (
    ROOT
    / "experiments/quality_gated_rgb_pose_tcn_v1_seed42/quality_gated_fusion_best.pt"
)
POSE_CSV = ROOT / "features/urfd_pose_v1/urfd_pose_features.csv"
RGB_DIR = ROOT / "features/urfd_rgb_roi_embeddings_v1"
RGB_METADATA = RGB_DIR / "metadata.csv"
RGB_EMBEDDINGS = RGB_DIR / "embeddings.npy"
OUT_DIR = ROOT / "experiments/urfd_quality_gated_fusion_external_v1"

TARGET_FPS = 20.0
SAMPLE_INTERVAL_MS = 1000.0 / TARGET_FPS
BATCH_SIZE = 128
EPSILON = 1e-6

JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def number(row, name, default=0.0):
    value = row.get(name, "")
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        result = float(value)
        return result if np.isfinite(result) else float(default)
    except ValueError:
        return float(default)


def state_id(label):
    normalized = str(label).strip().lower()
    mapping = {"adl": 0, "falling": 1, "fallen": 2}
    if normalized not in mapping:
        raise ValueError(f"Unknown event label: {label}")
    return mapping[normalized]


def validate_and_attach_embeddings(pose_rows, rgb_rows, embeddings):
    if len(rgb_rows) != len(embeddings):
        raise ValueError(
            f"RGB metadata/array mismatch: {len(rgb_rows)} vs {len(embeddings)}"
        )
    rgb_by_key = {}
    for index, row in enumerate(rgb_rows):
        key = (row["sequence_id"], int(float(row["frame_number"])))
        if key in rgb_by_key:
            raise ValueError(f"Duplicate RGB key: {key}")
        rgb_by_key[key] = index

    attached = []
    for row in pose_rows:
        key = (row["sequence_id"], int(float(row["frame_number"])))
        if key not in rgb_by_key:
            raise ValueError(f"Pose frame missing from RGB metadata: {key}")
        rgb_index = rgb_by_key[key]
        rgb_row = rgb_rows[rgb_index]
        if row["image_name"] != rgb_row["image_name"]:
            raise ValueError(f"Image mismatch at {key}")
        item = dict(row)
        item["rgb_index"] = rgb_index
        item["roi_crop_used"] = int(float(rgb_row.get("roi_crop_used", 0)))
        attached.append(item)

    if len(attached) != len(rgb_rows):
        raise ValueError(
            f"Pose/RGB row count mismatch after alignment: {len(attached)} vs {len(rgb_rows)}"
        )
    return attached


def nearest_resample(rows):
    rows = sorted(rows, key=lambda row: number(row, "timestamp_ms"))
    timestamps = np.asarray([number(row, "timestamp_ms") for row in rows])
    final_time = float(timestamps[-1])
    sample_times = np.arange(0.0, final_time + 0.1, SAMPLE_INTERVAL_MS)
    indices = []
    for target in sample_times:
        right = int(np.searchsorted(timestamps, target, side="left"))
        candidates = []
        if right < len(timestamps):
            candidates.append(right)
        if right > 0:
            candidates.append(right - 1)
        chosen = min(candidates, key=lambda value: (abs(timestamps[value] - target), value))
        indices.append(chosen)
    return [(sample_index + 1, float(target), rows[index]) for sample_index, (target, index) in enumerate(zip(sample_times, indices))]


def transform_pose_sequence(resampled, expected_names):
    transformed = []
    previous = None
    for _, _, row in resampled:
        visible_ratio = number(row, "visible_keypoint_count") / 17.0
        bbox_w = number(row, "bbox_w")
        bbox_h = number(row, "bbox_h")
        bbox_cx = number(row, "bbox_cx")
        bbox_cy = number(row, "bbox_cy")

        values = {
            "pose_found": number(row, "pose_found"),
            "person_conf": number(row, "person_conf"),
            "visible_keypoint_ratio": visible_ratio,
            "mean_keypoint_conf": number(row, "mean_keypoint_conf"),
            "torso_keypoint_conf": number(row, "torso_keypoint_conf"),
            "bbox_aspect_ratio": bbox_w / bbox_h if abs(bbox_h) > EPSILON else 0.0,
        }

        for joint in JOINT_NAMES:
            x_name = f"{joint}_x_bbox"
            y_name = f"{joint}_y_bbox"
            confidence_name = f"{joint}_conf"
            x_value = number(row, x_name)
            y_value = number(row, y_name)
            values[x_name] = x_value
            values[y_name] = y_value
            values[confidence_name] = number(row, confidence_name)
            if previous is None:
                values[f"delta_{joint}_x_bbox"] = 0.0
                values[f"delta_{joint}_y_bbox"] = 0.0
            else:
                values[f"delta_{joint}_x_bbox"] = x_value - number(previous, x_name)
                values[f"delta_{joint}_y_bbox"] = y_value - number(previous, y_name)

        if previous is None:
            values["delta_bbox_cx_over_w"] = 0.0
            values["delta_bbox_cy_over_h"] = 0.0
            values["delta_bbox_w_over_w"] = 0.0
            values["delta_bbox_h_over_h"] = 0.0
        else:
            values["delta_bbox_cx_over_w"] = np.clip(
                (bbox_cx - number(previous, "bbox_cx")) / max(abs(bbox_w), EPSILON),
                -2.0,
                2.0,
            )
            values["delta_bbox_cy_over_h"] = np.clip(
                (bbox_cy - number(previous, "bbox_cy")) / max(abs(bbox_h), EPSILON),
                -2.0,
                2.0,
            )
            values["delta_bbox_w_over_w"] = np.clip(
                (bbox_w - number(previous, "bbox_w")) / max(abs(bbox_w), EPSILON),
                -2.0,
                2.0,
            )
            values["delta_bbox_h_over_h"] = np.clip(
                (bbox_h - number(previous, "bbox_h")) / max(abs(bbox_h), EPSILON),
                -2.0,
                2.0,
            )

        missing = [name for name in expected_names if name not in values]
        if missing:
            raise ValueError(f"Cannot construct Pose features: {missing[:10]}")
        transformed.append(np.asarray([values[name] for name in expected_names], dtype=np.float32))
        previous = row
    return np.stack(transformed)


def construct_windows(aligned_rows, embeddings, checkpoint):
    grouped = defaultdict(list)
    for row in aligned_rows:
        grouped[row["sequence_id"]].append(row)

    pose_windows = []
    rgb_windows = []
    quality_windows = []
    targets = []
    metadata = []
    resampling_rows = []
    pose_names = list(checkpoint["pose_feature_names"])
    quality_names = list(checkpoint["quality_feature_names"])
    quality_indices = [pose_names.index(name) for name in quality_names]
    window_length = int(checkpoint["model_config"]["window_length"])

    for sequence_id in sorted(grouped):
        original = grouped[sequence_id]
        resampled = nearest_resample(original)
        pose_features = transform_pose_sequence(resampled, pose_names)
        rgb_features = np.stack(
            [embeddings[int(row["rgb_index"])] for _, _, row in resampled]
        ).astype(np.float32)
        quality_features = np.clip(pose_features[:, quality_indices], 0.0, 1.0)

        resampling_rows.append(
            {
                "sequence_id": sequence_id,
                "category": original[0]["category"],
                "original_frames": len(original),
                "resampled_frames": len(resampled),
                "windows": max(0, len(resampled) - window_length + 1),
            }
        )

        for end in range(window_length - 1, len(resampled)):
            start = end - window_length + 1
            sample_index, sample_timestamp, row = resampled[end]
            pose_windows.append(pose_features[start : end + 1])
            rgb_windows.append(rgb_features[start : end + 1])
            quality_windows.append(quality_features[start : end + 1])
            targets.append(state_id(row["event_label"]))
            metadata.append(
                {
                    "sequence_id": sequence_id,
                    "category": row["category"],
                    "sample_index": sample_index,
                    "sample_timestamp_ms": int(round(sample_timestamp)),
                    "source_frame_number": int(float(row["frame_number"])),
                    "source_timestamp_ms": int(round(number(row, "timestamp_ms"))),
                    "event_label": row["event_label"],
                    "gt_fall": int(float(row["gt_fall"])),
                    "pose_found_ratio": float(quality_features[start : end + 1, 0].mean()),
                    "mean_keypoint_conf": float(quality_features[start : end + 1, 3].mean()),
                    "rgb_roi_crop_ratio": float(
                        np.mean([value[2]["roi_crop_used"] for value in resampled[start : end + 1]])
                    ),
                }
            )

    return (
        np.stack(pose_windows).astype(np.float32),
        np.stack(rgb_windows).astype(np.float32),
        np.stack(quality_windows).astype(np.float32),
        np.asarray(targets, dtype=np.int64),
        metadata,
        resampling_rows,
    )


@torch.no_grad()
def predict(model, pose, rgb, quality, device):
    probabilities = []
    pose_weights = []
    model.eval()
    for start in range(0, len(pose), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(pose))
        output = model(
            torch.from_numpy(pose[start:end]).to(device),
            torch.from_numpy(rgb[start:end]).to(device),
            torch.from_numpy(quality[start:end]).to(device),
        )
        probabilities.append(torch.softmax(output["fused"], 1).cpu().numpy())
        pose_weights.append(output["pose_weight"].mean(1).cpu().numpy())
        if end % 1000 < BATCH_SIZE or end == len(pose):
            print(f"Inference: {end}/{len(pose)}", flush=True)
    return np.concatenate(probabilities), np.concatenate(pose_weights)


def calibration_metrics(labels, probabilities):
    one_hot = np.eye(3)[labels]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    predictions = probabilities.argmax(1)
    confidences = probabilities.max(1)
    correctness = (predictions == labels).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        if index == 9:
            mask = (confidences >= edges[index]) & (confidences <= edges[index + 1])
        else:
            mask = (confidences >= edges[index]) & (confidences < edges[index + 1])
        if mask.any():
            ece += mask.mean() * abs(correctness[mask].mean() - confidences[mask].mean())
    return brier, float(ece)


def collapsed_metrics(labels, probabilities):
    target = (labels != 0).astype(np.int64)
    prediction = (probabilities.argmax(1) != 0).astype(np.int64)
    score = 1.0 - probabilities[:, 0]
    tp = int(((target == 1) & (prediction == 1)).sum())
    fp = int(((target == 0) & (prediction == 1)).sum())
    tn = int(((target == 0) & (prediction == 0)).sum())
    fn = int(((target == 1) & (prediction == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(target),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "auroc": binary_roc_auc(target, score),
        "auprc": binary_average_precision(target, score),
    }


def build_event_boundaries(aligned_rows):
    grouped = defaultdict(list)
    for row in aligned_rows:
        grouped[row["sequence_id"]].append(row)
    boundaries = {}
    for sequence_id, rows in grouped.items():
        if rows[0]["category"] != "fall":
            continue
        rows = sorted(rows, key=lambda row: number(row, "timestamp_ms"))
        onset_candidates = [row for row in rows if row["event_label"] == "Falling"]
        stable_candidates = [row for row in rows if row["event_label"] == "Fallen"]
        onset = onset_candidates[0] if onset_candidates else None
        stable = stable_candidates[0] if stable_candidates else None
        boundaries[sequence_id] = {
            "onset_frame": int(float(onset["frame_number"])) if onset else None,
            "onset_timestamp_ms": int(round(number(onset, "timestamp_ms"))) if onset else None,
            "stable_fallen_frame": int(float(stable["frame_number"])) if stable else None,
            "stable_fallen_timestamp_ms": int(round(number(stable, "timestamp_ms"))) if stable else None,
        }
    return boundaries


def run_state_machine(metadata, predictions, policy, event_boundaries):
    grouped = defaultdict(list)
    for row, prediction in zip(metadata, predictions):
        item = dict(row)
        item["prediction"] = int(prediction)
        grouped[row["sequence_id"]].append(item)

    event_rows = []
    false_alarm_count = 0
    delays = []
    detected_count = 0

    for sequence_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row["sample_index"])
        recent_falling = []
        armed_until = -1
        fallen_streak = 0
        adl_streak = 0
        alarm = None
        for row in rows:
            sample_index = int(row["sample_index"])
            prediction = int(row["prediction"])
            recent_falling = [
                value
                for value in recent_falling
                if sample_index - value < policy["falling_lookback_frames"]
            ]
            if prediction == 1:
                recent_falling.append(sample_index)
            if len(recent_falling) >= policy["falling_required"]:
                armed_until = max(armed_until, sample_index + policy["arm_memory_frames"])
            fallen_streak = fallen_streak + 1 if prediction == 2 else 0
            adl_streak = adl_streak + 1 if prediction == 0 else 0
            if adl_streak >= policy["adl_reset_frames"]:
                recent_falling = []
                armed_until = -1
            if (
                alarm is None
                and sample_index <= armed_until
                and fallen_streak >= policy["fallen_consecutive_frames"]
            ):
                alarm = row

        category = rows[0]["category"]
        if category == "adl":
            if alarm is not None:
                false_alarm_count += 1
            continue

        boundary = event_boundaries[sequence_id]
        onset_timestamp = boundary["onset_timestamp_ms"]
        detected = int(
            alarm is not None
            and onset_timestamp is not None
            and alarm["source_timestamp_ms"] >= onset_timestamp
        )
        delay = (
            alarm["source_timestamp_ms"] - onset_timestamp if detected else None
        )
        if detected:
            detected_count += 1
            delays.append(delay)
        event_rows.append(
            {
                "sequence_id": sequence_id,
                "onset_frame": boundary["onset_frame"],
                "onset_timestamp_ms": boundary["onset_timestamp_ms"],
                "stable_fallen_frame": boundary["stable_fallen_frame"],
                "stable_fallen_timestamp_ms": boundary["stable_fallen_timestamp_ms"],
                "detected": detected,
                "first_alarm_source_frame": alarm["source_frame_number"] if alarm else None,
                "first_alarm_timestamp_ms": alarm["source_timestamp_ms"] if alarm else None,
                "detection_delay_ms": delay,
                "early_false_alarms": 0,
                "alarms_after_onset": detected,
            }
        )

    negative_frames = sum(row["event_label"] == "ADL" for row in metadata)
    negative_hours = negative_frames / TARGET_FPS / 3600.0
    metrics = {
        "total_fall_events": len(event_rows),
        "detected_fall_events": detected_count,
        "missed_fall_events": len(event_rows) - detected_count,
        "event_recall": detected_count / len(event_rows),
        "false_alarm_count": false_alarm_count,
        "negative_exposure_hours": negative_hours,
        "false_alarms_per_hour": false_alarm_count / negative_hours,
        "duplicate_alarms_after_onset": 0,
        "mean_detection_delay_ms": float(np.mean(delays)) if delays else None,
        "median_detection_delay_ms": float(np.median(delays)) if delays else None,
        "p95_detection_delay_ms": float(np.percentile(delays, 95)) if delays else None,
    }
    return metrics, event_rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    config = checkpoint["model_config"]
    model = QualityGatedFusionTCN(
        config["pose_dim"],
        config["rgb_dim"],
        config["quality_dim"],
        config["hidden_dim"],
        config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model: {MODEL_PATH}")
    print(f"GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print("Target data used for tuning: NO", flush=True)

    pose_rows = read_rows(POSE_CSV)
    rgb_rows = read_rows(RGB_METADATA)
    embeddings = np.load(RGB_EMBEDDINGS, mmap_mode="r")
    aligned = validate_and_attach_embeddings(pose_rows, rgb_rows, embeddings)
    pose, rgb, quality, labels, metadata, resampling_rows = construct_windows(
        aligned, embeddings, checkpoint
    )
    print(f"Aligned raw frames: {len(aligned)}")
    print(f"Causal windows: {len(labels)}")

    pose = (pose - np.asarray(checkpoint["pose_mean"])[None, None, :]) / np.asarray(
        checkpoint["pose_std"]
    )[None, None, :]
    rgb = (rgb - np.asarray(checkpoint["rgb_mean"])[None, None, :]) / np.asarray(
        checkpoint["rgb_std"]
    )[None, None, :]
    pose = pose.astype(np.float32)
    rgb = rgb.astype(np.float32)

    probabilities, pose_weights = predict(model, pose, rgb, quality, device)
    predictions = probabilities.argmax(1)
    frame_metrics = multiclass_metrics(labels, probabilities)
    brier, ece = calibration_metrics(labels, probabilities)
    frame_metrics["multiclass_brier_score"] = brier
    frame_metrics["top_label_ece_10_bins"] = ece
    binary_metrics = collapsed_metrics(labels, probabilities)
    event_metrics, event_rows = run_state_machine(
        metadata,
        predictions,
        checkpoint["state_machine"],
        build_event_boundaries(aligned),
    )

    prediction_rows = []
    for index, source in enumerate(metadata):
        row = dict(source)
        row.update(
            {
                "target_state_id": int(labels[index]),
                "target_state": STATE_NAMES[labels[index]],
                "prob_adl": float(probabilities[index, 0]),
                "prob_falling": float(probabilities[index, 1]),
                "prob_fallen": float(probabilities[index, 2]),
                "predicted_state_id": int(predictions[index]),
                "predicted_state": STATE_NAMES[predictions[index]],
                "mean_pose_gate_weight": float(pose_weights[index]),
            }
        )
        prediction_rows.append(row)
    write_csv_rows(OUT_DIR / "frame_predictions.csv", prediction_rows)
    write_csv_rows(OUT_DIR / "event_results.csv", event_rows)
    write_csv_rows(OUT_DIR / "urfd_resampling_summary.csv", resampling_rows)

    gate_stats = {}
    for class_id, name in enumerate(STATE_NAMES):
        values = pose_weights[labels == class_id]
        gate_stats[name] = {
            "count": len(values),
            "mean_pose_weight": float(values.mean()),
            "std_pose_weight": float(values.std()),
            "minimum_pose_weight": float(values.min()),
            "maximum_pose_weight": float(values.max()),
        }

    summary = {
        "protocol": {
            "source_training_dataset": "CAUCAFall",
            "external_test_dataset": "URFD Camera 0 RGB",
            "model": "quality-gated dual-stream RGB/Pose causal TCN",
            "fine_tuning": False,
            "target_data_used_for_model_or_policy_selection": False,
            "source_locked_state_machine": checkpoint["state_machine"],
            "external_resampled_fps": TARGET_FPS,
            "window_length": config["window_length"],
            "pose_features": config["pose_dim"],
            "rgb_features": config["rgb_dim"],
        },
        "environment": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "three_state_frame_level": frame_metrics,
        "collapsed_fall_vs_adl_frame_level": binary_metrics,
        "event_level_locked_state_machine": event_metrics,
        "gate_statistics": gate_stats,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(json_ready(summary), file, ensure_ascii=False, indent=2)

    print("\nExternal test completed.")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
