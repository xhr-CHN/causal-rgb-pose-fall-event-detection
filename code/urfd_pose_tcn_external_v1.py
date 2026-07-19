from __future__ import annotations

import csv
import json
import platform
from collections import defaultdict
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
import ultralytics
from torch.utils.data import DataLoader, TensorDataset
from ultralytics import YOLO

from build_caucafall_pose_windows_v1 import (
    base_feature,
    delta_feature,
    feature_names,
)
from extract_caucafall_pose_features_v1 import (
    missing_pose_values,
    pose_values,
)
from train_pose_mlp_ablation_v1 import (
    binary_metrics,
    ranking_metrics,
    safe_div,
)
from train_pose_tcn_baseline_v1 import PoseTCN


ROOT = Path("/home/data/yoloA27/URFD")
FRAME_LABELS = ROOT / "metadata/rgb_frame_labels.csv"
MANUAL_EVENTS = ROOT / "metadata/manual_event_annotations.csv"
POSE_MODEL_PATH = Path("/home/data/yoloA27/yolo26n-pose.pt")

FULL_CHECKPOINT = Path(
    "/home/data/yoloA27/experiments/pose_tcn_baseline_v1_seed42/"
    "pose_tcn_baseline_best.pt"
)
GEOMETRY_CHECKPOINT = Path(
    "/home/data/yoloA27/experiments/pose_tcn_feature_ablation_v1_seed42/"
    "tcn_geometry_only/tcn_geometry_only_best.pt"
)

FEATURE_DIR = Path("/home/data/yoloA27/features/urfd_pose_v1")
POSE_CSV = FEATURE_DIR / "urfd_pose_features.csv"
OUTPUT_DIR = Path(
    "/home/data/yoloA27/experiments/urfd_pose_tcn_external_v1"
)

IMAGE_SIZE = 640
DEVICE_INDEX = 0
DEVICE = torch.device(
    f"cuda:{DEVICE_INDEX}" if torch.cuda.is_available() else "cpu"
)
POSE_CONFIDENCE_FLOOR = 0.01
NMS_IOU = 0.7
SOURCE_CHUNK_SIZE = 32
POSE_BATCH_SIZE = 8
TCN_BATCH_SIZE = 256
REPORT_EVERY = 320

TARGET_FPS = 20.0
SAMPLE_INTERVAL_MS = 1000.0 / TARGET_FPS
WINDOW_LENGTH = 32
CONSECUTIVE_FRAMES = 5
POSITIVE_LABELS = {"Falling", "Fallen"}

MODEL_CONFIGS = {
    "full_tcn": {
        "checkpoint": FULL_CHECKPOINT,
        "expected_threshold": 0.148,
    },
    "geometry_only_tcn": {
        "checkpoint": GEOMETRY_CHECKPOINT,
        "expected_threshold": 0.339,
    },
}


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


def image_path_from_row(row: dict[str, str]) -> Path:
    category_dir = "falls" if row["category"].lower() == "fall" else "adl"
    sequence_dir = f'{row["sequence_id"]}-cam0-rgb'
    filename = PureWindowsPath(row["image_path"]).name
    return ROOT / category_dir / sequence_dir / filename


def pose_fieldnames() -> list[str]:
    metadata = [
        "sequence_id",
        "category",
        "frame_number",
        "timestamp_ms",
        "timestamp_source",
        "event_label",
        "gt_fall",
        "image_name",
        "image_path",
    ]
    return metadata + list(missing_pose_values().keys())


def pose_key(row: dict[str, str]) -> tuple[str, int]:
    return row["sequence_id"], int(row["frame_number"])


def read_existing_pose_rows() -> list[dict[str, str]]:
    if not POSE_CSV.is_file() or POSE_CSV.stat().st_size == 0:
        return []
    rows = read_csv(POSE_CSV)
    expected = pose_fieldnames()
    if list(rows[0].keys()) != expected:
        raise ValueError("Existing URFD pose CSV has an unexpected schema")
    keys = [pose_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Existing URFD pose CSV contains duplicate frames")
    return rows


def extract_pose_features(frame_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    existing = read_existing_pose_rows()
    processed = {pose_key(row) for row in existing}

    indexed = []
    for metadata in frame_rows:
        path = image_path_from_row(metadata)
        if not path.is_file():
            raise FileNotFoundError(f"Missing URFD frame: {path}")
        indexed.append((metadata, path))
    remaining = [item for item in indexed if pose_key(item[0]) not in processed]

    print(
        f"Pose extraction: total={len(indexed)}, already_done={len(existing)}, "
        f"remaining={len(remaining)}",
        flush=True,
    )
    if remaining:
        model = YOLO(str(POSE_MODEL_PATH))
        write_header = not POSE_CSV.exists() or POSE_CSV.stat().st_size == 0
        with POSE_CSV.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=pose_fieldnames())
            if write_header:
                writer.writeheader()

            completed = 0
            for start in range(0, len(remaining), SOURCE_CHUNK_SIZE):
                chunk = remaining[start : start + SOURCE_CHUNK_SIZE]
                paths = [str(path) for _, path in chunk]
                results = list(
                    model.predict(
                        source=paths,
                        imgsz=IMAGE_SIZE,
                        batch=POSE_BATCH_SIZE,
                        device=DEVICE_INDEX,
                        conf=POSE_CONFIDENCE_FLOOR,
                        iou=NMS_IOU,
                        stream=True,
                        verbose=False,
                    )
                )
                if len(results) != len(chunk):
                    raise RuntimeError(
                        f"Pose inference returned {len(results)} results for "
                        f"{len(chunk)} images"
                    )

                for (metadata, path), result in zip(chunk, results):
                    output: dict[str, object] = {
                        "sequence_id": metadata["sequence_id"],
                        "category": metadata["category"],
                        "frame_number": int(metadata["frame_number"]),
                        "timestamp_ms": int(metadata["timestamp_ms"]),
                        "timestamp_source": metadata["timestamp_source"],
                        "event_label": metadata["event_label"],
                        "gt_fall": int(metadata["event_label"] in POSITIVE_LABELS),
                        "image_name": path.name,
                        "image_path": str(path),
                    }
                    output.update(pose_values(result))
                    writer.writerow(output)

                completed += len(chunk)
                if completed % REPORT_EVERY == 0 or completed == len(remaining):
                    file.flush()
                    print(
                        f"Pose extraction: processed {completed}/{len(remaining)} "
                        "new frames",
                        flush=True,
                    )

    rows = read_csv(POSE_CSV)
    if len(rows) != 11936:
        raise RuntimeError(f"Expected 11936 pose rows, found {len(rows)}")
    keys = [pose_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Final pose CSV contains duplicate frames")
    rows.sort(key=lambda row: (row["sequence_id"], int(row["frame_number"])))
    return rows


def nearest_indices(timestamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    right = np.searchsorted(timestamps, targets, side="left")
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    choose_left = np.abs(targets - timestamps[left]) <= np.abs(
        timestamps[right] - targets
    )
    return np.where(choose_left, left, right)


def resample_to_20fps(pose_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pose_rows:
        groups[row["sequence_id"]].append(row)

    sampled: list[dict[str, object]] = []
    sequence_summary = []
    for sequence_id in sorted(groups):
        rows = sorted(groups[sequence_id], key=lambda row: int(row["timestamp_ms"]))
        timestamps = np.asarray([int(row["timestamp_ms"]) for row in rows])
        targets = np.arange(
            float(timestamps[0]),
            float(timestamps[-1]) + 1e-6,
            SAMPLE_INTERVAL_MS,
        )
        selected_indices = nearest_indices(timestamps, targets)
        if len(set(selected_indices.tolist())) != len(selected_indices):
            raise RuntimeError(f"Resampling duplicated source frames: {sequence_id}")

        for sample_index, (target_ms, source_index) in enumerate(
            zip(targets, selected_indices), start=1
        ):
            source = dict(rows[int(source_index)])
            source["sample_index"] = sample_index
            source["sample_timestamp_ms"] = int(round(float(target_ms)))
            source["source_timestamp_ms"] = int(source["timestamp_ms"])
            source["source_frame_number"] = int(source["frame_number"])
            sampled.append(source)

        sequence_summary.append(
            {
                "sequence_id": sequence_id,
                "category": rows[0]["category"],
                "source_frames": len(rows),
                "resampled_frames": len(targets),
                "start_timestamp_ms": int(timestamps[0]),
                "end_timestamp_ms": int(timestamps[-1]),
            }
        )

    save_csv(OUTPUT_DIR / "urfd_resampling_summary.csv", sequence_summary)
    print(
        f"Resampling: {len(pose_rows)} source frames -> "
        f"{len(sampled)} frames at {TARGET_FPS:.1f} FPS",
        flush=True,
    )
    return sampled


def vectorize_sequence(rows: list[dict[str, object]]) -> np.ndarray:
    vectors = []
    previous = None
    for row in rows:
        vectors.append(base_feature(row) + delta_feature(row, previous))
        previous = row
    result = np.asarray(vectors, dtype=np.float32)
    if result.shape[1] != 103 or not np.isfinite(result).all():
        raise ValueError(f"Invalid feature matrix shape/content: {result.shape}")
    return result


def build_windows(sampled_rows: list[dict[str, object]]):
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in sampled_rows:
        groups[str(row["sequence_id"])].append(row)

    windows = []
    metadata = []
    for sequence_id in sorted(groups):
        rows = sorted(groups[sequence_id], key=lambda row: int(row["sample_index"]))
        features = vectorize_sequence(rows)
        for target_index in range(WINDOW_LENGTH - 1, len(rows)):
            start = target_index - WINDOW_LENGTH + 1
            target = rows[target_index]
            windows.append(features[start : target_index + 1])
            metadata.append(
                {
                    "sequence_id": sequence_id,
                    "category": target["category"],
                    "sample_index": int(target["sample_index"]),
                    "sample_timestamp_ms": int(target["sample_timestamp_ms"]),
                    "source_frame_number": int(target["source_frame_number"]),
                    "source_timestamp_ms": int(target["source_timestamp_ms"]),
                    "event_label": target["event_label"],
                    "gt_fall": int(target["event_label"] in POSITIVE_LABELS),
                    "pose_found_ratio": float(
                        np.mean(
                            [int(row["pose_found"]) for row in rows[start : target_index + 1]]
                        )
                    ),
                    "mean_keypoint_conf": float(
                        np.mean(
                            [
                                float(row["mean_keypoint_conf"])
                                for row in rows[start : target_index + 1]
                            ]
                        )
                    ),
                }
            )

    x = np.stack(windows).astype(np.float32)
    if x.shape[1:] != (32, 103):
        raise RuntimeError(f"Unexpected external window shape: {x.shape}")
    print(f"External windows: {x.shape}", flush=True)
    return x, metadata


def expected_calibration_error(
    y_true: np.ndarray, scores: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(scores[mask])) - float(np.mean(y_true[mask]))
            )
    return float(ece)


def alarm_triggers(rows: list[dict]) -> list[dict]:
    triggers = []
    consecutive = 0
    active = False
    for row in rows:
        if int(row["prediction"]) == 1:
            consecutive += 1
            if consecutive >= CONSECUTIVE_FRAMES and not active:
                triggers.append(row)
                active = True
        else:
            consecutive = 0
            active = False
    return triggers


def event_evaluation(
    prediction_rows: list[dict],
    manual_rows: list[dict[str, str]],
    original_frame_rows: list[dict[str, str]],
):
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in prediction_rows:
        groups[row["sequence_id"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["sample_index"]))

    original_lookup = {
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
        triggers = alarm_triggers(rows)
        if sequence_id.startswith("fall-"):
            annotation = annotations[sequence_id]
            onset_frame = int(annotation["onset_frame"])
            onset_ms = original_lookup[(sequence_id, onset_frame)]
            valid = [
                row for row in triggers if int(row["sample_timestamp_ms"]) >= onset_ms
            ]
            early = [
                row for row in triggers if int(row["sample_timestamp_ms"]) < onset_ms
            ]
            false_alarms += len(early)
            duplicate_alarms += max(0, len(valid) - 1)
            if valid:
                alarm = valid[0]
                alarm_ms = int(alarm["sample_timestamp_ms"])
                delay_ms = alarm_ms - onset_ms
                delays_ms.append(delay_ms)
                detected = 1
                alarm_frame = int(alarm["source_frame_number"])
            else:
                alarm_ms = ""
                delay_ms = ""
                alarm_frame = ""
                detected = 0
            event_rows.append(
                {
                    "sequence_id": sequence_id,
                    "onset_frame": onset_frame,
                    "onset_timestamp_ms": onset_ms,
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

    detected_events = sum(int(row["detected"]) for row in event_rows)
    negative_windows = sum(int(row["gt_fall"]) == 0 for row in prediction_rows)
    negative_hours = negative_windows / TARGET_FPS / 3600.0
    summary = {
        "total_fall_events": len(event_rows),
        "detected_fall_events": detected_events,
        "missed_fall_events": len(event_rows) - detected_events,
        "event_recall": safe_div(detected_events, len(event_rows)),
        "false_alarm_count": false_alarms,
        "negative_exposure_hours": negative_hours,
        "false_alarms_per_hour": safe_div(false_alarms, negative_hours),
        "duplicate_alarms_after_onset": duplicate_alarms,
        "mean_detection_delay_ms": float(np.mean(delays_ms)) if delays_ms else None,
        "median_detection_delay_ms": float(np.median(delays_ms)) if delays_ms else None,
        "p95_detection_delay_ms": (
            float(np.percentile(delays_ms, 95)) if delays_ms else None
        ),
    }
    return event_rows, summary


@torch.no_grad()
def evaluate_checkpoint(
    model_name: str,
    config: dict,
    x: np.ndarray,
    metadata: list[dict],
    manual_rows: list[dict[str, str]],
    original_frame_rows: list[dict[str, str]],
):
    checkpoint = torch.load(config["checkpoint"], map_location="cpu")
    checkpoint_names = checkpoint["feature_names"]
    if list(checkpoint_names) != feature_names():
        raise ValueError(f"Feature names do not match for {model_name}")
    threshold = float(checkpoint["threshold"])
    if abs(threshold - float(config["expected_threshold"])) > 1e-9:
        raise ValueError(
            f"Unexpected locked threshold for {model_name}: {threshold}"
        )

    model_input = x.copy()
    masked_indices = checkpoint.get("masked_feature_indices", [])
    if masked_indices:
        model_input[:, :, masked_indices] = 0.0
    mean = checkpoint["input_mean"].numpy().reshape(1, 1, -1)
    std = checkpoint["input_std"].numpy().reshape(1, 1, -1)
    model_input = (model_input - mean) / std
    if not np.isfinite(model_input).all():
        raise ValueError(f"Non-finite standardized input for {model_name}")

    model = PoseTCN(input_features=103).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(model_input.astype(np.float32))),
        batch_size=TCN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    probabilities = []
    for (batch,) in loader:
        logits = model(batch.to(DEVICE, non_blocking=True))
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    scores = np.concatenate(probabilities)
    if len(scores) != len(metadata):
        raise RuntimeError("Prediction and metadata lengths differ")

    rows = []
    for item, probability in zip(metadata, scores):
        row = dict(item)
        row["probability"] = float(probability)
        row["prediction"] = int(probability >= threshold)
        rows.append(row)

    y_true = np.asarray([int(row["gt_fall"]) for row in rows], dtype=int)
    y_pred = np.asarray([int(row["prediction"]) for row in rows], dtype=int)
    metrics = binary_metrics(y_true, y_pred)
    auroc, auprc = ranking_metrics(y_true, scores)
    metrics.update(
        {
            "auroc": auroc,
            "auprc": auprc,
            "brier_score": float(np.mean((scores - y_true) ** 2)),
            "ece_10_bins": expected_calibration_error(y_true, scores),
        }
    )
    event_rows, event_summary = event_evaluation(
        rows, manual_rows, original_frame_rows
    )

    save_csv(OUTPUT_DIR / f"{model_name}_frame_predictions.csv", rows)
    save_csv(OUTPUT_DIR / f"{model_name}_event_results.csv", event_rows)
    summary = {
        "protocol": {
            "source_training_dataset": "CAUCAFall",
            "external_test_dataset": "URFD Camera 0 RGB",
            "fine_tuning": False,
            "target_data_used_for_threshold_selection": False,
            "source_locked_threshold": threshold,
            "source_locked_consecutive_frames": CONSECUTIVE_FRAMES,
            "source_fps": 20.0,
            "external_original_fps": "approximately 30",
            "external_resampled_fps": TARGET_FPS,
            "window_length": WINDOW_LENGTH,
            "positive_frame_definition": "Falling or Fallen",
            "masked_feature_indices": list(masked_indices),
        },
        "frame_level": metrics,
        "event_level": event_summary,
    }
    (OUTPUT_DIR / f"{model_name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{model_name} completed:", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    required = [
        ROOT,
        FRAME_LABELS,
        MANUAL_EVENTS,
        POSE_MODEL_PATH,
        FULL_CHECKPOINT,
        GEOMETRY_CHECKPOINT,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_rows = read_csv(FRAME_LABELS)
    manual_rows = read_csv(MANUAL_EVENTS)
    if len(frame_rows) != 11936:
        raise RuntimeError(f"Expected 11936 URFD frames, found {len(frame_rows)}")
    if len(manual_rows) != 30:
        raise RuntimeError(f"Expected 30 manual fall events, found {len(manual_rows)}")

    print(f"Python: {platform.python_version()}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Ultralytics: {ultralytics.__version__}", flush=True)
    print(
        f"Device: {DEVICE}; "
        f"GPU: {torch.cuda.get_device_name(DEVICE_INDEX) if torch.cuda.is_available() else None}",
        flush=True,
    )
    print("Protocol: source-only locked zero-shot external evaluation", flush=True)

    pose_rows = extract_pose_features(frame_rows)
    pose_found = sum(int(row["pose_found"]) for row in pose_rows)
    pose_summary = {
        "frames": len(pose_rows),
        "pose_found": pose_found,
        "pose_detection_rate": pose_found / len(pose_rows),
        "mean_visible_keypoints": float(
            np.mean([int(row["visible_keypoint_count"]) for row in pose_rows])
        ),
        "mean_keypoint_conf": float(
            np.mean([float(row["mean_keypoint_conf"]) for row in pose_rows])
        ),
        "boundary_touch_rate": float(
            np.mean([int(row["boundary_touch_count"]) > 0 for row in pose_rows])
        ),
    }
    (OUTPUT_DIR / "urfd_pose_summary.json").write_text(
        json.dumps(pose_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(pose_summary, indent=2, ensure_ascii=False), flush=True)

    sampled_rows = resample_to_20fps(pose_rows)
    x, metadata = build_windows(sampled_rows)

    summaries = {}
    for model_name, config in MODEL_CONFIGS.items():
        summaries[model_name] = evaluate_checkpoint(
            model_name,
            config,
            x,
            metadata,
            manual_rows,
            frame_rows,
        )

    comparison = {
        "pose_extraction": pose_summary,
        "models": summaries,
    }
    (OUTPUT_DIR / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nCompleted. Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
