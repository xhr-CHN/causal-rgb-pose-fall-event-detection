#!/usr/bin/env python3
"""Build causal multi-horizon recovery features for the V3 alarm verifier.

Inputs are frozen, previously extracted URFD and Le2i pose features plus the
existing V2 alarm feature table. No detector/model inference is performed.
GMDCSA24 is deliberately excluded because it was the final blind test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


ROOT = Path("/home/data/yoloA27")
INPUT_TABLE = (
    ROOT
    / "features/multidataset_alarm_confirmation_features_v2/"
    "alarm_confirmation_features.csv"
)
URFD_POSE = ROOT / "features/urfd_pose_v1/urfd_pose_features.csv"
LE2I_SEQUENCES = ROOT / "features/le2i_frozen_features_v1/sequences"
OUTPUT_DIR = ROOT / "features/multidataset_alarm_recovery_features_v3"
OUTPUT_TABLE = OUTPUT_DIR / "alarm_recovery_features.csv"

HORIZONS_MS = (200.0, 500.0, 1000.0, 1500.0)
HISTORY_MS = 750.0
KEYPOINT_CONFIDENCE_MINIMUM = 0.20
STATIC_SPEED_THRESHOLD = 0.05
EPSILON = 1e-8

KEYPOINTS = (
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
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def valid(value: float) -> bool:
    return math.isfinite(value)


def output_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def mean(values: list[float]) -> float:
    clean = [value for value in values if valid(value)]
    return sum(clean) / len(clean) if clean else math.nan


def maximum(values: list[float]) -> float:
    clean = [value for value in values if valid(value)]
    return max(clean) if clean else math.nan


def minimum(values: list[float]) -> float:
    clean = [value for value in values if valid(value)]
    return min(clean) if clean else math.nan


def slope_per_second(times_ms: list[float], values: list[float]) -> float:
    pairs = [
        (time_ms / 1000.0, value)
        for time_ms, value in zip(times_ms, values)
        if valid(time_ms) and valid(value)
    ]
    if len(pairs) < 2:
        return math.nan
    x_mean = sum(item[0] for item in pairs) / len(pairs)
    y_mean = sum(item[1] for item in pairs) / len(pairs)
    denominator = sum((item[0] - x_mean) ** 2 for item in pairs)
    if denominator <= EPSILON:
        return 0.0
    numerator = sum(
        (item[0] - x_mean) * (item[1] - y_mean) for item in pairs
    )
    return numerator / denominator


def point(row: dict[str, str], name: str) -> Optional[tuple[float, float]]:
    confidence = number(row.get(f"{name}_conf"))
    x = number(row.get(f"{name}_x_global"))
    y = number(row.get(f"{name}_y_global"))
    if (
        not valid(confidence)
        or confidence < KEYPOINT_CONFIDENCE_MINIMUM
        or not valid(x)
        or not valid(y)
    ):
        return None
    return x, y


def midpoint(
    row: dict[str, str], left_name: str, right_name: str
) -> Optional[tuple[float, float]]:
    points = [
        value
        for value in (point(row, left_name), point(row, right_name))
        if value is not None
    ]
    if not points:
        return None
    return (
        sum(value[0] for value in points) / len(points),
        sum(value[1] for value in points) / len(points),
    )


def derived_frame(row: dict[str, str]) -> dict[str, object]:
    width = number(row.get("bbox_w"))
    height = number(row.get("bbox_h"))
    aspect_ratio = (
        width / height if valid(width) and valid(height) and height > EPSILON else math.nan
    )

    shoulders = midpoint(row, "left_shoulder", "right_shoulder")
    hips = midpoint(row, "left_hip", "right_hip")
    if shoulders is not None and hips is not None:
        dx = hips[0] - shoulders[0]
        dy = hips[1] - shoulders[1]
        length = math.hypot(dx, dy)
        torso_horizontalness = abs(dx) / length if length > EPSILON else math.nan
        torso_center_y = (shoulders[1] + hips[1]) / 2.0
    else:
        torso_horizontalness = math.nan
        torso_center_y = math.nan

    keypoints: dict[str, tuple[float, float]] = {}
    for name in KEYPOINTS:
        value = point(row, name)
        if value is not None:
            keypoints[name] = value

    return {
        "timestamp_ms": number(row.get("timestamp_ms")),
        "pose_found": number(row.get("pose_found"), 0.0),
        "person_conf": number(row.get("person_conf")),
        "mean_keypoint_conf": number(row.get("mean_keypoint_conf")),
        "torso_keypoint_conf": number(row.get("torso_keypoint_conf")),
        "visible_keypoint_ratio": number(row.get("visible_keypoint_count")) / 17.0,
        "boundary_touch": 1.0 if number(row.get("boundary_touch_count"), 0.0) > 0 else 0.0,
        "bbox_cy": number(row.get("bbox_cy")),
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": number(row.get("bbox_area")),
        "bbox_aspect_ratio": aspect_ratio,
        "hip_y": hips[1] if hips is not None else math.nan,
        "torso_center_y": torso_center_y,
        "torso_horizontalness": torso_horizontalness,
        "keypoints": keypoints,
    }


def load_pose_file(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        fail(f"Pose feature file not found: {path}")
    rows = read_csv(path)
    output = [derived_frame(row) for row in rows]
    output = [row for row in output if valid(float(row["timestamp_ms"]))]
    output.sort(key=lambda row: float(row["timestamp_ms"]))
    if not output:
        fail(f"No valid pose frames: {path}")
    return output


def nearest_baseline_index(
    frames: list[dict[str, object]], candidate_timestamp_ms: float
) -> int:
    best_index = 0
    for index, row in enumerate(frames):
        if float(row["timestamp_ms"]) <= candidate_timestamp_ms:
            best_index = index
        else:
            break
    return best_index


def signal_summary(
    frames: list[dict[str, object]], signal: str, increasing_with_fall: bool
) -> dict[str, float]:
    times = [float(row["timestamp_ms"]) for row in frames]
    values = [float(row[signal]) for row in frames]
    valid_indices = [index for index, value in enumerate(values) if valid(value)]
    if not valid_indices:
        return {
            "base": math.nan,
            "last": math.nan,
            "delta": math.nan,
            "slope": math.nan,
            "fall_excursion": math.nan,
            "recovery": math.nan,
            "persistence_ratio": math.nan,
        }
    base = values[valid_indices[0]]
    last = values[valid_indices[-1]]
    if increasing_with_fall:
        extreme = maximum(values)
        excursion = extreme - base
        recovery = extreme - last
        persistence = (last - base) / (excursion + EPSILON)
    else:
        extreme = minimum(values)
        excursion = base - extreme
        recovery = last - extreme
        persistence = (base - last) / (excursion + EPSILON)
    return {
        "base": base,
        "last": last,
        "delta": last - base,
        "slope": slope_per_second(times, values),
        "fall_excursion": excursion,
        "recovery": recovery,
        "persistence_ratio": persistence,
    }


def frame_speeds(frames: list[dict[str, object]]) -> list[tuple[float, float]]:
    speeds: list[tuple[float, float]] = []
    for previous, current in zip(frames, frames[1:]):
        delta_seconds = (
            float(current["timestamp_ms"]) - float(previous["timestamp_ms"])
        ) / 1000.0
        if delta_seconds <= EPSILON:
            continue
        previous_points = previous["keypoints"]
        current_points = current["keypoints"]
        common = set(previous_points).intersection(current_points)
        if not common:
            continue
        displacement = mean(
            [
                math.hypot(
                    current_points[name][0] - previous_points[name][0],
                    current_points[name][1] - previous_points[name][1],
                )
                for name in common
            ]
        )
        speeds.append((float(current["timestamp_ms"]), displacement / delta_seconds))
    return speeds


def horizon_features(
    frames: list[dict[str, object]],
    baseline_index: int,
    candidate_timestamp_ms: float,
    horizon_ms: float,
) -> dict[str, float]:
    end_timestamp_ms = candidate_timestamp_ms + horizon_ms
    selected = [
        row
        for row in frames[baseline_index:]
        if float(row["timestamp_ms"]) <= end_timestamp_ms + EPSILON
    ]
    if not selected:
        selected = [frames[baseline_index]]
    tag = f"h{int(horizon_ms)}"
    last_timestamp_ms = float(selected[-1]["timestamp_ms"])
    observed_span_ms = max(0.0, last_timestamp_ms - candidate_timestamp_ms)
    output: dict[str, float] = {
        f"{tag}_observed_frames": float(len(selected)),
        f"{tag}_observed_span_ms": observed_span_ms,
        f"{tag}_coverage_fraction": min(1.0, observed_span_ms / horizon_ms),
        f"{tag}_right_padding_ms": max(0.0, horizon_ms - observed_span_ms),
    }

    signal_directions = {
        "bbox_cy": True,
        "hip_y": True,
        "torso_center_y": True,
        "bbox_aspect_ratio": True,
        "torso_horizontalness": True,
        "bbox_height": False,
    }
    for signal, increasing_with_fall in signal_directions.items():
        summary = signal_summary(selected, signal, increasing_with_fall)
        for statistic, value in summary.items():
            output[f"{tag}_{signal}_{statistic}"] = value

    for signal in (
        "pose_found",
        "person_conf",
        "mean_keypoint_conf",
        "torso_keypoint_conf",
        "visible_keypoint_ratio",
        "boundary_touch",
        "bbox_area",
    ):
        values = [float(row[signal]) for row in selected]
        output[f"{tag}_{signal}_mean"] = mean(values)
        output[f"{tag}_{signal}_last"] = next(
            (value for value in reversed(values) if valid(value)), math.nan
        )

    speeds = frame_speeds(selected)
    speed_values = [item[1] for item in speeds]
    late_start_ms = candidate_timestamp_ms + horizon_ms / 2.0
    early_values = [value for timestamp, value in speeds if timestamp <= late_start_ms]
    late_values = [value for timestamp, value in speeds if timestamp > late_start_ms]
    output[f"{tag}_joint_speed_mean"] = mean(speed_values)
    output[f"{tag}_joint_speed_max"] = maximum(speed_values)
    output[f"{tag}_joint_speed_early_mean"] = mean(early_values)
    output[f"{tag}_joint_speed_late_mean"] = mean(late_values)
    output[f"{tag}_joint_speed_late_to_early_ratio"] = (
        mean(late_values) / (mean(early_values) + EPSILON)
        if valid(mean(late_values)) and valid(mean(early_values))
        else math.nan
    )
    output[f"{tag}_static_transition_fraction"] = (
        sum(value < STATIC_SPEED_THRESHOLD for value in speed_values)
        / len(speed_values)
        if speed_values
        else math.nan
    )
    return output


def history_features(
    frames: list[dict[str, object]], candidate_timestamp_ms: float
) -> dict[str, float]:
    start_ms = candidate_timestamp_ms - HISTORY_MS
    selected = [
        row
        for row in frames
        if start_ms <= float(row["timestamp_ms"]) <= candidate_timestamp_ms
    ]
    if not selected:
        return {
            "history_750_observed_frames": 0.0,
            "history_750_observed_span_ms": 0.0,
            "history_750_coverage_fraction": 0.0,
            "history_750_left_padding_ms": HISTORY_MS,
            "early_candidate_flag": 1.0,
        }
    observed_span_ms = max(
        0.0, candidate_timestamp_ms - float(selected[0]["timestamp_ms"])
    )
    coverage = min(1.0, observed_span_ms / HISTORY_MS)
    return {
        "history_750_observed_frames": float(len(selected)),
        "history_750_observed_span_ms": observed_span_ms,
        "history_750_coverage_fraction": coverage,
        "history_750_left_padding_ms": max(0.0, HISTORY_MS - observed_span_ms),
        "early_candidate_flag": 1.0 if coverage < 0.95 else 0.0,
    }


def main() -> None:
    for path in (INPUT_TABLE, URFD_POSE, LE2I_SEQUENCES):
        if not path.exists():
            fail(f"Required input missing: {path}")
    if OUTPUT_DIR.exists():
        fail(f"Output already exists; refusing overwrite: {OUTPUT_DIR}")

    alarm_rows = read_csv(INPUT_TABLE)
    if not alarm_rows:
        fail("V2 alarm feature table is empty")
    required_columns = {
        "dataset",
        "candidate_key",
        "sequence_id",
        "candidate_timestamp_ms",
        "label_true_alarm",
    }
    missing_columns = required_columns.difference(alarm_rows[0])
    if missing_columns:
        fail(f"Missing V2 columns: {sorted(missing_columns)}")

    dataset_counts = Counter(row["dataset"] for row in alarm_rows)
    if set(dataset_counts) != {"URFD", "Le2i"}:
        fail(f"Unexpected development datasets: {dict(dataset_counts)}")

    required_urfd_sequences = {
        row["sequence_id"] for row in alarm_rows if row["dataset"] == "URFD"
    }
    urfd_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(URFD_POSE):
        sequence_id = row.get("sequence_id", "")
        if sequence_id in required_urfd_sequences:
            urfd_grouped[sequence_id].append(row)

    pose_cache: dict[tuple[str, str], list[dict[str, object]]] = {}

    def pose_frames(dataset: str, sequence_id: str) -> list[dict[str, object]]:
        key = (dataset, sequence_id)
        if key in pose_cache:
            return pose_cache[key]
        if dataset == "Le2i":
            path = LE2I_SEQUENCES / sequence_id / "pose_features.csv"
            frames = load_pose_file(path)
        elif dataset == "URFD":
            raw_rows = urfd_grouped.get(sequence_id, [])
            if not raw_rows:
                fail(f"No URFD pose rows for {sequence_id}")
            frames = [derived_frame(row) for row in raw_rows]
            frames = [
                row for row in frames if valid(float(row["timestamp_ms"]))
            ]
            frames.sort(key=lambda row: float(row["timestamp_ms"]))
        else:
            fail(f"Unsupported dataset: {dataset}")
        pose_cache[key] = frames
        return frames

    output_rows: list[dict[str, object]] = []
    for index, alarm_row in enumerate(alarm_rows, start=1):
        dataset = alarm_row["dataset"]
        sequence_id = alarm_row["sequence_id"]
        candidate_timestamp_ms = number(alarm_row["candidate_timestamp_ms"])
        if not valid(candidate_timestamp_ms):
            fail(f"Invalid candidate timestamp: {alarm_row['candidate_key']}")
        frames = pose_frames(dataset, sequence_id)
        baseline_index = nearest_baseline_index(frames, candidate_timestamp_ms)
        new_features: dict[str, object] = {}
        new_features.update(history_features(frames, candidate_timestamp_ms))
        for horizon_ms in HORIZONS_MS:
            new_features.update(
                horizon_features(
                    frames,
                    baseline_index,
                    candidate_timestamp_ms,
                    horizon_ms,
                )
            )
        output_row: dict[str, object] = dict(alarm_row)
        output_row.update(new_features)
        output_rows.append(output_row)
        if index % 25 == 0 or index == len(alarm_rows):
            print(f"Processed {index}/{len(alarm_rows)} alarms", flush=True)

    if len(output_rows) != len(alarm_rows):
        fail("Output row count differs from V2 input")
    if len({row["candidate_key"] for row in output_rows}) != len(output_rows):
        fail("candidate_key is not unique")

    OUTPUT_DIR.mkdir(parents=True)
    fieldnames = list(output_rows[0].keys())
    with OUTPUT_TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({key: output_value(value) for key, value in row.items()})

    new_feature_names = [name for name in fieldnames if name not in alarm_rows[0]]
    coverage = {}
    for horizon_ms in HORIZONS_MS:
        name = f"h{int(horizon_ms)}_coverage_fraction"
        values = [number(row[name]) for row in output_rows]
        coverage[name] = {
            "mean": mean(values),
            "full_coverage_count": sum(value >= 0.95 for value in values),
        }
    summary = {
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "V3 causal multi-horizon recovery features",
        "source_datasets": dict(dataset_counts),
        "gmdcsa24_used": False,
        "model_inference_performed": False,
        "input_rows": len(alarm_rows),
        "output_rows": len(output_rows),
        "input_columns": len(alarm_rows[0]),
        "new_feature_count": len(new_feature_names),
        "output_columns": len(fieldnames),
        "horizons_ms": list(HORIZONS_MS),
        "history_ms": HISTORY_MS,
        "keypoint_confidence_minimum": KEYPOINT_CONFIDENCE_MINIMUM,
        "static_speed_threshold": STATIC_SPEED_THRESHOLD,
        "coverage": coverage,
        "early_candidate_count": sum(
            number(row["early_candidate_flag"], 0.0) > 0.5
            for row in output_rows
        ),
        "input_sha256": sha256(INPUT_TABLE),
        "urfd_pose_sha256": sha256(URFD_POSE),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "new_feature_names.txt").write_text(
        "\n".join(new_feature_names) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "output_sha256.txt").write_text(
        f"{sha256(OUTPUT_TABLE)}  {OUTPUT_TABLE.name}\n"
        f"{sha256(OUTPUT_DIR / 'summary.json')}  summary.json\n"
        f"{sha256(OUTPUT_DIR / 'new_feature_names.txt')}  new_feature_names.txt\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Completed: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
