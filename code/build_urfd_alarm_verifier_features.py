"""Build causal, alarm-centred URFD V2 features for a second-stage verifier.

Only frames at or before each candidate alarm are used. Short histories are
left-padded with their earliest available prediction so that clip boundaries
cannot be used as a shortcut. Le2i is never read.
"""

from collections import defaultdict
from pathlib import Path
import csv
import json

import numpy as np


ROOT = Path("/home/data/yoloA27")
PREDICTIONS = ROOT / "experiments/urfd_quality_gated_logit_fusion_v3_w16_external/frame_predictions.csv"
ALARMS = ROOT / "experiments/urfd_w16_sequential_evidence_policy_v1/alarm_results.csv"
POSE_FEATURES = ROOT / "features/urfd_pose_v1/urfd_pose_features.csv"
OUTPUT_DIR = ROOT / "features/urfd_alarm_verifier_v2"
HISTORY_LENGTH = 20

SIGNAL_NAMES = [
    "fused_fall_score",
    "prob_falling",
    "prob_fallen",
    "prob_adl",
    "pose_fall_score",
    "rgb_fall_score",
    "pose_gate_weight",
    "branch_disagreement",
    "pose_found_ratio",
    "mean_keypoint_conf",
    "rgb_roi_crop_ratio",
    "person_conf",
    "visible_keypoint_ratio",
    "torso_keypoint_conf",
    "bbox_cy",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_aspect_ratio",
    "bbox_valid",
]
STAT_NAMES = ["last", "mean", "std", "min", "max", "delta", "slope"]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, name, default=0.0):
    try:
        value = row.get(name, "")
        return float(default if value in ("", None) else value)
    except (TypeError, ValueError):
        return float(default)


def integer(row, name, default=0):
    return int(round(number(row, name, default)))


def calculate_stats(values):
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        array = np.zeros(1, dtype=np.float64)
    if len(array) > 1:
        x = np.arange(len(array), dtype=np.float64)
        slope = float(np.polyfit(x, array, 1)[0])
        delta = float(array[-1] - array[0])
    else:
        slope = 0.0
        delta = 0.0
    return {
        "last": float(array[-1]),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "delta": delta,
        "slope": slope,
    }


def extract_signals(prediction, pose_lookup):
    key = (prediction["sequence_id"], integer(prediction, "source_frame_number"))
    pose = pose_lookup.get(key, {})
    bbox_width = number(pose, "bbox_w")
    bbox_height = number(pose, "bbox_h")
    bbox_valid = float(
        number(pose, "pose_found") > 0.5
        and bbox_width > 0.0
        and bbox_height > 0.0
    )
    aspect_ratio = bbox_width / bbox_height if bbox_height > 1e-8 else 0.0
    return {
        "fused_fall_score": number(prediction, "prob_falling") + number(prediction, "prob_fallen"),
        "prob_falling": number(prediction, "prob_falling"),
        "prob_fallen": number(prediction, "prob_fallen"),
        "prob_adl": number(prediction, "prob_adl"),
        "pose_fall_score": number(prediction, "pose_prob_falling") + number(prediction, "pose_prob_fallen"),
        "rgb_fall_score": number(prediction, "rgb_prob_falling") + number(prediction, "rgb_prob_fallen"),
        "pose_gate_weight": number(prediction, "pose_gate_weight"),
        "branch_disagreement": number(prediction, "branch_disagreement"),
        "pose_found_ratio": number(prediction, "pose_found_ratio"),
        "mean_keypoint_conf": number(prediction, "mean_keypoint_conf"),
        "rgb_roi_crop_ratio": number(prediction, "rgb_roi_crop_ratio"),
        "person_conf": number(pose, "person_conf"),
        "visible_keypoint_ratio": number(pose, "visible_keypoint_count") / 17.0,
        "torso_keypoint_conf": number(pose, "torso_keypoint_conf"),
        "bbox_cy": number(pose, "bbox_cy"),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": number(pose, "bbox_area"),
        "bbox_aspect_ratio": aspect_ratio,
        "bbox_valid": bbox_valid,
    }


def main():
    for path in (PREDICTIONS, ALARMS, POSE_FEATURES):
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    prediction_rows = read_csv(PREDICTIONS)
    alarm_rows = read_csv(ALARMS)
    pose_rows = read_csv(POSE_FEATURES)

    predictions_by_sequence = defaultdict(list)
    for row in prediction_rows:
        predictions_by_sequence[row["sequence_id"]].append(row)
    for rows in predictions_by_sequence.values():
        rows.sort(key=lambda row: integer(row, "sample_index"))

    pose_lookup = {
        (row["sequence_id"], integer(row, "frame_number")): row
        for row in pose_rows
    }

    output_rows = []
    skipped = []
    for candidate_id, alarm in enumerate(alarm_rows, start=1):
        sequence_id = alarm["sequence_id"]
        sample_index = integer(alarm, "sample_index")
        sequence_rows = predictions_by_sequence.get(sequence_id, [])
        positions = {
            integer(row, "sample_index"): position
            for position, row in enumerate(sequence_rows)
        }
        if sample_index not in positions:
            skipped.append({
                "sequence_id": sequence_id,
                "sample_index": sample_index,
                "reason": "sample_index_not_found",
            })
            continue

        end = positions[sample_index]
        start = max(0, end - HISTORY_LENGTH + 1)
        observed_history = sequence_rows[start : end + 1]
        observed_count = len(observed_history)
        left_padding = HISTORY_LENGTH - observed_count
        history = [observed_history[0]] * left_padding + observed_history
        signal_history = [extract_signals(row, pose_lookup) for row in history]
        alarm_type = alarm.get("alarm_type", "")

        output = {
            "candidate_id": candidate_id,
            "sequence_id": sequence_id,
            "category": alarm.get("category", ""),
            "sample_index": sample_index,
            "sample_timestamp_ms": number(alarm, "sample_timestamp_ms"),
            "source_frame_number": integer(alarm, "source_frame_number"),
            "alarm_type": alarm_type,
            "label_true_alarm": int(alarm_type == "event_alarm"),
            "observed_history_frames": observed_count,
            "left_padding_frames": left_padding,
            "history_frames_used": len(history),
            "history_coverage": observed_count / HISTORY_LENGTH,
            "trigger_fall_score": number(alarm, "fall_score"),
            "trigger_accumulated_evidence": number(alarm, "accumulated_evidence"),
        }
        for signal_name in SIGNAL_NAMES:
            stats = calculate_stats([row[signal_name] for row in signal_history])
            for stat_name in STAT_NAMES:
                output[f"{signal_name}_{stat_name}"] = stats[stat_name]
        output_rows.append(output)

    if not output_rows:
        raise RuntimeError("No verifier candidates were generated")

    feature_path = OUTPUT_DIR / "alarm_features.csv"
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    true_count = sum(row["label_true_alarm"] for row in output_rows)
    summary = {
        "development_dataset": "URFD Camera 0 RGB",
        "final_blind_dataset": "Le2i",
        "final_blind_dataset_used": False,
        "feature_protocol": "causal history ending at candidate alarm",
        "short_history_boundary_treatment": "left-pad earliest available prediction",
        "future_frames_used": 0,
        "history_length": HISTORY_LENGTH,
        "candidate_count": len(output_rows),
        "true_alarm_candidates": true_count,
        "false_alarm_candidates": len(output_rows) - true_count,
        "sequence_count": len({row["sequence_id"] for row in output_rows}),
        "feature_count_excluding_metadata": len(SIGNAL_NAMES) * len(STAT_NAMES),
        "signal_names": SIGNAL_NAMES,
        "statistics": STAT_NAMES,
        "skipped_alarm_count": len(skipped),
        "left_padded_candidate_count": sum(
            row["left_padding_frames"] > 0 for row in output_rows
        ),
        "output_file": str(feature_path),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if skipped:
        with (OUTPUT_DIR / "skipped_alarms.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(skipped[0]))
            writer.writeheader()
            writer.writerows(skipped)

    print(json.dumps(summary, indent=2))
    print(f"\nCompleted: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
