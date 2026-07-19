#!/usr/bin/env python3
"""Post-hoc frozen fusion ablation on URFD and Le2i.

This script compares four inference variants derived from the SAME trained
quality-gated W16 checkpoint outputs:

1. Pose branch only
2. RGB branch only
3. Fixed 0.5/0.5 late-logit fusion
4. Learned quality-gated late-logit fusion

The comparison reads cached frame_predictions.csv files. It does not retrain a
model, extract features, or tune a threshold on either external dataset. The
source-selected sequential evidence policy is applied unchanged to every
variant. Consequently, the event results isolate the effect of fusion at the
candidate-generation stage; they do not include the later alarm verifier or
causal-rescue rule.

Important reporting note:
    The Pose-only and RGB-only variants below are the auxiliary branches of the
    jointly trained quality-gated checkpoint. They are not the separately
    trained standalone Pose-TCN and RGB-TCN checkpoints. This is intentional:
    holding the branch parameters fixed makes the fixed-vs-gated comparison a
    controlled fusion ablation.

Default server command:
    python external_fusion_ablation_urfd_le2i_v1.py

Run only one dataset:
    python external_fusion_ablation_urfd_le2i_v1.py --dataset urfd
    python external_fusion_ablation_urfd_le2i_v1.py --dataset le2i

The script requires only Python's standard library and NumPy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_ROOT = Path("/home/data/yoloA27")
DEFAULT_OUTPUT_NAME = "external_fusion_ablation_urfd_le2i_v1"
FPS = 20.0
EPSILON = 1e-12

VARIANT_ORDER = (
    "pose_branch_only",
    "rgb_branch_only",
    "fixed_0.5_late_logit",
    "quality_gated_late_logit",
)

PROBABILITY_COLUMNS = {
    "pose": ("pose_prob_adl", "pose_prob_falling", "pose_prob_fallen"),
    "rgb": ("rgb_prob_adl", "rgb_prob_falling", "rgb_prob_fallen"),
    "gated": ("prob_adl", "prob_falling", "prob_fallen"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Pose, RGB, fixed late-logit, and quality-gated fusion."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Project root (default: /home/data/yoloA27).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<root>/experiments/external_fusion_ablation_urfd_le2i_v1"
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "urfd", "le2i"),
        default="all",
        help="Dataset(s) to evaluate (default: all).",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, indent=2, ensure_ascii=False)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(row: Mapping[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name)
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def integer(row: Mapping[str, str], name: str, default: int = 0) -> int:
    return int(round(number(row, name, default)))


def divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required file(s) missing:\n  " + "\n  ".join(missing))


def require_columns(rows: Sequence[Mapping[str, str]], columns: Iterable[str], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} is empty")
    available = set(rows[0])
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(
            f"{label} lacks required columns: {missing}\n"
            f"Available columns: {sorted(available)}"
        )


def normalize_probabilities(values: np.ndarray, label: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(f"{label} must have shape [N, 3], found {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{label} contains non-finite values")
    if np.any(probabilities < -1e-8):
        raise ValueError(f"{label} contains negative probabilities")
    probabilities = np.clip(probabilities, 0.0, None)
    sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(sums <= EPSILON):
        raise ValueError(f"{label} contains a row with zero probability mass")
    return probabilities / sums


def probability_matrix(rows: Sequence[Mapping[str, str]], kind: str) -> np.ndarray:
    columns = PROBABILITY_COLUMNS[kind]
    require_columns(rows, columns, f"{kind} probabilities")
    values = np.asarray(
        [[number(row, column, float("nan")) for column in columns] for row in rows],
        dtype=np.float64,
    )
    return normalize_probabilities(values, kind)


def fixed_equal_logit_fusion(pose: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Return softmax(0.5 * pose_logits + 0.5 * rgb_logits).

    The original logits are recoverable up to an additive sample-wise constant
    as log(softmax(logits)). Additive constants cancel under the final softmax,
    so this computation is exactly equal to averaging the two stored branch
    logits, apart from numerical clipping near zero.
    """

    pose_log = np.log(np.clip(pose, EPSILON, 1.0))
    rgb_log = np.log(np.clip(rgb, EPSILON, 1.0))
    logits = 0.5 * (pose_log + rgb_log)
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def build_variants(rows: Sequence[Mapping[str, str]]) -> Dict[str, np.ndarray]:
    pose = probability_matrix(rows, "pose")
    rgb = probability_matrix(rows, "rgb")
    gated = probability_matrix(rows, "gated")
    return {
        "pose_branch_only": pose,
        "rgb_branch_only": rgb,
        "fixed_0.5_late_logit": fixed_equal_logit_fusion(pose, rgb),
        "quality_gated_late_logit": gated,
    }


def tied_average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def binary_auroc(target: np.ndarray, score: np.ndarray) -> Optional[float]:
    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = tied_average_ranks(score)
    positive_rank_sum = float(ranks[target == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def average_precision(target: np.ndarray, score: np.ndarray) -> Optional[float]:
    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positives = int(target.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order]
    sorted_score = score[order]
    true_positives = np.cumsum(sorted_target)
    false_positives = np.cumsum(1 - sorted_target)

    # Evaluate only after every group of tied scores. This makes AP invariant
    # to row ordering inside a tie and matches the usual step-wise PR integral.
    threshold_ends = np.r_[
        np.where(np.diff(sorted_score) != 0)[0], len(sorted_score) - 1
    ]
    tp = true_positives[threshold_ends].astype(np.float64)
    fp = false_positives[threshold_ends].astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / positives
    recall_increase = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increase * precision))


def expected_calibration_error(
    target: np.ndarray, score: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (score >= edges[index]) & (score <= edges[index + 1])
        else:
            mask = (score >= edges[index]) & (score < edges[index + 1])
        if mask.any():
            result += float(mask.mean()) * abs(
                float(target[mask].mean()) - float(score[mask].mean())
            )
    return float(result)


def binary_frame_metrics(target: np.ndarray, probabilities: np.ndarray) -> Dict:
    target = np.asarray(target, dtype=np.int64)
    if len(target) != len(probabilities):
        raise ValueError("Target/probability length mismatch")
    prediction = (np.argmax(probabilities, axis=1) != 0).astype(np.int64)
    score = np.clip(probabilities[:, 1] + probabilities[:, 2], 0.0, 1.0)

    tp = int(np.sum((target == 1) & (prediction == 1)))
    fp = int(np.sum((target == 0) & (prediction == 1)))
    tn = int(np.sum((target == 0) & (prediction == 0)))
    fn = int(np.sum((target == 1) & (prediction == 0)))

    positive_precision = divide(tp, tp + fp)
    positive_recall = divide(tp, tp + fn)
    positive_f1 = divide(
        2.0 * positive_precision * positive_recall,
        positive_precision + positive_recall,
    )
    negative_precision = divide(tn, tn + fn)
    negative_recall = divide(tn, tn + fp)
    negative_f1 = divide(
        2.0 * negative_precision * negative_recall,
        negative_precision + negative_recall,
    )

    positive_ap = average_precision(target, score)
    negative_ap = average_precision(1 - target, 1.0 - score)
    macro_ap = (
        None
        if positive_ap is None or negative_ap is None
        else 0.5 * (positive_ap + negative_ap)
    )

    return {
        "evaluation_unit": "20-FPS causal-window endpoint",
        "positive_definition": "Falling or Fallen / official fall interval",
        "samples": int(len(target)),
        "positive_samples": int(target.sum()),
        "negative_samples": int(len(target) - target.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": divide(tp + tn, len(target)),
        "positive_precision": positive_precision,
        "positive_recall_sensitivity": positive_recall,
        "specificity": negative_recall,
        "positive_f1": positive_f1,
        "negative_f1": negative_f1,
        "binary_macro_f1": 0.5 * (positive_f1 + negative_f1),
        "balanced_accuracy": 0.5 * (positive_recall + negative_recall),
        "auroc": binary_auroc(target, score),
        "positive_auprc_average_precision": positive_ap,
        "negative_auprc_average_precision": negative_ap,
        "binary_macro_auprc": macro_ap,
        "brier_score": float(np.mean((score - target) ** 2)),
        "ece_10_bins": expected_calibration_error(target, score, bins=10),
    }


def load_policy(path: Path) -> Dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    policy = document.get("selected_policy", document)
    required = {
        "alpha",
        "alarm_threshold",
        "consecutive_evidence_frames",
        "reset_threshold",
        "reset_frames",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"Policy lacks required fields: {missing}")
    return dict(policy)


def row_sort_key(row: Mapping[str, str]) -> Tuple[float, float, float]:
    return (
        number(row, "sample_index", float("inf")),
        number(row, "source_timestamp_ms", float("inf")),
        number(row, "source_frame_number", float("inf")),
    )


def source_timestamp_ms(row: Mapping[str, str], fps: float = FPS) -> float:
    if row.get("source_timestamp_ms") not in (None, ""):
        return number(row, "source_timestamp_ms")
    if row.get("sample_timestamp_ms") not in (None, ""):
        return number(row, "sample_timestamp_ms")
    frame = integer(row, "source_frame_number")
    return 1000.0 * max(0, frame - 1) / fps


def generate_candidate_alarms(
    rows: Sequence[Mapping[str, str]],
    probabilities: np.ndarray,
    policy: Mapping,
    sequence_metadata: Mapping[str, Mapping],
) -> List[Dict]:
    if len(rows) != len(probabilities):
        raise ValueError("Prediction row/probability length mismatch")

    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        sequence_id = str(row["sequence_id"])
        grouped[sequence_id].append(index)

    alpha = float(policy["alpha"])
    threshold = float(policy["alarm_threshold"])
    required = int(policy["consecutive_evidence_frames"])
    reset_threshold = float(policy["reset_threshold"])
    reset_frames = int(policy["reset_frames"])

    alarms: List[Dict] = []
    candidate_id = 0
    for sequence_id in sorted(grouped):
        indices = sorted(grouped[sequence_id], key=lambda index: row_sort_key(rows[index]))
        evidence = 0.0
        above_count = 0
        below_count = 0
        latched = False

        for index in indices:
            row = rows[index]
            fall_score = float(
                np.clip(probabilities[index, 1] + probabilities[index, 2], 0.0, 1.0)
            )
            evidence = alpha * evidence + (1.0 - alpha) * fall_score

            if not latched:
                above_count = above_count + 1 if evidence >= threshold else 0
                if above_count >= required:
                    candidate_id += 1
                    metadata = sequence_metadata[sequence_id]
                    alarms.append(
                        {
                            "candidate_id": candidate_id,
                            "sequence_id": sequence_id,
                            "scene_or_category": metadata.get(
                                "scene", metadata.get("category", "")
                            ),
                            "sample_index": integer(row, "sample_index"),
                            "sample_timestamp_ms": number(
                                row, "sample_timestamp_ms", source_timestamp_ms(row)
                            ),
                            "source_frame_number": integer(
                                row, "source_frame_number"
                            ),
                            "source_timestamp_ms": source_timestamp_ms(row),
                            "prob_adl": float(probabilities[index, 0]),
                            "prob_falling": float(probabilities[index, 1]),
                            "prob_fallen": float(probabilities[index, 2]),
                            "fall_score": fall_score,
                            "accumulated_evidence": float(evidence),
                        }
                    )
                    latched = True
                    above_count = 0
                    below_count = 0
            else:
                below_count = below_count + 1 if evidence <= reset_threshold else 0
                if below_count >= reset_frames:
                    latched = False
                    below_count = 0
    return alarms


def evaluate_events(
    alarms: Sequence[Mapping],
    sequence_metadata: Mapping[str, Mapping],
    event_metadata: Mapping[str, Mapping],
    negative_exposure_hours: float,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    grouped: Dict[str, List[Mapping]] = defaultdict(list)
    for alarm in alarms:
        grouped[str(alarm["sequence_id"])].append(alarm)
    for sequence_alarms in grouped.values():
        sequence_alarms.sort(
            key=lambda row: (
                number(row, "source_timestamp_ms"),
                integer(row, "candidate_id"),
            )
        )

    classified_alarms: List[Dict] = []
    event_rows: List[Dict] = []
    delays_ms: List[float] = []
    detected_events = 0
    false_alarm_count = 0
    duplicate_count = 0

    for sequence_id, metadata in sequence_metadata.items():
        sequence_alarms = grouped.get(sequence_id, [])
        is_fall = sequence_id in event_metadata
        if not is_fall:
            false_alarm_count += len(sequence_alarms)
            for alarm in sequence_alarms:
                output = dict(alarm)
                output["alarm_classification"] = "false_alarm_adl"
                classified_alarms.append(output)
            continue

        event = event_metadata[sequence_id]
        onset_ms = float(event["onset_ms"])
        early = [
            alarm
            for alarm in sequence_alarms
            if number(alarm, "source_timestamp_ms") < onset_ms
        ]
        after_onset = [
            alarm
            for alarm in sequence_alarms
            if number(alarm, "source_timestamp_ms") >= onset_ms
        ]
        false_alarm_count += len(early)
        for alarm in early:
            output = dict(alarm)
            output["alarm_classification"] = "false_alarm_early"
            classified_alarms.append(output)
        for alarm_index, alarm in enumerate(after_onset):
            output = dict(alarm)
            output["alarm_classification"] = (
                "event_alarm" if alarm_index == 0 else "duplicate_event_alarm"
            )
            classified_alarms.append(output)

        detected = bool(after_onset)
        first_frame = None
        first_timestamp = None
        delay_ms = None
        duplicates = max(0, len(after_onset) - 1)
        duplicate_count += duplicates
        if detected:
            detected_events += 1
            first = after_onset[0]
            first_frame = integer(first, "source_frame_number")
            first_timestamp = number(first, "source_timestamp_ms")
            delay_ms = first_timestamp - onset_ms
            if delay_ms < -1e-6:
                raise RuntimeError(f"Negative delay for {sequence_id}: {delay_ms}")
            delays_ms.append(delay_ms)

        event_rows.append(
            {
                "sequence_id": sequence_id,
                "scene_or_category": metadata.get(
                    "scene", metadata.get("category", "")
                ),
                "onset_frame": event.get("onset_frame"),
                "onset_ms": onset_ms,
                "detected": int(detected),
                "first_alarm_source_frame": first_frame,
                "first_alarm_timestamp_ms": first_timestamp,
                "detection_delay_ms": delay_ms,
                "early_false_alarms": len(early),
                "alarms_after_onset": len(after_onset),
                "duplicate_alarms": duplicates,
            }
        )

    total_events = len(event_metadata)
    metrics = {
        "candidate_stage_only": True,
        "total_fall_events": total_events,
        "detected_fall_events": detected_events,
        "missed_fall_events": total_events - detected_events,
        "event_recall": divide(detected_events, total_events),
        "candidate_alarm_count": len(alarms),
        "false_alarm_count": false_alarm_count,
        "negative_exposure_hours": float(negative_exposure_hours),
        "false_alarms_per_hour": divide(
            false_alarm_count, negative_exposure_hours
        ),
        "duplicate_alarms_after_onset": duplicate_count,
        "mean_detection_delay_ms": (
            float(np.mean(delays_ms)) if delays_ms else None
        ),
        "median_detection_delay_ms": (
            float(np.median(delays_ms)) if delays_ms else None
        ),
        "p95_detection_delay_ms": (
            float(np.percentile(delays_ms, 95)) if delays_ms else None
        ),
        "mean_detection_delay_s": (
            float(np.mean(delays_ms) / 1000.0) if delays_ms else None
        ),
        "median_detection_delay_s": (
            float(np.median(delays_ms) / 1000.0) if delays_ms else None
        ),
        "p95_detection_delay_s": (
            float(np.percentile(delays_ms, 95) / 1000.0) if delays_ms else None
        ),
    }
    return metrics, classified_alarms, event_rows


def urfd_targets(rows: Sequence[Mapping[str, str]]) -> np.ndarray:
    available = set(rows[0])
    if "target_state_id" in available:
        return np.asarray(
            [int(integer(row, "target_state_id") != 0) for row in rows],
            dtype=np.int64,
        )
    if "gt_fall" in available:
        return np.asarray(
            [int(integer(row, "gt_fall") != 0) for row in rows], dtype=np.int64
        )
    if "event_label" in available:
        return np.asarray(
            [int(str(row["event_label"]).strip().lower() != "adl") for row in rows],
            dtype=np.int64,
        )
    raise ValueError(
        "URFD predictions need one of: target_state_id, gt_fall, event_label"
    )


def load_urfd_ground_truth(
    rows: Sequence[Mapping[str, str]], event_path: Path
) -> Tuple[Dict[str, Dict], Dict[str, Dict], np.ndarray, float]:
    event_rows = read_csv(event_path)
    if not event_rows:
        raise ValueError(f"URFD event file is empty: {event_path}")
    require_columns(event_rows, ("sequence_id", "onset_timestamp_ms"), "URFD events")

    events: Dict[str, Dict] = {}
    for row in event_rows:
        sequence_id = str(row["sequence_id"])
        if sequence_id in events:
            raise ValueError(f"Duplicate URFD event: {sequence_id}")
        events[sequence_id] = {
            "sequence_id": sequence_id,
            "onset_frame": integer(row, "onset_frame", integer(row, "onset_source_frame")),
            "onset_ms": number(row, "onset_timestamp_ms"),
        }

    sequences: Dict[str, Dict] = {}
    for row in rows:
        sequence_id = str(row["sequence_id"])
        if sequence_id not in sequences:
            sequences[sequence_id] = {
                "sequence_id": sequence_id,
                "category": row.get(
                    "category", "fall" if sequence_id in events else "adl"
                ),
                "is_fall": sequence_id in events,
            }
    missing = sorted(set(events) - set(sequences))
    if missing:
        raise ValueError(f"URFD events missing from predictions: {missing[:10]}")

    target = urfd_targets(rows)
    negative_exposure_hours = float(np.sum(target == 0) / FPS / 3600.0)
    return sequences, events, target, negative_exposure_hours


def load_le2i_ground_truth(
    rows: Sequence[Mapping[str, str]], inventory_path: Path, event_path: Path
) -> Tuple[Dict[str, Dict], Dict[str, Dict], np.ndarray, float]:
    inventory_rows = read_csv(inventory_path)
    event_rows = read_csv(event_path)
    require_columns(
        inventory_rows,
        ("sequence_id", "sequence_type", "negative_exposure_seconds"),
        "Le2i inventory",
    )
    require_columns(
        event_rows,
        ("sequence_id", "onset_frame", "end_frame", "onset_ms"),
        "Le2i events",
    )

    sequences: Dict[str, Dict] = {}
    for row in inventory_rows:
        sequence_id = str(row["sequence_id"])
        if sequence_id in sequences:
            raise ValueError(f"Duplicate Le2i sequence: {sequence_id}")
        sequence_type = str(row["sequence_type"])
        sequences[sequence_id] = {
            "sequence_id": sequence_id,
            "scene": row.get("scene", ""),
            "sequence_type": sequence_type,
            "is_fall": sequence_type == "fall",
            "negative_exposure_seconds": number(
                row, "negative_exposure_seconds"
            ),
        }

    events: Dict[str, Dict] = {}
    for row in event_rows:
        sequence_id = str(row["sequence_id"])
        if sequence_id in events:
            raise ValueError(f"Duplicate Le2i event: {sequence_id}")
        events[sequence_id] = {
            "sequence_id": sequence_id,
            "onset_frame": integer(row, "onset_frame"),
            "end_frame": integer(row, "end_frame"),
            "onset_ms": number(row, "onset_ms"),
            "end_ms": number(row, "end_ms"),
        }

    expected_events = {
        sequence_id
        for sequence_id, metadata in sequences.items()
        if metadata["is_fall"]
    }
    if set(events) != expected_events:
        raise ValueError("Le2i fall-event and inventory sequence sets differ")

    prediction_sequences = {str(row["sequence_id"]) for row in rows}
    if prediction_sequences != set(sequences):
        missing = sorted(set(sequences) - prediction_sequences)
        extra = sorted(prediction_sequences - set(sequences))
        raise ValueError(
            "Le2i prediction/inventory sequence sets differ; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    target = []
    for row in rows:
        sequence_id = str(row["sequence_id"])
        frame = integer(row, "source_frame_number")
        is_positive = 0
        if sequence_id in events:
            event = events[sequence_id]
            is_positive = int(event["onset_frame"] <= frame <= event["end_frame"])
        target.append(is_positive)

    negative_exposure_hours = sum(
        metadata["negative_exposure_seconds"] for metadata in sequences.values()
    ) / 3600.0
    return (
        sequences,
        events,
        np.asarray(target, dtype=np.int64),
        float(negative_exposure_hours),
    )


def evaluate_dataset(
    dataset: str,
    prediction_path: Path,
    policy_path: Path,
    output_dir: Path,
    inventory_path: Optional[Path] = None,
    event_path: Optional[Path] = None,
) -> Tuple[List[Dict], Dict]:
    required = [prediction_path, policy_path]
    if inventory_path is not None:
        required.append(inventory_path)
    if event_path is not None:
        required.append(event_path)
    require_files(required)

    rows = read_csv(prediction_path)
    require_columns(
        rows,
        (
            "sequence_id",
            "source_frame_number",
            "prob_adl",
            "prob_falling",
            "prob_fallen",
            "pose_prob_adl",
            "pose_prob_falling",
            "pose_prob_fallen",
            "rgb_prob_adl",
            "rgb_prob_falling",
            "rgb_prob_fallen",
        ),
        f"{dataset} frame predictions",
    )
    variants = build_variants(rows)
    policy = load_policy(policy_path)

    if dataset == "URFD":
        if event_path is None:
            raise ValueError("URFD event path is required")
        sequences, events, target, negative_hours = load_urfd_ground_truth(
            rows, event_path
        )
    elif dataset == "Le2i":
        if inventory_path is None or event_path is None:
            raise ValueError("Le2i inventory and event paths are required")
        sequences, events, target, negative_hours = load_le2i_ground_truth(
            rows, inventory_path, event_path
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dataset_dir = output_dir / dataset.lower()
    comparison_rows: List[Dict] = []
    summaries: Dict[str, Dict] = {}
    for variant in VARIANT_ORDER:
        probabilities = variants[variant]
        frame_metrics = binary_frame_metrics(target, probabilities)
        alarms = generate_candidate_alarms(rows, probabilities, policy, sequences)
        event_metrics, classified_alarms, event_rows = evaluate_events(
            alarms, sequences, events, negative_hours
        )

        variant_dir = dataset_dir / variant
        alarm_fields = (
            "candidate_id",
            "sequence_id",
            "scene_or_category",
            "sample_index",
            "sample_timestamp_ms",
            "source_frame_number",
            "source_timestamp_ms",
            "prob_adl",
            "prob_falling",
            "prob_fallen",
            "fall_score",
            "accumulated_evidence",
            "alarm_classification",
        )
        event_fields = (
            "sequence_id",
            "scene_or_category",
            "onset_frame",
            "onset_ms",
            "detected",
            "first_alarm_source_frame",
            "first_alarm_timestamp_ms",
            "detection_delay_ms",
            "early_false_alarms",
            "alarms_after_onset",
            "duplicate_alarms",
        )
        write_csv(variant_dir / "candidate_alarms.csv", classified_alarms, alarm_fields)
        write_csv(variant_dir / "event_results.csv", event_rows, event_fields)

        summary = {
            "dataset": dataset,
            "variant": variant,
            "variant_definition": {
                "pose_branch_only": "Pose auxiliary branch of the jointly trained gated checkpoint",
                "rgb_branch_only": "RGB auxiliary branch of the jointly trained gated checkpoint",
                "fixed_0.5_late_logit": "softmax(0.5*pose_logits + 0.5*rgb_logits)",
                "quality_gated_late_logit": "stored learned quality-gated fused probabilities",
            }[variant],
            "protocol": {
                "post_hoc_frozen_external_ablation": True,
                "retraining": False,
                "external_threshold_tuning": False,
                "candidate_stage_only": True,
                "verifier_or_rescue_applied": False,
                "window_endpoint_fps": FPS,
                "locked_source_policy": policy,
            },
            "binary_frame_level": frame_metrics,
            "event_level_locked_candidate_policy": event_metrics,
        }
        write_json(variant_dir / "summary.json", summary)
        summaries[variant] = summary

        comparison_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "windows": frame_metrics["samples"],
                "binary_macro_f1": frame_metrics["binary_macro_f1"],
                "binary_macro_auprc": frame_metrics["binary_macro_auprc"],
                "positive_f1": frame_metrics["positive_f1"],
                "positive_auprc": frame_metrics[
                    "positive_auprc_average_precision"
                ],
                "balanced_accuracy": frame_metrics["balanced_accuracy"],
                "event_recall": event_metrics["event_recall"],
                "detected_events": event_metrics["detected_fall_events"],
                "total_events": event_metrics["total_fall_events"],
                "false_alarm_count": event_metrics["false_alarm_count"],
                "negative_exposure_hours": event_metrics[
                    "negative_exposure_hours"
                ],
                "false_alarms_per_hour": event_metrics[
                    "false_alarms_per_hour"
                ],
                "median_detection_delay_s": event_metrics[
                    "median_detection_delay_s"
                ],
                "duplicate_alarms": event_metrics[
                    "duplicate_alarms_after_onset"
                ],
            }
        )

    dataset_summary = {
        "dataset": dataset,
        "input_frame_predictions": str(prediction_path),
        "input_frame_predictions_sha256": sha256(prediction_path),
        "locked_policy_file": str(policy_path),
        "locked_policy_sha256": sha256(policy_path),
        "variants": summaries,
    }
    write_json(dataset_dir / "dataset_summary.json", dataset_summary)
    return comparison_rows, dataset_summary


def print_comparison(rows: Sequence[Mapping]) -> None:
    columns = (
        "dataset",
        "variant",
        "binary_macro_f1",
        "binary_macro_auprc",
        "event_recall",
        "false_alarms_per_hour",
        "median_detection_delay_s",
    )
    widths = {column: len(column) for column in columns}
    formatted: List[Dict[str, str]] = []
    for row in rows:
        output: Dict[str, str] = {}
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                output[column] = f"{value:.6f}"
            elif value is None:
                output[column] = "NA"
            else:
                output[column] = str(value)
            widths[column] = max(widths[column], len(output[column]))
        formatted.append(output)

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    print("\n" + header)
    print(separator)
    for row in formatted:
        print(" | ".join(row[column].ljust(widths[column]) for column in columns))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "experiments" / DEFAULT_OUTPUT_NAME
    )
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}\n"
            "Move or rename the existing directory before rerunning."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if partial_dir.exists():
        # This directory is owned exclusively by this script and represents an
        # interrupted prior attempt, never a completed result.
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    policy_path = (
        root
        / "experiments/w16_sequential_evidence_policy_source_v1/selected_policy.json"
    )
    all_rows: List[Dict] = []
    dataset_summaries: Dict[str, Dict] = {}

    if args.dataset in {"all", "urfd"}:
        rows, summary = evaluate_dataset(
            dataset="URFD",
            prediction_path=(
                root
                / "experiments/urfd_quality_gated_logit_fusion_v3_w16_external/frame_predictions.csv"
            ),
            event_path=(
                root
                / "experiments/urfd_quality_gated_logit_fusion_v3_w16_external/event_results.csv"
            ),
            policy_path=policy_path,
            output_dir=partial_dir,
        )
        all_rows.extend(rows)
        dataset_summaries["URFD"] = summary

    if args.dataset in {"all", "le2i"}:
        rows, summary = evaluate_dataset(
            dataset="Le2i",
            prediction_path=(
                root / "experiments/le2i_frozen_inference_v1/frame_predictions.csv"
            ),
            inventory_path=(
                root / "experiments/le2i_inventory_v2/sequence_inventory.csv"
            ),
            event_path=(root / "experiments/le2i_inventory_v2/fall_events.csv"),
            policy_path=policy_path,
            output_dir=partial_dir,
        )
        all_rows.extend(rows)
        dataset_summaries["Le2i"] = summary

    comparison_fields = (
        "dataset",
        "variant",
        "windows",
        "binary_macro_f1",
        "binary_macro_auprc",
        "positive_f1",
        "positive_auprc",
        "balanced_accuracy",
        "event_recall",
        "detected_events",
        "total_events",
        "false_alarm_count",
        "negative_exposure_hours",
        "false_alarms_per_hour",
        "median_detection_delay_s",
        "duplicate_alarms",
    )
    write_csv(partial_dir / "comparison_summary.csv", all_rows, comparison_fields)
    final_summary = {
        "analysis_name": DEFAULT_OUTPUT_NAME,
        "purpose": "Controlled external comparison of fusion rules",
        "reporting_label": "post-hoc frozen external fusion ablation",
        "interpretation": (
            "All variants reuse the jointly trained quality-gated checkpoint "
            "branches and the unchanged CAUCAFall-selected candidate policy. "
            "The results isolate fusion behavior and must not be described as "
            "a new blind evaluation."
        ),
        "comparison_rows": all_rows,
        "datasets": dataset_summaries,
    }
    write_json(partial_dir / "comparison_summary.json", final_summary)

    os.replace(partial_dir, output_dir)

    print_comparison(all_rows)
    print(f"\nCompleted. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
