#!/usr/bin/env python3
"""Reveal Le2i labels once and evaluate the already frozen alarm outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/data/yoloA27")
INFERENCE_DIR = ROOT / "experiments/le2i_frozen_inference_v1"
INVENTORY_DIR = ROOT / "experiments/le2i_inventory_v2"
INVENTORY = INVENTORY_DIR / "sequence_inventory.csv"
EVENTS = INVENTORY_DIR / "fall_events.csv"
INVENTORY_SUMMARY = INVENTORY_DIR / "summary.json"
OUTPUT_DIR = ROOT / "experiments/le2i_final_blind_evaluation_v1"
PARTIAL_OUTPUT_DIR = ROOT / "experiments/le2i_final_blind_evaluation_v1.partial"

INPUT_PATHS = {
    "frame_predictions": INFERENCE_DIR / "frame_predictions.csv",
    "candidate_alarms": INFERENCE_DIR / "candidate_alarms.csv",
    "final_alarms": INFERENCE_DIR / "final_alarms.csv",
    "resampling_summary": INFERENCE_DIR / "resampling_summary.csv",
    "inference_summary": INFERENCE_DIR / "inference_summary.json",
    "inference_script": ROOT / "run_le2i_frozen_inference_v1.py",
}

EXPECTED_SHA256 = {
    "frame_predictions": "667d0ffa084c542015d3ae567bee94df64e068615517d250703afb4772247296",
    "candidate_alarms": "e86580bcc3baf768db3209ff93f7e795c1cd55b9dbf7235997d6a2f29eeb44e0",
    "final_alarms": "322e5f435381bf818003e0dc739858721a6008ba18616343a18cdedbf9a80a8b",
    "resampling_summary": "a4331f7446fa3ebfce58221187c5cd4c5368adc1054f3bbc8b9b33feddc596e2",
    "inference_summary": "7642d2d80d13049bd2c7da51e220f38c571e33f39900da196defe0674b8f6b27",
    "inference_script": "875eb5da6c30b4c3d7eb39c42cd68c0e4e09a4c56b84ff956003fc6856adbf23",
}

EXPECTED_SEQUENCES = 190
EXPECTED_FALL_EVENTS = 99
EXPECTED_ADL_SEQUENCES = 91
EXPECTED_RAW_FRAMES = 75911
EXPECTED_WINDOWS = 58328
EXPECTED_CANDIDATES = 227
EXPECTED_FINAL_ALARMS = 113
EXPECTED_NEGATIVE_EXPOSURE_HOURS = 0.7356900838888888
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260717


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict, name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        result = float(value)
        return result if np.isfinite(result) else float(default)
    except (TypeError, ValueError):
        return float(default)


def integer(row: dict, name: str, default: int = 0) -> int:
    return int(round(number(row, name, default)))


def divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def verify_frozen_inputs() -> dict:
    actual = {}
    for name, path in INPUT_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[name] = sha256(path)
        if actual[name] != EXPECTED_SHA256[name]:
            raise RuntimeError(
                f"Frozen inference artifact changed: {path}\n"
                f"expected={EXPECTED_SHA256[name]}\nactual={actual[name]}"
            )
    inference = json.loads(
        INPUT_PATHS["inference_summary"].read_text(encoding="utf-8")
    )
    protocol = inference["protocol"]
    if protocol["labels_or_annotations_read"] is not False:
        raise RuntimeError("Inference was not label-isolated")
    if protocol["model_or_threshold_tuning_performed"] is not False:
        raise RuntimeError("Target tuning occurred before label reveal")
    if protocol["future_frames_used"] != 0:
        raise RuntimeError("Frozen inference used future frames")
    expected_counts = {
        "sequences": EXPECTED_SEQUENCES,
        "raw_frames": EXPECTED_RAW_FRAMES,
        "causal_windows": EXPECTED_WINDOWS,
        "evidence_candidate_alarms": EXPECTED_CANDIDATES,
        "verifier_retained_alarms": EXPECTED_FINAL_ALARMS,
    }
    for name, expected in expected_counts.items():
        if int(inference[name]) != expected:
            raise RuntimeError(f"Frozen inference count changed: {name}")
    return {"hashes": actual, "inference_summary": inference}


def load_ground_truth() -> tuple[dict, dict, dict]:
    for path in (INVENTORY, EVENTS, INVENTORY_SUMMARY):
        if not path.is_file():
            raise FileNotFoundError(path)
    inventory_summary = json.loads(INVENTORY_SUMMARY.read_text(encoding="utf-8"))
    if inventory_summary["error_count"] != 0:
        raise RuntimeError("Le2i inventory contains errors")
    inventory_rows = read_csv(INVENTORY)
    event_rows = read_csv(EVENTS)
    if len(inventory_rows) != EXPECTED_SEQUENCES:
        raise RuntimeError("Unexpected Le2i sequence count")
    if len(event_rows) != EXPECTED_FALL_EVENTS:
        raise RuntimeError("Unexpected Le2i event count")

    inventory_by_sequence = {}
    fall_count = 0
    adl_count = 0
    for row in inventory_rows:
        sequence_id = row["sequence_id"]
        if sequence_id in inventory_by_sequence:
            raise ValueError(f"Duplicate sequence: {sequence_id}")
        sequence_type = row["sequence_type"]
        is_fall = sequence_type == "fall"
        if is_fall:
            fall_count += 1
        elif sequence_type in {"adl_annotated", "adl_unannotated"}:
            adl_count += 1
        else:
            raise ValueError(f"Unexpected sequence type: {sequence_type}")
        inventory_by_sequence[sequence_id] = {
            "sequence_id": sequence_id,
            "scene": row["scene"],
            "sequence_type": sequence_type,
            "is_fall": is_fall,
            "frame_count": integer(row, "frame_count"),
            "fps": number(row, "fps"),
            "duration_seconds": number(row, "duration_seconds"),
            "negative_exposure_seconds": number(
                row, "negative_exposure_seconds"
            ),
            "onset_frame": (
                integer(row, "fall_onset_frame") if is_fall else None
            ),
            "end_frame": integer(row, "fall_end_frame") if is_fall else None,
        }
    if fall_count != EXPECTED_FALL_EVENTS or adl_count != EXPECTED_ADL_SEQUENCES:
        raise RuntimeError("Unexpected fall/ADL sequence counts")

    events_by_sequence = {}
    for row in event_rows:
        sequence_id = row["sequence_id"]
        if sequence_id in events_by_sequence:
            raise ValueError(f"Duplicate event: {sequence_id}")
        events_by_sequence[sequence_id] = {
            "sequence_id": sequence_id,
            "scene": row["scene"],
            "event_id": row["event_id"],
            "onset_frame": integer(row, "onset_frame"),
            "end_frame": integer(row, "end_frame"),
            "onset_ms": number(row, "onset_ms"),
            "end_ms": number(row, "end_ms"),
        }
    expected_fall_ids = {
        sequence_id
        for sequence_id, row in inventory_by_sequence.items()
        if row["is_fall"]
    }
    if set(events_by_sequence) != expected_fall_ids:
        raise RuntimeError("Fall-event and inventory sequence sets differ")
    for sequence_id, event in events_by_sequence.items():
        inventory_row = inventory_by_sequence[sequence_id]
        if (
            event["onset_frame"] != inventory_row["onset_frame"]
            or event["end_frame"] != inventory_row["end_frame"]
        ):
            raise RuntimeError(f"Event boundary mismatch: {sequence_id}")

    negative_hours = sum(
        row["negative_exposure_seconds"] for row in inventory_by_sequence.values()
    ) / 3600.0
    if abs(negative_hours - EXPECTED_NEGATIVE_EXPOSURE_HOURS) > 1e-12:
        raise RuntimeError("Negative exposure changed")
    return inventory_by_sequence, events_by_sequence, inventory_summary


def validate_alarm_rows(
    rows: list[dict[str, str]], inventory: dict, expected_count: int
) -> None:
    if len(rows) != expected_count:
        raise RuntimeError(f"Alarm count changed: {len(rows)}/{expected_count}")
    for row in rows:
        sequence_id = row["sequence_id"]
        if sequence_id not in inventory:
            raise KeyError(f"Unknown alarm sequence: {sequence_id}")
        frame = integer(row, "source_frame_number")
        if not 1 <= frame <= inventory[sequence_id]["frame_count"]:
            raise ValueError(f"Alarm frame outside sequence: {sequence_id}/{frame}")


def classify_alarms(
    rows: list[dict[str, str]], inventory: dict, events: dict
) -> tuple[list[dict], list[dict], dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sequence_id"]].append(row)
    for sequence_rows in grouped.values():
        sequence_rows.sort(
            key=lambda row: (
                integer(row, "source_frame_number"),
                integer(row, "candidate_id"),
            )
        )

    classified = []
    event_results = []
    sequence_stats = {}
    delays = []
    false_alarm_count = 0
    duplicate_count = 0
    detected_count = 0

    for sequence_id, metadata in inventory.items():
        alarms = grouped.get(sequence_id, [])
        false_alarms = []
        event_alarms = []
        if metadata["is_fall"]:
            onset = events[sequence_id]["onset_frame"]
            false_alarms = [
                row for row in alarms if integer(row, "source_frame_number") < onset
            ]
            event_alarms = [
                row for row in alarms if integer(row, "source_frame_number") >= onset
            ]
        else:
            false_alarms = alarms

        for row in false_alarms:
            output = dict(row)
            output["alarm_classification"] = (
                "false_alarm_early" if metadata["is_fall"] else "false_alarm_adl"
            )
            classified.append(output)
        for index, row in enumerate(event_alarms):
            output = dict(row)
            output["alarm_classification"] = (
                "event_alarm" if index == 0 else "duplicate_event_alarm"
            )
            classified.append(output)

        false_alarm_count += len(false_alarms)
        detected = bool(event_alarms)
        delay = None
        first_frame = None
        first_timestamp = None
        duplicates = max(0, len(event_alarms) - 1)
        if metadata["is_fall"]:
            event = events[sequence_id]
            if detected:
                detected_count += 1
                first = event_alarms[0]
                first_frame = integer(first, "source_frame_number")
                first_timestamp = number(first, "source_timestamp_ms")
                delay = first_timestamp - event["onset_ms"]
                if delay < -1e-6:
                    raise RuntimeError("Negative post-onset detection delay")
                delays.append(delay)
                duplicate_count += duplicates
            event_results.append(
                {
                    "sequence_id": sequence_id,
                    "scene": metadata["scene"],
                    "event_id": event["event_id"],
                    "onset_frame": event["onset_frame"],
                    "onset_ms": event["onset_ms"],
                    "detected": int(detected),
                    "first_alarm_source_frame": first_frame,
                    "first_alarm_timestamp_ms": first_timestamp,
                    "detection_delay_ms": delay,
                    "early_false_alarms": len(false_alarms),
                    "alarms_after_onset": len(event_alarms),
                    "duplicate_alarms": duplicates,
                }
            )
        sequence_stats[sequence_id] = {
            "sequence_id": sequence_id,
            "scene": metadata["scene"],
            "event_count": int(metadata["is_fall"]),
            "detected_event_count": int(metadata["is_fall"] and detected),
            "false_alarm_count": len(false_alarms),
            "duplicate_alarm_count": duplicates,
            "negative_exposure_hours": metadata["negative_exposure_seconds"]
            / 3600.0,
            "detection_delay_ms": delay,
        }

    negative_hours = sum(
        row["negative_exposure_hours"] for row in sequence_stats.values()
    )
    metrics = {
        "total_fall_events": EXPECTED_FALL_EVENTS,
        "detected_fall_events": detected_count,
        "missed_fall_events": EXPECTED_FALL_EVENTS - detected_count,
        "event_recall": detected_count / EXPECTED_FALL_EVENTS,
        "total_alarm_count": len(rows),
        "false_alarm_count": false_alarm_count,
        "sequences_with_false_alarm": sum(
            row["false_alarm_count"] > 0 for row in sequence_stats.values()
        ),
        "negative_exposure_hours": negative_hours,
        "false_alarms_per_hour": divide(false_alarm_count, negative_hours),
        "duplicate_alarms_after_onset": duplicate_count,
        "mean_detection_delay_ms": float(np.mean(delays)) if delays else None,
        "median_detection_delay_ms": float(np.median(delays)) if delays else None,
        "p95_detection_delay_ms": (
            float(np.percentile(delays, 95)) if delays else None
        ),
        "minimum_detection_delay_ms": min(delays) if delays else None,
        "maximum_detection_delay_ms": max(delays) if delays else None,
    }
    return classified, event_results, {"metrics": metrics, "sequence_stats": sequence_stats}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return [None, None]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def bootstrap_intervals(sequence_stats: dict, seed_offset: int) -> dict:
    rows = list(sequence_stats.values())
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    recalls = []
    false_alarm_rates = []
    median_delays = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, len(rows), size=len(rows))
        sample = [rows[index] for index in indices]
        event_count = sum(row["event_count"] for row in sample)
        detected_count = sum(row["detected_event_count"] for row in sample)
        exposure = sum(row["negative_exposure_hours"] for row in sample)
        false_alarms = sum(row["false_alarm_count"] for row in sample)
        delays = [
            row["detection_delay_ms"]
            for row in sample
            if row["detection_delay_ms"] is not None
        ]
        if event_count:
            recalls.append(detected_count / event_count)
        if exposure:
            false_alarm_rates.append(false_alarms / exposure)
        if delays:
            median_delays.append(float(np.median(delays)))

    def interval(values):
        if not values:
            return [None, None]
        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]

    return {
        "method": "sequence-level percentile bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED + seed_offset,
        "event_recall_95ci": interval(recalls),
        "false_alarms_per_hour_95ci": interval(false_alarm_rates),
        "median_detection_delay_ms_95ci": interval(median_delays),
    }


def scene_metrics(sequence_stats: dict) -> list[dict]:
    grouped = defaultdict(list)
    for row in sequence_stats.values():
        grouped[row["scene"]].append(row)
    output = []
    for scene in sorted(grouped):
        rows = grouped[scene]
        event_count = sum(row["event_count"] for row in rows)
        detected_count = sum(row["detected_event_count"] for row in rows)
        false_alarms = sum(row["false_alarm_count"] for row in rows)
        exposure = sum(row["negative_exposure_hours"] for row in rows)
        delays = [
            row["detection_delay_ms"]
            for row in rows
            if row["detection_delay_ms"] is not None
        ]
        output.append(
            {
                "scene": scene,
                "sequences": len(rows),
                "fall_events": event_count,
                "detected_events": detected_count,
                "event_recall": (
                    detected_count / event_count if event_count else None
                ),
                "false_alarm_count": false_alarms,
                "negative_exposure_hours": exposure,
                "false_alarms_per_hour": divide(false_alarms, exposure),
                "median_detection_delay_ms": (
                    float(np.median(delays)) if delays else None
                ),
            }
        )
    return output


def binary_auroc(target: np.ndarray, score: np.ndarray) -> float:
    positive_count = int(target.sum())
    negative_count = len(target) - positive_count
    if not positive_count or not negative_count:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    for value in np.unique(score):
        indices = np.flatnonzero(score == value)
        if len(indices) > 1:
            ranks[indices] = ranks[indices].mean()
    rank_sum = ranks[target == 1].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order]
    positive_count = int(sorted_target.sum())
    if not positive_count:
        return float("nan")
    true_positive = np.cumsum(sorted_target)
    precision = true_positive / np.arange(1, len(target) + 1)
    return float(precision[sorted_target == 1].sum() / positive_count)


def frame_level_metrics(predictions: list[dict], inventory: dict, events: dict):
    target = []
    prediction = []
    score = []
    for row in predictions:
        sequence_id = row["sequence_id"]
        if sequence_id not in inventory:
            raise KeyError(f"Unknown frame-prediction sequence: {sequence_id}")
        frame = integer(row, "source_frame_number")
        is_positive = 0
        if inventory[sequence_id]["is_fall"]:
            event = events[sequence_id]
            is_positive = int(event["onset_frame"] <= frame <= event["end_frame"])
        target.append(is_positive)
        prediction.append(int(integer(row, "predicted_state_id") != 0))
        score.append(number(row, "prob_falling") + number(row, "prob_fallen"))
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    score = np.clip(np.asarray(score, dtype=np.float64), 0.0, 1.0)
    tp = int(np.sum((target == 1) & (prediction == 1)))
    fp = int(np.sum((target == 0) & (prediction == 1)))
    tn = int(np.sum((target == 0) & (prediction == 0)))
    fn = int(np.sum((target == 1) & (prediction == 0)))
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    f1 = divide(2.0 * precision * recall, precision + recall)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        mask = (score >= edges[index]) & (
            score <= edges[index + 1]
            if index == 9
            else score < edges[index + 1]
        )
        if mask.any():
            ece += float(mask.mean()) * abs(float(target[mask].mean() - score[mask].mean()))
    return {
        "evaluation_unit": "20-FPS causal-window endpoint",
        "positive_definition": "official Le2i onset-through-end interval, inclusive",
        "prediction_definition": "argmax state is Falling or Fallen",
        "samples": len(target),
        "positive_samples": int(target.sum()),
        "negative_samples": int(len(target) - target.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": divide(tp + tn, len(target)),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "auroc": binary_auroc(target, score),
        "auprc_average_precision": average_precision(target, score),
        "brier_score": float(np.mean((score - target) ** 2)),
        "ece_10_bins": float(ece),
    }


def write_stage_outputs(name: str, classified: list[dict], events: list[dict], stats: dict):
    alarm_path = PARTIAL_OUTPUT_DIR / f"{name}_alarm_classification.csv"
    event_path = PARTIAL_OUTPUT_DIR / f"{name}_event_results.csv"
    scene_path = PARTIAL_OUTPUT_DIR / f"{name}_scene_results.csv"
    if classified:
        write_csv(alarm_path, classified, list(classified[0]))
    else:
        alarm_path.write_text("candidate_id\n", encoding="utf-8")
    write_csv(event_path, events, list(events[0]))
    scenes = scene_metrics(stats["sequence_stats"])
    write_csv(scene_path, scenes, list(scenes[0]))
    return scenes


def main() -> None:
    frozen = verify_frozen_inputs()
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Final blind evaluation already exists; do not overwrite: {OUTPUT_DIR}"
        )
    if PARTIAL_OUTPUT_DIR.exists():
        shutil.rmtree(PARTIAL_OUTPUT_DIR)
    PARTIAL_OUTPUT_DIR.mkdir(parents=True)

    print("Frozen inference hashes: VERIFIED", flush=True)
    print("Le2i labels revealed for final evaluation: YES", flush=True)
    print("Model/policy/threshold adjustment permitted: NO", flush=True)

    inventory, events, inventory_summary = load_ground_truth()
    candidate_rows = read_csv(INPUT_PATHS["candidate_alarms"])
    final_rows = read_csv(INPUT_PATHS["final_alarms"])
    predictions = read_csv(INPUT_PATHS["frame_predictions"])
    validate_alarm_rows(candidate_rows, inventory, EXPECTED_CANDIDATES)
    validate_alarm_rows(final_rows, inventory, EXPECTED_FINAL_ALARMS)
    if len(predictions) != EXPECTED_WINDOWS:
        raise RuntimeError("Frame-prediction count changed")

    candidate_ids = {integer(row, "candidate_id") for row in candidate_rows}
    final_ids = {integer(row, "candidate_id") for row in final_rows}
    if len(candidate_ids) != len(candidate_rows):
        raise RuntimeError("Duplicate candidate alarm identifiers")
    if len(final_ids) != len(final_rows):
        raise RuntimeError("Duplicate final alarm identifiers")
    passed_ids = {
        integer(row, "candidate_id")
        for row in candidate_rows
        if integer(row, "verifier_passed") == 1
    }
    if final_ids != passed_ids or not final_ids.issubset(candidate_ids):
        raise RuntimeError("Final alarms do not match verifier-passed candidates")

    pre_classified, pre_events, pre = classify_alarms(
        candidate_rows, inventory, events
    )
    post_classified, post_events, post = classify_alarms(
        final_rows, inventory, events
    )
    pre_scenes = write_stage_outputs("pre_verifier", pre_classified, pre_events, pre)
    post_scenes = write_stage_outputs(
        "post_verifier", post_classified, post_events, post
    )

    pre_metrics = pre["metrics"]
    post_metrics = post["metrics"]
    pre_metrics["event_recall_wilson_95ci"] = wilson_interval(
        pre_metrics["detected_fall_events"], EXPECTED_FALL_EVENTS
    )
    post_metrics["event_recall_wilson_95ci"] = wilson_interval(
        post_metrics["detected_fall_events"], EXPECTED_FALL_EVENTS
    )
    pre_metrics["bootstrap_95ci"] = bootstrap_intervals(
        pre["sequence_stats"], 0
    )
    post_metrics["bootstrap_95ci"] = bootstrap_intervals(
        post["sequence_stats"], 1
    )

    pre_event_map = {row["sequence_id"]: integer(row, "detected") for row in pre_events}
    post_event_map = {row["sequence_id"]: integer(row, "detected") for row in post_events}
    paired = {
        "both_detected": sum(
            pre_event_map[key] == 1 and post_event_map[key] == 1
            for key in pre_event_map
        ),
        "pre_only_detected": sum(
            pre_event_map[key] == 1 and post_event_map[key] == 0
            for key in pre_event_map
        ),
        "post_only_detected": sum(
            pre_event_map[key] == 0 and post_event_map[key] == 1
            for key in pre_event_map
        ),
        "neither_detected": sum(
            pre_event_map[key] == 0 and post_event_map[key] == 0
            for key in pre_event_map
        ),
    }
    false_alarm_reduction = (
        1.0
        - post_metrics["false_alarm_count"] / pre_metrics["false_alarm_count"]
        if pre_metrics["false_alarm_count"]
        else None
    )
    comparison = {
        "candidate_alarms_before_verifier": len(candidate_rows),
        "alarms_retained_after_verifier": len(final_rows),
        "alarms_rejected_by_verifier": len(candidate_rows) - len(final_rows),
        "alarm_rejection_fraction": 1.0 - len(final_rows) / len(candidate_rows),
        "false_alarm_reduction_fraction": false_alarm_reduction,
        "event_recall_change": post_metrics["event_recall"]
        - pre_metrics["event_recall"],
        "false_alarms_per_hour_change": post_metrics["false_alarms_per_hour"]
        - pre_metrics["false_alarms_per_hour"],
        "paired_event_detection": paired,
    }

    summary = {
        "protocol": {
            "dataset": "Le2i",
            "role": "single frozen final blind evaluation after label reveal",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_training_dataset": "CAUCAFall",
            "development_dataset": "URFD Camera 0 RGB",
            "target_labels_used_for_training_or_selection": False,
            "target_labels_revealed_only_after_inference_outputs_were_hashed": True,
            "model_policy_or_threshold_changed_after_label_reveal": False,
            "event_detection_rule": inventory_summary["protocol"][
                "event_detection_rule"
            ],
            "negative_exposure_rule": inventory_summary["protocol"][
                "evaluation_negative_exposure"
            ],
            "negative_exposure_hours": EXPECTED_NEGATIVE_EXPOSURE_HOURS,
            "bootstrap_unit": "complete video sequence",
        },
        "frozen_inference_artifact_hashes": frozen["hashes"],
        "dataset": {
            "sequences": EXPECTED_SEQUENCES,
            "fall_events": EXPECTED_FALL_EVENTS,
            "adl_sequences": EXPECTED_ADL_SEQUENCES,
            "raw_frames": EXPECTED_RAW_FRAMES,
            "causal_window_endpoints": EXPECTED_WINDOWS,
        },
        "frame_level_fusion_model": frame_level_metrics(
            predictions, inventory, events
        ),
        "event_level_before_verifier": pre_metrics,
        "event_level_after_verifier": post_metrics,
        "verifier_effect": comparison,
        "scene_level_before_verifier": pre_scenes,
        "scene_level_after_verifier": post_scenes,
        "important_interpretation": (
            "Le2i was used once as the untouched final test set. These results "
            "must not be used to retune the model, evidence policy, or verifier."
        ),
    }
    (PARTIAL_OUTPUT_DIR / "final_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (PARTIAL_OUTPUT_DIR / "LABELS_REVEALED.txt").write_text(
        "Le2i labels were revealed only for the frozen final evaluation.\n"
        "No model, policy, or threshold may be retuned using these results.\n",
        encoding="utf-8",
    )
    os.replace(PARTIAL_OUTPUT_DIR, OUTPUT_DIR)

    print("\nLe2i final blind evaluation completed.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
