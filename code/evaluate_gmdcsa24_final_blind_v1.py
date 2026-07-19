#!/usr/bin/env python3
"""Evaluate the already frozen GMDCSA24 inference exactly once.

This script may read the private mapping and official GMDCSA24 CSV files, so
it must only be run after frozen inference outputs and their SHA-256 values
exist. It never changes a model, threshold, candidate rule, verifier, or rescue
rule. Event decisions use the actual alarm decision timestamp; the base
candidate-only diagnostic uses its original candidate timestamp.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/data/yoloA27")
RAW_ROOT = ROOT / "GMDCSA24"
PRIVATE_MAPPING = (
    ROOT
    / "experiments/gmdcsa24_final_blind_lock_v1/"
    "PRIVATE_EVALUATION_MAPPING_DO_NOT_USE_FOR_INFERENCE.csv"
)
DATASET_LOCK_SUMMARY = (
    ROOT / "experiments/gmdcsa24_final_blind_lock_v1/dataset_lock_summary.json"
)
FINAL_PROTOCOL_LOCK = (
    ROOT / "experiments/gmdcsa24_final_blind_protocol_lock_v1/lock_summary.json"
)
FEATURE_DIR = ROOT / "features/gmdcsa24_frozen_features_v1"
FEATURE_MANIFEST = FEATURE_DIR / "sequence_feature_manifest.csv"
INFERENCE_DIR = ROOT / "experiments/gmdcsa24_frozen_inference_v1"
FRAME_PREDICTIONS = INFERENCE_DIR / "frame_predictions.csv"
CANDIDATE_ALARMS = INFERENCE_DIR / "candidate_alarms.csv"
FINAL_ALARMS = INFERENCE_DIR / "final_alarms.csv"
RESAMPLING_SUMMARY = INFERENCE_DIR / "resampling_summary.csv"
INFERENCE_SUMMARY = INFERENCE_DIR / "inference_summary.json"
OUTPUT_DIR = ROOT / "experiments/gmdcsa24_final_blind_evaluation_v1"
PARTIAL_OUTPUT_DIR = ROOT / "experiments/gmdcsa24_final_blind_evaluation_v1.partial"

EXPECTED_SEQUENCE_COUNT = 160
EXPECTED_FALL_COUNT = 79
EXPECTED_ADL_COUNT = 81
EXPECTED_OFFICIAL_CSV_COUNT = 8
EXPECTED_CANDIDATES = 110
EXPECTED_PRIMARY_ALARMS = 68
EXPECTED_RESCUE_ALARMS = 6
EXPECTED_FINAL_ALARMS = 74
EXPECTED_OUTPUT_SHA256 = {
    str(FRAME_PREDICTIONS): (
        "14127bcc02306fbdd4269d7e1f9ef45624e92fe4470d9aaacd77893c732394ec"
    ),
    str(CANDIDATE_ALARMS): (
        "3aad362194960215d3e91a402bae6ef7af155871ba3783d5bd951f824d2a31ad"
    ),
    str(FINAL_ALARMS): (
        "5c0ea1a8d220d9cd68d70dcc892c104a7cf0a11ea3d0b7a07f67259d7278cbc2"
    ),
    str(RESAMPLING_SUMMARY): (
        "49e97cc60181b8ab639185a46f3a81ffcee0adfd01f98bacd166efb6fb403e24"
    ),
}

MAPPING_FIELDS = [
    "sequence_id",
    "source_relative_path",
    "subject",
    "category",
    "source_sha256",
    "blind_video_file",
    "link_method",
]
FALL_INTERVAL_PATTERN = re.compile(
    r"(?P<label>[^;\[\]]*?\bfall\w*(?:\s*\([^)]*\))?[^;\[\]]*)"
    r"\[\s*(?P<start>\d+(?:\.\d+)?)\s*to\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*\]",
    flags=re.IGNORECASE,
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def write_csv(path: Path, rows: list, fields: list) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict, name: str, default: float = 0.0) -> float:
    try:
        value = row.get(name, "")
        result = float(default if value in ("", None) else value)
        return result if math.isfinite(result) else float(default)
    except (TypeError, ValueError):
        return float(default)


def integer(row: dict, name: str, default: int = 0) -> int:
    return int(round(number(row, name, default)))


def divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def normalized_header(row: dict) -> dict:
    return {str(key).strip(): value for key, value in row.items()}


def normalize_fall_type(label: str) -> str:
    upper = label.upper()
    direction = re.search(r"\((FW|BW|SW|LS|RS)\)", upper)
    if direction:
        code = direction.group(1)
        aliases = {"LS": "SW", "RS": "SW"}
        return aliases.get(code, code)
    if "FORWARD" in upper:
        return "FW"
    if "BACKWARD" in upper:
        return "BW"
    if "SIDE" in upper or "LEFT" in upper or "RIGHT" in upper:
        return "SW"
    return "UNSPECIFIED"


def parse_fall_intervals(classes_text: str):
    intervals = []
    for match in FALL_INTERVAL_PATTERN.finditer(classes_text or ""):
        start = float(match.group("start"))
        end = float(match.group("end"))
        label = " ".join(match.group("label").split())
        intervals.append(
            {
                "label": label,
                "fall_type": normalize_fall_type(label),
                "start_seconds": start,
                "end_seconds": end,
            }
        )
    return intervals


def verify_frozen_outputs() -> dict:
    required = [
        PRIVATE_MAPPING,
        DATASET_LOCK_SUMMARY,
        FINAL_PROTOCOL_LOCK,
        FEATURE_MANIFEST,
        FRAME_PREDICTIONS,
        CANDIDATE_ALARMS,
        FINAL_ALARMS,
        RESAMPLING_SUMMARY,
        INFERENCE_SUMMARY,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    verified = {}
    for text, expected in EXPECTED_OUTPUT_SHA256.items():
        path = Path(text)
        actual = sha256(path)
        if actual != expected:
            fail(
                f"Frozen inference output changed: {path}; "
                f"expected={expected}, actual={actual}"
            )
        verified[text] = actual

    inference = json.loads(INFERENCE_SUMMARY.read_text(encoding="utf-8"))
    protocol = inference.get("protocol", {})
    if protocol.get("labels_annotations_private_mapping_or_raw_dataset_read") is not False:
        fail("Inference summary does not prove label isolation")
    if protocol.get("model_or_threshold_tuning_performed") is not False:
        fail("Inference summary reports model or threshold tuning")
    if int(inference.get("sequences", -1)) != EXPECTED_SEQUENCE_COUNT:
        fail("Unexpected sequence count in inference summary")
    if int(inference.get("evidence_candidate_alarms", -1)) != EXPECTED_CANDIDATES:
        fail("Unexpected frozen candidate count")
    if int(inference.get("primary_retained_alarms", -1)) != EXPECTED_PRIMARY_ALARMS:
        fail("Unexpected frozen primary-alarm count")
    if int(inference.get("causal_rescue_alarms", -1)) != EXPECTED_RESCUE_ALARMS:
        fail("Unexpected frozen rescue-alarm count")
    if int(inference.get("final_retained_alarms", -1)) != EXPECTED_FINAL_ALARMS:
        fail("Unexpected frozen final-alarm count")
    embedded = inference.get("output_sha256", {})
    for text, expected in EXPECTED_OUTPUT_SHA256.items():
        if embedded.get(Path(text).name) != expected:
            fail(f"Inference summary hash differs for {Path(text).name}")
    verified[str(INFERENCE_SUMMARY)] = sha256(INFERENCE_SUMMARY)
    return verified


def load_actual_durations() -> dict:
    manifest, _ = read_csv(FEATURE_MANIFEST)
    durations = {}
    for row in manifest:
        summary_path = Path(row["sequence_summary_json"])
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sequence_id = row["sequence_id"]
        if summary.get("sequence_id") != sequence_id:
            fail(f"Sequence summary identity mismatch: {sequence_id}")
        durations[sequence_id] = float(summary["duration_seconds"])
    if len(durations) != EXPECTED_SEQUENCE_COUNT:
        fail(f"Expected {EXPECTED_SEQUENCE_COUNT} actual durations")
    return durations


def load_official_metadata() -> tuple:
    mapping_rows, mapping_fields = read_csv(PRIVATE_MAPPING)
    if mapping_fields != MAPPING_FIELDS:
        fail(f"Private mapping fields changed: {mapping_fields}")
    if len(mapping_rows) != EXPECTED_SEQUENCE_COUNT:
        fail(f"Expected {EXPECTED_SEQUENCE_COUNT} private mapping rows")
    if len({row["sequence_id"] for row in mapping_rows}) != len(mapping_rows):
        fail("Duplicate sequence_id in private mapping")

    csv_paths = sorted(RAW_ROOT.glob("Subject */*.csv"))
    if len(csv_paths) != EXPECTED_OFFICIAL_CSV_COUNT:
        fail(f"Expected {EXPECTED_OFFICIAL_CSV_COUNT} official CSV files")
    official_lookup = {}
    for path in csv_paths:
        rows, _ = read_csv(path)
        for raw_row in rows:
            row = normalized_header(raw_row)
            file_name = str(row.get("File Name", "")).strip()
            if not file_name:
                fail(f"Missing File Name in {path}")
            key = (path.parent.name, path.stem, file_name)
            if key in official_lookup:
                fail(f"Duplicate official metadata key: {key}")
            official_lookup[key] = row

    durations = load_actual_durations()
    metadata = {}
    audit_rows = []
    parse_errors = []
    for mapping in mapping_rows:
        sequence_id = mapping["sequence_id"]
        source = Path(mapping["source_relative_path"])
        if len(source.parts) != 3:
            fail(f"Unexpected source mapping path: {source}")
        subject, category, file_name = source.parts
        if mapping["subject"] != subject or mapping["category"] != category:
            fail(f"Private mapping identity disagreement: {sequence_id}")
        official = official_lookup.get((subject, category, file_name))
        if official is None:
            fail(f"Official metadata row not found: {source}")
        classes_text = str(official.get("Classes", "")).strip()
        intervals = parse_fall_intervals(classes_text)
        if category == "Fall" and not intervals:
            parse_errors.append(
                {
                    "sequence_id": sequence_id,
                    "source_relative_path": source.as_posix(),
                    "classes": classes_text,
                    "reason": "fall_interval_not_parsed",
                }
            )
            continue
        if category == "ADL" and intervals:
            parse_errors.append(
                {
                    "sequence_id": sequence_id,
                    "source_relative_path": source.as_posix(),
                    "classes": classes_text,
                    "reason": "unexpected_fall_interval_in_adl",
                }
            )
            continue
        actual_duration = durations[sequence_id]
        official_duration = number(official, "Length (seconds)")
        if category == "Fall":
            onset = min(item["start_seconds"] for item in intervals)
            end = max(item["end_seconds"] for item in intervals)
            if onset < 0.0 or end < onset or onset > actual_duration + 1.0:
                parse_errors.append(
                    {
                        "sequence_id": sequence_id,
                        "source_relative_path": source.as_posix(),
                        "classes": classes_text,
                        "reason": (
                            f"invalid_fall_interval_{onset}_to_{end}_"
                            f"duration_{actual_duration}"
                        ),
                    }
                )
                continue
            fall_label = " | ".join(item["label"] for item in intervals)
            fall_types = sorted({item["fall_type"] for item in intervals})
            fall_type = "+".join(fall_types)
        else:
            onset = None
            end = None
            fall_label = ""
            fall_type = "ADL"
        row = {
            "sequence_id": sequence_id,
            "subject": subject,
            "category": category,
            "file_name": file_name,
            "source_relative_path": source.as_posix(),
            "actual_duration_seconds": actual_duration,
            "official_duration_seconds": official_duration,
            "fall_onset_seconds": onset,
            "fall_end_seconds": end,
            "fall_label": fall_label,
            "fall_type": fall_type,
            "time_of_recording": str(official.get("Time of Recording", "")).strip(),
            "attire": str(official.get("Attire", "")).strip(),
            "description": str(official.get("Description", "")).strip(),
            "classes": classes_text,
        }
        metadata[sequence_id] = row
        audit_rows.append(row)

    if parse_errors:
        PARTIAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_csv(
            PARTIAL_OUTPUT_DIR / "annotation_parse_errors.csv",
            parse_errors,
            list(parse_errors[0]),
        )
        fail(
            f"Official annotation parsing failed for {len(parse_errors)} videos; "
            f"see {PARTIAL_OUTPUT_DIR / 'annotation_parse_errors.csv'}"
        )
    counts = Counter(row["category"] for row in metadata.values())
    if counts != Counter({"ADL": EXPECTED_ADL_COUNT, "Fall": EXPECTED_FALL_COUNT}):
        fail(f"Unexpected category counts after metadata join: {dict(counts)}")
    if len(metadata) != EXPECTED_SEQUENCE_COUNT:
        fail("Metadata join did not produce 160 sequences")
    return metadata, audit_rows


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": divide(tp + tn, len(y_true)),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": divide(2.0 * precision * recall, precision + recall),
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    for value in np.unique(scores):
        indices = np.flatnonzero(scores == value)
        if len(indices) > 1:
            ranks[indices] = ranks[indices].mean()
    rank_sum = float(ranks[y_true == 1].sum())
    return divide(
        rank_sum - positives * (positives + 1) / 2.0,
        positives * negatives,
    )


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    ordered = y_true[order]
    positives = int(ordered.sum())
    if positives == 0:
        return float("nan")
    cumulative = np.cumsum(ordered)
    ranks = np.arange(1, len(ordered) + 1)
    return float(np.sum((cumulative / ranks) * ordered) / positives)


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    result = 0.0
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        mask = (scores >= lower) & (
            scores <= upper if index == 9 else scores < upper
        )
        if np.any(mask):
            result += float(mask.mean()) * abs(
                float(scores[mask].mean()) - float(y_true[mask].mean())
            )
    return result


def evaluate_frames(metadata: dict):
    rows, _ = read_csv(FRAME_PREDICTIONS)
    output = []
    y_true = []
    y_pred = []
    scores = []
    for row in rows:
        sequence_id = row["sequence_id"]
        info = metadata.get(sequence_id)
        if info is None:
            fail(f"Frame prediction has unknown sequence_id: {sequence_id}")
        timestamp_seconds = number(row, "sample_timestamp_ms") / 1000.0
        gt_fall = int(
            info["category"] == "Fall"
            and timestamp_seconds >= float(info["fall_onset_seconds"])
            and timestamp_seconds <= float(info["fall_end_seconds"]) + 1e-9
        )
        fall_score = min(
            1.0, max(0.0, number(row, "prob_falling") + number(row, "prob_fallen"))
        )
        predicted = int(integer(row, "predicted_state_id") in (1, 2))
        enriched = dict(row)
        enriched.update(
            {
                "subject": info["subject"],
                "category": info["category"],
                "fall_type": info["fall_type"],
                "fall_onset_seconds": info["fall_onset_seconds"],
                "fall_end_seconds": info["fall_end_seconds"],
                "gt_fall": gt_fall,
                "fall_score": fall_score,
                "binary_predicted_fall": predicted,
            }
        )
        output.append(enriched)
        y_true.append(gt_fall)
        y_pred.append(predicted)
        scores.append(fall_score)
    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_pred_array = np.asarray(y_pred, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    metrics = binary_metrics(y_true_array, y_pred_array)
    metrics.update(
        {
            "evaluation_scope": "available W16 causal windows only",
            "positive_definition": "official Fall interval",
            "prediction_definition": "argmax state is Falling or Fallen",
            "windows": len(output),
            "positive_windows": int(y_true_array.sum()),
            "negative_windows": int(len(y_true_array) - y_true_array.sum()),
            "auroc": auroc(y_true_array, score_array),
            "auprc_average_precision": average_precision(y_true_array, score_array),
            "brier_score": float(np.mean((score_array - y_true_array) ** 2)),
            "ece_10_bins": expected_calibration_error(y_true_array, score_array),
        }
    )
    return metrics, output


def alarm_timestamp(row: dict, timestamp_field: str) -> float:
    return number(row, timestamp_field)


def evaluate_alarm_stage(
    stage_name: str, alarm_rows: list, timestamp_field: str, metadata: dict
):
    grouped = defaultdict(list)
    for row in alarm_rows:
        if row["sequence_id"] not in metadata:
            fail(f"Alarm has unknown sequence_id: {row['sequence_id']}")
        grouped[row["sequence_id"]].append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                alarm_timestamp(row, timestamp_field),
                integer(row, "candidate_id"),
            )
        )

    classifications = []
    event_rows = []
    delays = []
    false_alarm_count = 0
    duplicate_count = 0
    detected_events = 0
    negative_exposure_seconds = 0.0
    for sequence_id, info in sorted(metadata.items()):
        rows = grouped.get(sequence_id, [])
        if info["category"] == "ADL":
            negative_exposure_seconds += float(info["actual_duration_seconds"])
            for row in rows:
                false_alarm_count += 1
                classifications.append(
                    {
                        "stage": stage_name,
                        "candidate_id": row.get("candidate_id", ""),
                        "candidate_key": row.get("candidate_key", ""),
                        "sequence_id": sequence_id,
                        "subject": info["subject"],
                        "category": info["category"],
                        "fall_type": info["fall_type"],
                        "alarm_timestamp_ms": alarm_timestamp(row, timestamp_field),
                        "fall_onset_ms": "",
                        "classification": "false_alarm_adl",
                    }
                )
            event_rows.append(
                {
                    "stage": stage_name,
                    "sequence_id": sequence_id,
                    "subject": info["subject"],
                    "category": info["category"],
                    "fall_type": info["fall_type"],
                    "fall_onset_ms": "",
                    "detected": "",
                    "first_detection_timestamp_ms": "",
                    "detection_delay_ms": "",
                    "false_alarms_before_onset_or_in_adl": len(rows),
                    "duplicate_alarms_after_detection": 0,
                }
            )
            continue

        onset_ms = float(info["fall_onset_seconds"]) * 1000.0
        negative_exposure_seconds += min(
            float(info["fall_onset_seconds"]), float(info["actual_duration_seconds"])
        )
        early = [row for row in rows if alarm_timestamp(row, timestamp_field) < onset_ms]
        after = [row for row in rows if alarm_timestamp(row, timestamp_field) >= onset_ms]
        false_alarm_count += len(early)
        for row in early:
            classifications.append(
                {
                    "stage": stage_name,
                    "candidate_id": row.get("candidate_id", ""),
                    "candidate_key": row.get("candidate_key", ""),
                    "sequence_id": sequence_id,
                    "subject": info["subject"],
                    "category": info["category"],
                    "fall_type": info["fall_type"],
                    "alarm_timestamp_ms": alarm_timestamp(row, timestamp_field),
                    "fall_onset_ms": onset_ms,
                    "classification": "false_alarm_pre_onset",
                }
            )
        if after:
            first = after[0]
            detected_events += 1
            delay = alarm_timestamp(first, timestamp_field) - onset_ms
            delays.append(delay)
            classifications.append(
                {
                    "stage": stage_name,
                    "candidate_id": first.get("candidate_id", ""),
                    "candidate_key": first.get("candidate_key", ""),
                    "sequence_id": sequence_id,
                    "subject": info["subject"],
                    "category": info["category"],
                    "fall_type": info["fall_type"],
                    "alarm_timestamp_ms": alarm_timestamp(first, timestamp_field),
                    "fall_onset_ms": onset_ms,
                    "classification": "true_event_alarm",
                }
            )
            for row in after[1:]:
                duplicate_count += 1
                classifications.append(
                    {
                        "stage": stage_name,
                        "candidate_id": row.get("candidate_id", ""),
                        "candidate_key": row.get("candidate_key", ""),
                        "sequence_id": sequence_id,
                        "subject": info["subject"],
                        "category": info["category"],
                        "fall_type": info["fall_type"],
                        "alarm_timestamp_ms": alarm_timestamp(row, timestamp_field),
                        "fall_onset_ms": onset_ms,
                        "classification": "duplicate_after_event_detection",
                    }
                )
            first_timestamp = alarm_timestamp(first, timestamp_field)
        else:
            delay = ""
            first_timestamp = ""
        event_rows.append(
            {
                "stage": stage_name,
                "sequence_id": sequence_id,
                "subject": info["subject"],
                "category": info["category"],
                "fall_type": info["fall_type"],
                "fall_onset_ms": onset_ms,
                "detected": int(bool(after)),
                "first_detection_timestamp_ms": first_timestamp,
                "detection_delay_ms": delay,
                "false_alarms_before_onset_or_in_adl": len(early),
                "duplicate_alarms_after_detection": max(0, len(after) - 1),
            }
        )

    delay_array = np.asarray(delays, dtype=np.float64)
    metrics = {
        "stage": stage_name,
        "alarm_timestamp_definition": timestamp_field,
        "total_alarm_rows": len(alarm_rows),
        "total_fall_events": EXPECTED_FALL_COUNT,
        "detected_fall_events": detected_events,
        "missed_fall_events": EXPECTED_FALL_COUNT - detected_events,
        "event_recall": divide(detected_events, EXPECTED_FALL_COUNT),
        "false_alarm_count": false_alarm_count,
        "negative_exposure_hours": negative_exposure_seconds / 3600.0,
        "false_alarms_per_hour": divide(
            false_alarm_count, negative_exposure_seconds / 3600.0
        ),
        "duplicate_alarms_after_event_detection": duplicate_count,
        "event_alarm_precision_excluding_duplicates": divide(
            detected_events, detected_events + false_alarm_count
        ),
        "mean_detection_delay_ms": (
            float(delay_array.mean()) if len(delay_array) else None
        ),
        "median_detection_delay_ms": (
            float(np.median(delay_array)) if len(delay_array) else None
        ),
        "p95_detection_delay_ms": (
            float(np.percentile(delay_array, 95)) if len(delay_array) else None
        ),
    }
    return metrics, classifications, event_rows


def subgroup_rows(final_event_rows: list, metadata: dict, field: str) -> list:
    output = []
    values = sorted({str(info[field]) for info in metadata.values()})
    for value in values:
        sequence_ids = {
            sequence_id
            for sequence_id, info in metadata.items()
            if str(info[field]) == value
        }
        subset = [row for row in final_event_rows if row["sequence_id"] in sequence_ids]
        falls = [row for row in subset if row["category"] == "Fall"]
        adl = [row for row in subset if row["category"] == "ADL"]
        detected = sum(integer(row, "detected") for row in falls)
        false_alarms = sum(
            integer(row, "false_alarms_before_onset_or_in_adl") for row in subset
        )
        exposure_seconds = 0.0
        for sequence_id in sequence_ids:
            info = metadata[sequence_id]
            exposure_seconds += (
                float(info["actual_duration_seconds"])
                if info["category"] == "ADL"
                else min(
                    float(info["fall_onset_seconds"]),
                    float(info["actual_duration_seconds"]),
                )
            )
        output.append(
            {
                "group_field": field,
                "group_value": value,
                "sequences": len(sequence_ids),
                "fall_events": len(falls),
                "adl_videos": len(adl),
                "detected_fall_events": detected,
                "event_recall": divide(detected, len(falls)) if falls else "",
                "false_alarm_count": false_alarms,
                "negative_exposure_hours": exposure_seconds / 3600.0,
                "false_alarms_per_hour": divide(
                    false_alarms, exposure_seconds / 3600.0
                ),
            }
        )
    return output


def main() -> None:
    if OUTPUT_DIR.exists():
        fail(f"Final evaluation output already exists; refusing overwrite: {OUTPUT_DIR}")
    if PARTIAL_OUTPUT_DIR.exists():
        shutil.rmtree(PARTIAL_OUTPUT_DIR)
    frozen_hashes = verify_frozen_outputs()
    PARTIAL_OUTPUT_DIR.mkdir(parents=True)

    try:
        metadata, annotation_audit = load_official_metadata()
        candidate_rows, _ = read_csv(CANDIDATE_ALARMS)
        final_rows, _ = read_csv(FINAL_ALARMS)
        if len(candidate_rows) != EXPECTED_CANDIDATES:
            fail(f"Expected {EXPECTED_CANDIDATES} candidate rows")
        primary_rows = [row for row in candidate_rows if integer(row, "primary_accepted")]
        rescue_rows = [row for row in candidate_rows if integer(row, "rescue_accepted")]
        accepted_rows = [row for row in candidate_rows if integer(row, "final_accepted")]
        if len(primary_rows) != EXPECTED_PRIMARY_ALARMS:
            fail(f"Expected {EXPECTED_PRIMARY_ALARMS} primary alarms")
        if len(rescue_rows) != EXPECTED_RESCUE_ALARMS:
            fail(f"Expected {EXPECTED_RESCUE_ALARMS} rescue alarms")
        if len(accepted_rows) != EXPECTED_FINAL_ALARMS or len(final_rows) != EXPECTED_FINAL_ALARMS:
            fail(f"Expected {EXPECTED_FINAL_ALARMS} final alarms")
        accepted_keys = {row["candidate_key"] for row in accepted_rows}
        final_keys = {row["candidate_key"] for row in final_rows}
        if accepted_keys != final_keys or len(final_keys) != len(final_rows):
            fail("candidate_alarms and final_alarms disagree")

        frame_metrics, enriched_frames = evaluate_frames(metadata)
        stages = [
            ("base_candidate", candidate_rows, "sample_timestamp_ms"),
            ("primary_verifier", primary_rows, "decision_timestamp_ms"),
            ("final_causal_rescue", accepted_rows, "decision_timestamp_ms"),
        ]
        stage_metrics = {}
        all_classifications = []
        all_event_rows = []
        final_event_rows = []
        for name, alarms, timestamp_field in stages:
            metrics, classifications, event_rows = evaluate_alarm_stage(
                name, alarms, timestamp_field, metadata
            )
            stage_metrics[name] = metrics
            all_classifications.extend(classifications)
            all_event_rows.extend(event_rows)
            if name == "final_causal_rescue":
                final_event_rows = event_rows

        subgroup = subgroup_rows(final_event_rows, metadata, "subject")
        subgroup.extend(subgroup_rows(final_event_rows, metadata, "fall_type"))
        subgroup.extend(subgroup_rows(final_event_rows, metadata, "time_of_recording"))

        write_csv(
            PARTIAL_OUTPUT_DIR / "annotation_audit.csv",
            annotation_audit,
            list(annotation_audit[0]),
        )
        write_csv(
            PARTIAL_OUTPUT_DIR / "frame_predictions_with_ground_truth.csv",
            enriched_frames,
            list(enriched_frames[0]),
        )
        write_csv(
            PARTIAL_OUTPUT_DIR / "alarm_classification.csv",
            all_classifications,
            list(all_classifications[0]),
        )
        write_csv(
            PARTIAL_OUTPUT_DIR / "event_results.csv",
            all_event_rows,
            list(all_event_rows[0]),
        )
        write_csv(
            PARTIAL_OUTPUT_DIR / "subgroup_metrics.csv",
            subgroup,
            list(subgroup[0]),
        )

        summary = {
            "protocol": {
                "dataset": "GMDCSA24 v2.1",
                "role": "one-time frozen final blind evaluation",
                "evaluation_run_after_frozen_output_hash_verification": True,
                "model_threshold_or_policy_changed": False,
                "event_positive_definition": (
                    "first accepted alarm whose actual decision timestamp is at "
                    "or after official Fall onset"
                ),
                "false_alarm_definition": (
                    "alarm in an ADL video or before official Fall onset"
                ),
                "duplicate_definition": (
                    "additional accepted alarm after the first event detection in "
                    "the same Fall video"
                ),
                "negative_exposure_definition": (
                    "all actual ADL video duration plus pre-onset duration of Fall videos"
                ),
                "base_candidate_diagnostic_timestamp": "sample_timestamp_ms",
                "primary_and_final_alarm_timestamp": "decision_timestamp_ms",
                "frame_metric_scope": "available W16 causal windows only",
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "sequences": len(metadata),
                "fall_videos": sum(
                    info["category"] == "Fall" for info in metadata.values()
                ),
                "adl_videos": sum(
                    info["category"] == "ADL" for info in metadata.values()
                ),
                "subjects": len({info["subject"] for info in metadata.values()}),
                "fall_type_counts": dict(
                    sorted(
                        Counter(
                            info["fall_type"]
                            for info in metadata.values()
                            if info["category"] == "Fall"
                        ).items()
                    )
                ),
            },
            "frame_level": frame_metrics,
            "event_level": stage_metrics,
            "rescue_effect": {
                "rescued_alarm_rows": len(rescue_rows),
                "additional_detected_events": (
                    stage_metrics["final_causal_rescue"]["detected_fall_events"]
                    - stage_metrics["primary_verifier"]["detected_fall_events"]
                ),
                "additional_false_alarms": (
                    stage_metrics["final_causal_rescue"]["false_alarm_count"]
                    - stage_metrics["primary_verifier"]["false_alarm_count"]
                ),
            },
            "frozen_inference_sha256_verified": frozen_hashes,
            "revealed_label_artifact_sha256": {
                str(PRIVATE_MAPPING): sha256(PRIVATE_MAPPING),
                **{
                    str(path): sha256(path)
                    for path in sorted(RAW_ROOT.glob("Subject */*.csv"))
                },
            },
            "evaluation_script_sha256": sha256(Path(__file__).resolve()),
        }
        output_paths = [
            PARTIAL_OUTPUT_DIR / "annotation_audit.csv",
            PARTIAL_OUTPUT_DIR / "frame_predictions_with_ground_truth.csv",
            PARTIAL_OUTPUT_DIR / "alarm_classification.csv",
            PARTIAL_OUTPUT_DIR / "event_results.csv",
            PARTIAL_OUTPUT_DIR / "subgroup_metrics.csv",
        ]
        summary["output_sha256"] = {
            path.name: sha256(path) for path in output_paths
        }
        (PARTIAL_OUTPUT_DIR / "final_evaluation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (PARTIAL_OUTPUT_DIR / "evaluation_complete.lock").write_text(
            (
                "Frozen GMDCSA24 evaluation completed once. Do not tune and rerun "
                "this protocol as the final blind result.\n"
            ),
            encoding="utf-8",
        )
        os_summary_hash = sha256(
            PARTIAL_OUTPUT_DIR / "final_evaluation_summary.json"
        )
        (PARTIAL_OUTPUT_DIR / "final_evaluation_summary.sha256").write_text(
            f"{os_summary_hash}  final_evaluation_summary.json\n",
            encoding="utf-8",
        )
        PARTIAL_OUTPUT_DIR.replace(OUTPUT_DIR)
    except Exception:
        if not (PARTIAL_OUTPUT_DIR / "annotation_parse_errors.csv").exists():
            shutil.rmtree(PARTIAL_OUTPUT_DIR, ignore_errors=True)
        raise

    print("GMDCSA24 frozen final evaluation completed.", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Summary SHA-256: {os_summary_hash}", flush=True)
    print(f"Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
