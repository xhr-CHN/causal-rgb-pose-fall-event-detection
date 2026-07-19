#!/usr/bin/env python3
"""Build causal confirmation features from URFD and revealed Le2i candidates.

The base evidence policy raises a candidate alarm at sample t. This script
waits for at most four additional 20-FPS samples and makes the confirmation
decision at t+4 (maximum added delay: 200 ms). No sample later than that
decision time is used. GMDCSA24 is never read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/data/yoloA27")
URFD_FEATURES = ROOT / "features/urfd_alarm_verifier_v2/alarm_features.csv"
URFD_PREDICTIONS = (
    ROOT
    / "experiments/urfd_quality_gated_logit_fusion_v3_w16_external/frame_predictions.csv"
)
LE2I_FEATURES = (
    ROOT
    / "experiments/le2i_final_blind_evaluation_v1/pre_verifier_alarm_classification.csv"
)
LE2I_PREDICTIONS = ROOT / "experiments/le2i_frozen_inference_v1/frame_predictions.csv"
OUTPUT_DIR = ROOT / "features/multidataset_alarm_confirmation_features_v2"

CONFIRMATION_FUTURE_SAMPLES = 4
CONFIRMATION_LENGTH = CONFIRMATION_FUTURE_SAMPLES + 1
EXPECTED_SAMPLE_INTERVAL_MS = 50.0

PRE_SIGNAL_NAMES = [
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

CONFIRM_SIGNALS = [
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
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        fail(f"CSV is empty: {path}")
    return rows, fields


def number(row: dict[str, str], name: str, default: float = 0.0) -> float:
    try:
        value = row.get(name, "")
        result = float(default if value in ("", None) else value)
        return result if math.isfinite(result) else float(default)
    except (TypeError, ValueError):
        return float(default)


def integer(row: dict[str, str], name: str, default: int = 0) -> int:
    return int(round(number(row, name, default)))


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        values = [0.0]
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / n
    if n > 1:
        x_mean = (n - 1) / 2.0
        denominator = sum((index - x_mean) ** 2 for index in range(n))
        slope = (
            sum((index - x_mean) * (value - mean) for index, value in enumerate(values))
            / denominator
            if denominator > 0.0
            else 0.0
        )
        delta = values[-1] - values[0]
    else:
        slope = 0.0
        delta = 0.0
    return {
        "last": values[-1],
        "mean": mean,
        "std": math.sqrt(max(0.0, variance)),
        "min": min(values),
        "max": max(values),
        "delta": delta,
        "slope": slope,
    }


def longest_run(values: list[float], threshold: float) -> int:
    best = 0
    current = 0
    for value in values:
        current = current + 1 if value >= threshold else 0
        best = max(best, current)
    return best


def extract_prediction_signals(row: dict[str, str]) -> dict[str, float]:
    return {
        "fused_fall_score": number(row, "prob_falling") + number(row, "prob_fallen"),
        "prob_falling": number(row, "prob_falling"),
        "prob_fallen": number(row, "prob_fallen"),
        "prob_adl": number(row, "prob_adl"),
        "pose_fall_score": number(row, "pose_prob_falling")
        + number(row, "pose_prob_fallen"),
        "rgb_fall_score": number(row, "rgb_prob_falling")
        + number(row, "rgb_prob_fallen"),
        "pose_gate_weight": number(row, "pose_gate_weight"),
        "branch_disagreement": number(row, "branch_disagreement"),
        "pose_found_ratio": number(row, "pose_found_ratio"),
        "mean_keypoint_conf": number(row, "mean_keypoint_conf"),
        "rgb_roi_crop_ratio": number(row, "rgb_roi_crop_ratio"),
    }


def prediction_index(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence_id"]].append(row)
    for sequence_rows in grouped.values():
        sequence_rows.sort(key=lambda row: integer(row, "sample_index"))
    return grouped


def normalize_source_candidates(
    dataset: str,
    feature_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    normalized = []
    for source_index, row in enumerate(feature_rows, start=1):
        if dataset == "URFD":
            label = integer(row, "label_true_alarm")
            classification = row.get("alarm_type", "")
            candidate_score = number(row, "trigger_fall_score")
            accumulated = number(row, "trigger_accumulated_evidence")
        elif dataset == "Le2i":
            classification = row.get("alarm_classification", "")
            label = int(classification == "event_alarm")
            candidate_score = number(row, "fall_score")
            accumulated = number(row, "accumulated_evidence")
        else:
            fail(f"Unsupported dataset: {dataset}")
        if label not in (0, 1):
            fail(f"Non-binary alarm label in {dataset} row {source_index}")
        normalized.append(
            {
                "dataset": dataset,
                "source_candidate_index": source_index,
                "sequence_id": row["sequence_id"],
                "sample_index": integer(row, "sample_index"),
                "sample_timestamp_ms": number(row, "sample_timestamp_ms"),
                "alarm_classification": classification,
                "label_true_alarm": label,
                "candidate_fall_score": candidate_score,
                "candidate_accumulated_evidence": accumulated,
                "source_row": row,
            }
        )
    return normalized


def build_dataset(
    dataset: str,
    candidate_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates = normalize_source_candidates(dataset, candidate_rows)
    predictions = prediction_index(prediction_rows)
    output: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    required_pre_features = [
        f"{signal}_{stat}" for signal in PRE_SIGNAL_NAMES for stat in STAT_NAMES
    ]
    missing = [name for name in required_pre_features if name not in candidate_rows[0]]
    if missing:
        fail(f"{dataset} candidate table misses pre-history features: {missing[:10]}")

    for candidate in candidates:
        sequence_id = str(candidate["sequence_id"])
        sequence_rows = predictions.get(sequence_id, [])
        positions = {
            integer(row, "sample_index"): position
            for position, row in enumerate(sequence_rows)
        }
        sample_index = int(candidate["sample_index"])
        if sample_index not in positions:
            skipped.append(
                {
                    "dataset": dataset,
                    "sequence_id": sequence_id,
                    "sample_index": sample_index,
                    "reason": "candidate_sample_not_found",
                }
            )
            continue

        start = positions[sample_index]
        observed = sequence_rows[start : start + CONFIRMATION_LENGTH]
        observed_count = len(observed)
        if not observed:
            fail(f"No candidate prediction row: {dataset}/{sequence_id}/{sample_index}")
        confirmation = observed + [observed[-1]] * (CONFIRMATION_LENGTH - observed_count)
        signal_rows = [extract_prediction_signals(row) for row in confirmation]
        decision_row = observed[-1]
        decision_delay = max(
            0.0,
            number(decision_row, "sample_timestamp_ms")
            - float(candidate["sample_timestamp_ms"]),
        )
        if decision_delay > CONFIRMATION_FUTURE_SAMPLES * EXPECTED_SAMPLE_INTERVAL_MS + 1.0:
            fail(
                f"Unexpected confirmation delay {decision_delay} ms: "
                f"{dataset}/{sequence_id}/{sample_index}"
            )

        row: dict[str, object] = {
            "dataset": dataset,
            "candidate_key": (
                f"{dataset.lower()}::{sequence_id}::{sample_index}::"
                f"{candidate['source_candidate_index']}"
            ),
            "sequence_id": sequence_id,
            "sample_index": sample_index,
            "candidate_timestamp_ms": candidate["sample_timestamp_ms"],
            "decision_sample_index": integer(decision_row, "sample_index"),
            "decision_timestamp_ms": number(decision_row, "sample_timestamp_ms"),
            "confirmation_added_delay_ms": decision_delay,
            "confirmation_observed_frames": observed_count,
            "confirmation_right_padding_frames": CONFIRMATION_LENGTH - observed_count,
            "alarm_classification": candidate["alarm_classification"],
            "label_true_alarm": candidate["label_true_alarm"],
            "candidate_fall_score": candidate["candidate_fall_score"],
            "candidate_accumulated_evidence": candidate["candidate_accumulated_evidence"],
        }
        source_row = candidate["source_row"]
        assert isinstance(source_row, dict)
        for name in required_pre_features:
            row[f"pre_{name}"] = number(source_row, name)

        for signal_name in CONFIRM_SIGNALS:
            values = [signals[signal_name] for signals in signal_rows]
            for stat_name, value in stats(values).items():
                row[f"confirm_{signal_name}_{stat_name}"] = value

        fall_scores = [signals["fused_fall_score"] for signals in signal_rows]
        pose_scores = [signals["pose_fall_score"] for signals in signal_rows]
        rgb_scores = [signals["rgb_fall_score"] for signals in signal_rows]
        for threshold in (0.50, 0.70, 0.85):
            tag = f"{int(round(threshold * 100)):02d}"
            row[f"confirm_fall_support_fraction_{tag}"] = (
                sum(value >= threshold for value in fall_scores) / CONFIRMATION_LENGTH
            )
            row[f"confirm_fall_longest_run_{tag}"] = (
                longest_run(fall_scores, threshold) / CONFIRMATION_LENGTH
            )
        row["confirm_modal_agreement_fraction_50"] = (
            sum(
                pose >= 0.50 and rgb >= 0.50
                for pose, rgb in zip(pose_scores, rgb_scores)
            )
            / CONFIRMATION_LENGTH
        )
        row["confirm_modal_either_fraction_50"] = (
            sum(
                pose >= 0.50 or rgb >= 0.50
                for pose, rgb in zip(pose_scores, rgb_scores)
            )
            / CONFIRMATION_LENGTH
        )
        row["confirm_recovery_drop_from_candidate"] = max(
            0.0, fall_scores[0] - min(fall_scores)
        )
        row["confirm_fallen_rise"] = (
            signal_rows[-1]["prob_fallen"] - signal_rows[0]["prob_fallen"]
        )
        row["confirm_falling_to_fallen_shift"] = (
            signal_rows[-1]["prob_fallen"]
            - signal_rows[-1]["prob_falling"]
            - signal_rows[0]["prob_fallen"]
            + signal_rows[0]["prob_falling"]
        )
        output.append(row)

    return output, skipped


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        fail(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    inputs = [URFD_FEATURES, URFD_PREDICTIONS, LE2I_FEATURES, LE2I_PREDICTIONS]
    for path in inputs:
        if not path.exists():
            fail(f"Required input does not exist: {path}")
    if OUTPUT_DIR.exists():
        fail(f"Output directory already exists; refusing to overwrite: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    try:
        urfd_features, _ = read_csv(URFD_FEATURES)
        urfd_predictions, _ = read_csv(URFD_PREDICTIONS)
        le2i_features, _ = read_csv(LE2I_FEATURES)
        le2i_predictions, _ = read_csv(LE2I_PREDICTIONS)

        urfd_rows, urfd_skipped = build_dataset(
            "URFD", urfd_features, urfd_predictions
        )
        le2i_rows, le2i_skipped = build_dataset(
            "Le2i", le2i_features, le2i_predictions
        )
        rows = urfd_rows + le2i_rows
        skipped = urfd_skipped + le2i_skipped
        if skipped:
            write_csv(OUTPUT_DIR / "skipped_candidates.csv", skipped)
            fail(f"Candidate alignment failed for {len(skipped)} rows")

        expected = {"URFD": len(urfd_features), "Le2i": len(le2i_features)}
        actual = Counter(str(row["dataset"]) for row in rows)
        if dict(actual) != expected:
            fail(f"Candidate counts differ: expected {expected}, got {dict(actual)}")

        rows.sort(
            key=lambda row: (
                str(row["dataset"]),
                str(row["sequence_id"]),
                int(row["sample_index"]),
            )
        )
        write_csv(OUTPUT_DIR / "alarm_confirmation_features.csv", rows)

        dataset_summary = {}
        for dataset in ("URFD", "Le2i"):
            subset = [row for row in rows if row["dataset"] == dataset]
            positives = sum(int(row["label_true_alarm"]) for row in subset)
            dataset_summary[dataset] = {
                "candidates": len(subset),
                "positive_true_event_alarms": positives,
                "negative_false_or_duplicate_alarms": len(subset) - positives,
                "sequences": len({str(row["sequence_id"]) for row in subset}),
                "fully_observed_confirmation": sum(
                    int(row["confirmation_observed_frames"]) == CONFIRMATION_LENGTH
                    for row in subset
                ),
            }

        metadata_names = {
            "dataset",
            "candidate_key",
            "sequence_id",
            "sample_index",
            "candidate_timestamp_ms",
            "decision_sample_index",
            "decision_timestamp_ms",
            "confirmation_added_delay_ms",
            "confirmation_observed_frames",
            "confirmation_right_padding_frames",
            "alarm_classification",
            "label_true_alarm",
        }
        feature_names = [name for name in rows[0] if name not in metadata_names]
        summary = {
            "protocol": {
                "development_datasets": ["URFD Camera 0 RGB", "Le2i revealed V1 test"],
                "untouched_final_blind_dataset": "GMDCSA24 v2.1",
                "gmdcsa24_read": False,
                "feature_protocol": (
                    "candidate at t plus four subsequent 20-FPS samples; "
                    "decision at the last observed confirmation sample"
                ),
                "future_samples_after_candidate": CONFIRMATION_FUTURE_SAMPLES,
                "maximum_added_decision_delay_ms": (
                    CONFIRMATION_FUTURE_SAMPLES * EXPECTED_SAMPLE_INTERVAL_MS
                ),
                "post_decision_samples_used": 0,
                "near_end_treatment": "right-pad last causally observed sample",
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_summary": dataset_summary,
            "total_candidates": len(rows),
            "total_positive_candidates": sum(
                int(row["label_true_alarm"]) for row in rows
            ),
            "total_negative_candidates": sum(
                1 - int(row["label_true_alarm"]) for row in rows
            ),
            "model_eligible_feature_count": len(feature_names),
            "model_eligible_feature_names": feature_names,
            "excluded_from_model": sorted(metadata_names),
            "input_sha256": {str(path): sha256_file(path) for path in inputs},
            "output": str(OUTPUT_DIR / "alarm_confirmation_features.csv"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        }
        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"\nCompleted: {OUTPUT_DIR}", flush=True)
    except Exception:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
