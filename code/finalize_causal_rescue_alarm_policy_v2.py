#!/usr/bin/env python3
"""Finalize the fixed causal rescue alarm policy on development OOF data.

Primary path: accept the compact V2 verifier at its original grouped-OOF
threshold. Rescue path: for a primary rejection, accept only when the Pose fall
score rose rapidly before the candidate and the fused Falling probability did
not decline during the 200-ms causal confirmation interval. GMDCSA24 is never
read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/data/yoloA27")
TRAIN_DIR = ROOT / "experiments/multidataset_alarm_verifier_v2_seed42"
TRAIN_SUMMARY = TRAIN_DIR / "summary.json"
OOF_PREDICTIONS = TRAIN_DIR / "oof_predictions.csv"
VERIFIER_MODEL = TRAIN_DIR / "alarm_verifier_v2_model.json"
FEATURES = (
    ROOT
    / "features/multidataset_alarm_confirmation_features_v2/"
    "alarm_confirmation_features.csv"
)
LOW_THRESHOLD_SUMMARY = ROOT / "experiments/event_preserving_alarm_policy_v2/summary.json"
OUTPUT_DIR = ROOT / "experiments/causal_rescue_alarm_policy_v2"

SELECTED_CONFIG = "pre_plus_confirmation_v2"
SCORE_COLUMN = f"{SELECTED_CONFIG}_oof_probability"
POSE_FALL_SCORE_SLOPE_MINIMUM = 0.05
CONFIRM_FALLING_PROBABILITY_DELTA_MINIMUM = 0.0
MAXIMUM_ADDED_DELAY_MS = 200.0

DATASET_CONSTANTS = {
    "URFD": {
        "total_fall_events": 30,
        "negative_exposure_hours": 0.08044444444444446,
        "expected_base_detected_events": 28,
        "expected_base_false_alarms": 26,
    },
    "Le2i": {
        "total_fall_events": 99,
        "negative_exposure_hours": 0.7356900838888888,
        "expected_base_detected_events": 94,
        "expected_base_false_alarms": 131,
    },
}


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        fail(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def is_duplicate(classification: str) -> bool:
    return "duplicate" in classification.lower()


def path_decision(row: dict[str, object], primary_threshold: float) -> tuple[bool, bool, bool]:
    primary = float(row["score"]) >= primary_threshold
    rescue = (
        not primary
        and float(row["pre_pose_fall_score_slope"])
        >= POSE_FALL_SCORE_SLOPE_MINIMUM
        and float(row["confirm_prob_falling_delta"])
        >= CONFIRM_FALLING_PROBABILITY_DELTA_MINIMUM
    )
    return primary, rescue, primary or rescue


def evaluate(
    rows: list[dict[str, object]],
    accepted_field: str,
) -> dict[str, object]:
    by_dataset: dict[str, dict[str, object]] = {}
    reductions = []
    for dataset, constants in DATASET_CONSTANTS.items():
        subset = [row for row in rows if row["dataset"] == dataset]
        accepted = [row for row in subset if bool(row[accepted_field])]
        base_true = sum(int(row["label_true_alarm"]) for row in subset)
        retained_true = sum(int(row["label_true_alarm"]) for row in accepted)
        base_false = sum(
            int(
                not int(row["label_true_alarm"])
                and not is_duplicate(str(row["alarm_classification"]))
            )
            for row in subset
        )
        retained_false = sum(
            int(
                not int(row["label_true_alarm"])
                and not is_duplicate(str(row["alarm_classification"]))
            )
            for row in accepted
        )
        base_duplicates = sum(
            int(
                not int(row["label_true_alarm"])
                and is_duplicate(str(row["alarm_classification"]))
            )
            for row in subset
        )
        retained_duplicates = sum(
            int(
                not int(row["label_true_alarm"])
                and is_duplicate(str(row["alarm_classification"]))
            )
            for row in accepted
        )
        if base_true != int(constants["expected_base_detected_events"]):
            fail(f"{dataset}: unexpected base detected-event count {base_true}")
        if base_false != int(constants["expected_base_false_alarms"]):
            fail(f"{dataset}: unexpected base false-alarm count {base_false}")

        total_events = int(constants["total_fall_events"])
        exposure = float(constants["negative_exposure_hours"])
        reduction = divide(base_false - retained_false, base_false)
        reductions.append(reduction)
        by_dataset[dataset] = {
            "total_fall_events": total_events,
            "base_detected_events": base_true,
            "retained_detected_events": retained_true,
            "rejected_detected_events": base_true - retained_true,
            "base_event_recall": divide(base_true, total_events),
            "retained_event_recall": divide(retained_true, total_events),
            "base_detected_event_retention": divide(retained_true, base_true),
            "base_false_alarms": base_false,
            "retained_false_alarms": retained_false,
            "rejected_false_alarms": base_false - retained_false,
            "false_alarm_reduction_fraction": reduction,
            "base_false_alarms_per_hour": divide(base_false, exposure),
            "retained_false_alarms_per_hour": divide(retained_false, exposure),
            "base_duplicate_alarms": base_duplicates,
            "retained_duplicate_alarms": retained_duplicates,
            "accepted_alarm_precision_excluding_duplicates": divide(
                retained_true, retained_true + retained_false
            ),
        }
    return {
        "by_dataset": by_dataset,
        "macro_false_alarm_reduction_fraction": sum(reductions) / len(reductions),
        "total_retained_detected_events": sum(
            int(value["retained_detected_events"]) for value in by_dataset.values()
        ),
        "total_retained_false_alarms": sum(
            int(value["retained_false_alarms"]) for value in by_dataset.values()
        ),
    }


def main() -> None:
    inputs = [
        TRAIN_SUMMARY,
        OOF_PREDICTIONS,
        VERIFIER_MODEL,
        FEATURES,
        LOW_THRESHOLD_SUMMARY,
    ]
    for path in inputs:
        if not path.exists():
            fail(f"Required input does not exist: {path}")
    if OUTPUT_DIR.exists():
        fail(f"Output directory already exists; refusing to overwrite: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    try:
        train_summary = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8"))
        verifier_model = json.loads(VERIFIER_MODEL.read_text(encoding="utf-8"))
        low_threshold_summary = json.loads(
            LOW_THRESHOLD_SUMMARY.read_text(encoding="utf-8")
        )
        oof_rows, oof_fields = read_csv(OOF_PREDICTIONS)
        feature_rows, feature_fields = read_csv(FEATURES)
        required_feature_fields = {
            "candidate_key",
            "alarm_classification",
            "pre_pose_fall_score_slope",
            "confirm_prob_falling_delta",
        }
        missing = sorted(required_feature_fields - set(feature_fields))
        if missing:
            fail(f"Feature table misses required fields: {missing}")
        if SCORE_COLUMN not in oof_fields:
            fail(f"OOF table misses score column: {SCORE_COLUMN}")
        if train_summary.get("selected_feature_configuration") != SELECTED_CONFIG:
            fail("Training summary selected a different feature configuration")
        if verifier_model.get("selected_feature_configuration") != SELECTED_CONFIG:
            fail("Verifier model selected a different feature configuration")

        primary_threshold = float(train_summary["selected_threshold"])
        if abs(primary_threshold - float(verifier_model["decision_threshold"])) > 1e-12:
            fail("Training-summary and verifier-model thresholds differ")
        feature_lookup = {row["candidate_key"]: row for row in feature_rows}
        if len(feature_lookup) != len(feature_rows):
            fail("Duplicate candidate_key in feature table")

        joined: list[dict[str, object]] = []
        for prediction in oof_rows:
            key = prediction["candidate_key"]
            feature = feature_lookup.get(key)
            if feature is None:
                fail(f"Missing feature row for candidate: {key}")
            if (
                prediction["dataset"] != feature["dataset"]
                or prediction["sequence_id"] != feature["sequence_id"]
                or int(prediction["label_true_alarm"])
                != int(feature["label_true_alarm"])
            ):
                fail(f"OOF/feature identity mismatch: {key}")
            row: dict[str, object] = {
                "candidate_key": key,
                "dataset": prediction["dataset"],
                "sequence_id": prediction["sequence_id"],
                "sample_index": int(prediction["sample_index"]),
                "alarm_classification": feature["alarm_classification"],
                "label_true_alarm": int(prediction["label_true_alarm"]),
                "score": float(prediction[SCORE_COLUMN]),
                "pre_pose_fall_score_slope": float(
                    feature["pre_pose_fall_score_slope"]
                ),
                "confirm_prob_falling_delta": float(
                    feature["confirm_prob_falling_delta"]
                ),
                "confirmation_added_delay_ms": float(
                    feature["confirmation_added_delay_ms"]
                ),
            }
            primary, rescue, accepted = path_decision(row, primary_threshold)
            row["primary_accepted"] = primary
            row["rescue_accepted"] = rescue
            row["final_accepted"] = accepted
            joined.append(row)
        if len(joined) != 281:
            fail(f"Expected 281 candidates, got {len(joined)}")

        primary_metrics = evaluate(joined, "primary_accepted")
        final_metrics = evaluate(joined, "final_accepted")
        rescue_rows = [row for row in joined if bool(row["rescue_accepted"])]
        rescue_summary = {
            "rescued_candidates": len(rescue_rows),
            "rescued_true_events": sum(
                int(row["label_true_alarm"]) for row in rescue_rows
            ),
            "rescued_false_alarms": sum(
                int(
                    not int(row["label_true_alarm"])
                    and not is_duplicate(str(row["alarm_classification"]))
                )
                for row in rescue_rows
            ),
            "rescued_duplicates": sum(
                int(
                    not int(row["label_true_alarm"])
                    and is_duplicate(str(row["alarm_classification"]))
                )
                for row in rescue_rows
            ),
            "by_dataset": dict(Counter(str(row["dataset"]) for row in rescue_rows)),
        }
        if rescue_summary != {
            "rescued_candidates": 8,
            "rescued_true_events": 5,
            "rescued_false_alarms": 3,
            "rescued_duplicates": 0,
            "by_dataset": {"Le2i": 8},
        }:
            fail(f"Unexpected fixed rescue-rule result: {rescue_summary}")

        policy = {
            "policy_name": "causal_rescue_alarm_policy_v2",
            "base_candidate_policy": {
                "resampled_fps": 20.0,
                "window_length": 16,
                "fall_score": "P(Falling) + P(Fallen)",
                "evidence_alpha": 0.5,
                "evidence_alarm_threshold": 0.85,
                "consecutive_evidence_frames": 2,
                "reset_threshold": 0.1,
                "reset_frames": 5,
            },
            "primary_verifier": {
                "model_path": str(VERIFIER_MODEL),
                "model_sha256": sha256_file(VERIFIER_MODEL),
                "score": "standardized L2 logistic probability",
                "threshold": primary_threshold,
                "feature_configuration": SELECTED_CONFIG,
            },
            "causal_rescue_path": {
                "applies_only_when_primary_rejected": True,
                "all_conditions_required": {
                    "pre_pose_fall_score_slope_minimum": (
                        POSE_FALL_SCORE_SLOPE_MINIMUM
                    ),
                    "confirm_prob_falling_delta_minimum": (
                        CONFIRM_FALLING_PROBABILITY_DELTA_MINIMUM
                    ),
                },
                "interpretation": (
                    "rapid pre-candidate rise in Pose fall evidence followed by "
                    "non-decreasing fused Falling probability during confirmation"
                ),
            },
            "timing": {
                "confirmation_future_samples": 4,
                "maximum_added_delay_ms": MAXIMUM_ADDED_DELAY_MS,
                "post_decision_samples_used": 0,
            },
            "development_datasets": ["URFD Camera 0 RGB", "Le2i revealed V1 test"],
            "untouched_final_blind_dataset": "GMDCSA24 v2.1",
            "gmdcsa24_read": False,
        }
        policy_path = OUTPUT_DIR / "selected_policy.json"
        policy_path.write_text(
            json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_csv(OUTPUT_DIR / "oof_policy_decisions.csv", joined)

        summary = {
            "protocol": {
                "selection_role": "development-only fixed causal rescue policy",
                "development_datasets": [
                    "URFD Camera 0 RGB",
                    "Le2i revealed V1 test",
                ],
                "untouched_final_blind_dataset": "GMDCSA24 v2.1",
                "gmdcsa24_read": False,
                "maximum_added_delay_ms": MAXIMUM_ADDED_DELAY_MS,
                "post_decision_samples_used": 0,
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "primary_threshold": primary_threshold,
            "rescue_rule": policy["causal_rescue_path"],
            "primary_only_oof": primary_metrics,
            "causal_rescue_oof": final_metrics,
            "rescue_effect": rescue_summary,
            "comparison_to_low_threshold_policy": {
                "low_threshold": low_threshold_summary["selected_threshold"],
                "low_threshold_metrics": low_threshold_summary[
                    "selected_threshold_metrics"
                ],
                "interpretation": (
                    "fixed rescue restores the same 121/122 detected events "
                    "with substantially fewer false alarms than global threshold lowering"
                ),
            },
            "selected_policy": str(policy_path),
            "selected_policy_sha256": sha256_file(policy_path),
            "verifier_model": str(VERIFIER_MODEL),
            "verifier_model_sha256": sha256_file(VERIFIER_MODEL),
            "input_sha256": {str(path): sha256_file(path) for path in inputs},
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
