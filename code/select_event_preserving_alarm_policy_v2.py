#!/usr/bin/env python3
"""Select an event-preserving operating threshold from development OOF scores.

The full V2 verifier is retained. Its threshold is reselected so that at least
98% of the base detector's already-detected events are retained separately on
URFD and Le2i. Among feasible thresholds, macro false-alarm retention is
minimized. GMDCSA24 is never read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/data/yoloA27")
TRAIN_DIR = ROOT / "experiments/multidataset_alarm_verifier_v2_seed42"
TRAIN_SUMMARY = TRAIN_DIR / "summary.json"
OOF_PREDICTIONS = TRAIN_DIR / "oof_predictions.csv"
SOURCE_MODEL = TRAIN_DIR / "alarm_verifier_v2_model.json"
FEATURES = (
    ROOT
    / "features/multidataset_alarm_confirmation_features_v2/"
    "alarm_confirmation_features.csv"
)
OUTPUT_DIR = ROOT / "experiments/event_preserving_alarm_policy_v2"

SELECTED_CONFIG = "pre_plus_confirmation_v2"
SCORE_COLUMN = f"{SELECTED_CONFIG}_oof_probability"
MINIMUM_BASE_EVENT_RETENTION = 0.98

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


def threshold_margin(scores: np.ndarray, threshold: float) -> float:
    below = scores[scores < threshold]
    above = scores[scores >= threshold]
    if len(below) == 0 or len(above) == 0:
        return 0.0
    return float(above.min() - below.max())


def is_duplicate(classification: str) -> bool:
    return "duplicate" in classification.lower()


def evaluate_threshold(
    rows: list[dict[str, object]], threshold: float
) -> dict[str, object]:
    by_dataset: dict[str, dict[str, object]] = {}
    false_alarm_retention = []
    for dataset, constants in DATASET_CONSTANTS.items():
        subset = [row for row in rows if row["dataset"] == dataset]
        accepted = [row for row in subset if float(row["score"]) >= threshold]
        base_true = sum(int(row["label_true_alarm"]) for row in subset)
        retained_true = sum(int(row["label_true_alarm"]) for row in accepted)
        base_duplicates = sum(
            int(not int(row["label_true_alarm"]) and is_duplicate(str(row["alarm_classification"])))
            for row in subset
        )
        retained_duplicates = sum(
            int(not int(row["label_true_alarm"]) and is_duplicate(str(row["alarm_classification"])))
            for row in accepted
        )
        base_false = sum(
            int(not int(row["label_true_alarm"]) and not is_duplicate(str(row["alarm_classification"])))
            for row in subset
        )
        retained_false = sum(
            int(not int(row["label_true_alarm"]) and not is_duplicate(str(row["alarm_classification"])))
            for row in accepted
        )
        if base_true != int(constants["expected_base_detected_events"]):
            fail(f"{dataset}: expected {constants['expected_base_detected_events']} base true alarms, got {base_true}")
        if base_false != int(constants["expected_base_false_alarms"]):
            fail(f"{dataset}: expected {constants['expected_base_false_alarms']} base false alarms, got {base_false}")

        required_retained = int(math.ceil(MINIMUM_BASE_EVENT_RETENTION * base_true))
        exposure = float(constants["negative_exposure_hours"])
        total_events = int(constants["total_fall_events"])
        event_delays = [
            float(row["confirmation_added_delay_ms"])
            for row in accepted
            if int(row["label_true_alarm"])
        ]
        metrics = {
            "base_detected_events": base_true,
            "required_retained_events": required_retained,
            "retained_detected_events": retained_true,
            "rejected_detected_events": base_true - retained_true,
            "base_event_recall": divide(base_true, total_events),
            "retained_event_recall": divide(retained_true, total_events),
            "base_event_retention": divide(retained_true, base_true),
            "constraint_satisfied": retained_true >= required_retained,
            "base_false_alarms": base_false,
            "retained_false_alarms": retained_false,
            "rejected_false_alarms": base_false - retained_false,
            "false_alarm_reduction_fraction": divide(base_false - retained_false, base_false),
            "base_false_alarms_per_hour": divide(base_false, exposure),
            "retained_false_alarms_per_hour": divide(retained_false, exposure),
            "base_duplicate_alarms": base_duplicates,
            "retained_duplicate_alarms": retained_duplicates,
            "confirmation_added_delay_ms_mean": (
                float(np.mean(event_delays)) if event_delays else None
            ),
            "confirmation_added_delay_ms_max": (
                float(np.max(event_delays)) if event_delays else None
            ),
        }
        by_dataset[dataset] = metrics
        false_alarm_retention.append(divide(retained_false, base_false))

    return {
        "by_dataset": by_dataset,
        "all_retention_constraints_satisfied": all(
            bool(metrics["constraint_satisfied"])
            for metrics in by_dataset.values()
        ),
        "macro_false_alarm_retention_fraction": float(np.mean(false_alarm_retention)),
        "macro_false_alarm_reduction_fraction": 1.0 - float(np.mean(false_alarm_retention)),
        "total_retained_true_events": sum(
            int(metrics["retained_detected_events"]) for metrics in by_dataset.values()
        ),
        "total_retained_false_alarms": sum(
            int(metrics["retained_false_alarms"]) for metrics in by_dataset.values()
        ),
    }


def main() -> None:
    inputs = [TRAIN_SUMMARY, OOF_PREDICTIONS, SOURCE_MODEL, FEATURES]
    for path in inputs:
        if not path.exists():
            fail(f"Required input does not exist: {path}")
    if OUTPUT_DIR.exists():
        fail(f"Output directory already exists; refusing to overwrite: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    try:
        train_summary = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8"))
        model = json.loads(SOURCE_MODEL.read_text(encoding="utf-8"))
        oof_rows, oof_fields = read_csv(OOF_PREDICTIONS)
        feature_rows, _ = read_csv(FEATURES)
        if SCORE_COLUMN not in oof_fields:
            fail(f"OOF score column not found: {SCORE_COLUMN}")
        if train_summary.get("selected_feature_configuration") != SELECTED_CONFIG:
            fail("Training summary selected a different feature configuration")
        if model.get("selected_feature_configuration") != SELECTED_CONFIG:
            fail("Source model selected a different feature configuration")

        feature_lookup = {row["candidate_key"]: row for row in feature_rows}
        if len(feature_lookup) != len(feature_rows):
            fail("Duplicate candidate_key in feature table")
        joined: list[dict[str, object]] = []
        for row in oof_rows:
            key = row["candidate_key"]
            feature = feature_lookup.get(key)
            if feature is None:
                fail(f"OOF candidate missing from feature table: {key}")
            if row["dataset"] != feature["dataset"] or row["sequence_id"] != feature["sequence_id"]:
                fail(f"OOF/feature identity mismatch: {key}")
            joined.append(
                {
                    "candidate_key": key,
                    "dataset": row["dataset"],
                    "sequence_id": row["sequence_id"],
                    "sample_index": int(row["sample_index"]),
                    "label_true_alarm": int(row["label_true_alarm"]),
                    "alarm_classification": feature["alarm_classification"],
                    "confirmation_added_delay_ms": float(feature["confirmation_added_delay_ms"]),
                    "score": float(row[SCORE_COLUMN]),
                }
            )
        if len(joined) != 281:
            fail(f"Expected 281 joined candidates, got {len(joined)}")

        scores = np.asarray([float(row["score"]) for row in joined], dtype=np.float64)
        unique = np.unique(scores)
        candidates = [0.0]
        candidates.extend(
            float((left + right) / 2.0)
            for left, right in zip(unique[:-1], unique[1:])
        )
        candidates.append(1.0)
        feasible = []
        for threshold in candidates:
            metrics = evaluate_threshold(joined, threshold)
            if metrics["all_retention_constraints_satisfied"]:
                feasible.append((threshold, metrics))
        if not feasible:
            fail("No threshold satisfies the event-retention constraints")
        threshold, selected_metrics = min(
            feasible,
            key=lambda item: (
                float(item[1]["macro_false_alarm_retention_fraction"]),
                -int(item[1]["total_retained_true_events"]),
                -threshold_margin(scores, item[0]),
                -item[0],
            ),
        )
        selected_metrics["threshold_margin"] = threshold_margin(scores, threshold)

        original_threshold = float(model["decision_threshold"])
        original_metrics = evaluate_threshold(joined, original_threshold)
        updated_model = dict(model)
        updated_model["decision_threshold"] = float(threshold)
        updated_model["threshold_policy"] = {
            "name": "event_preserving_oof_v2",
            "minimum_base_event_retention_per_dataset": MINIMUM_BASE_EVENT_RETENTION,
            "selection_data": ["URFD", "Le2i revealed V1 test"],
            "untouched_final_blind_dataset": "GMDCSA24 v2.1",
            "selection_score": SCORE_COLUMN,
            "selection_order": [
                "satisfy per-dataset base-event retention",
                "minimize macro false-alarm retention",
                "maximize retained true events",
                "maximize adjacent score margin",
            ],
        }
        updated_model_path = OUTPUT_DIR / "alarm_verifier_v2_event_preserving_model.json"
        updated_model_path.write_text(
            json.dumps(updated_model, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        decision_rows = []
        for row in joined:
            decision_rows.append(
                {
                    **row,
                    "selected_threshold": threshold,
                    "selected_prediction": int(float(row["score"]) >= threshold),
                }
            )
        write_csv(OUTPUT_DIR / "oof_event_preserving_decisions.csv", decision_rows)

        summary = {
            "protocol": {
                "development_datasets": ["URFD Camera 0 RGB", "Le2i revealed V1 test"],
                "untouched_final_blind_dataset": "GMDCSA24 v2.1",
                "gmdcsa24_read": False,
                "score_source": "complete-sequence grouped OOF predictions",
                "selected_feature_configuration": SELECTED_CONFIG,
                "minimum_base_event_retention_per_dataset": MINIMUM_BASE_EVENT_RETENTION,
                "maximum_confirmation_added_delay_ms": 200.0,
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "original_threshold": original_threshold,
            "original_threshold_metrics": original_metrics,
            "selected_threshold": float(threshold),
            "selected_threshold_metrics": selected_metrics,
            "source_model": str(SOURCE_MODEL),
            "source_model_sha256": sha256_file(SOURCE_MODEL),
            "updated_model": str(updated_model_path),
            "updated_model_sha256": sha256_file(updated_model_path),
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
