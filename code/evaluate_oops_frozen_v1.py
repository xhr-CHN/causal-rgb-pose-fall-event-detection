#!/usr/bin/env python3
"""Evaluate frozen CAUCAFall-to-OOPS predictions at event level."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import external_fusion_ablation_urfd_le2i_v1 as evaluator


ROOT = Path("/home/data/yoloA27")
OMNI_ROOT = Path("/home/data/omnifall_benchmark_2026_09")
PREDICTIONS = ROOT / "experiments/caucafall_to_oops_v1/frame_predictions.csv"
MANIFEST = OMNI_ROOT / "outputs/omnifall_oops_test_manifest_v2.csv"
POLICY = ROOT / "experiments/w16_sequential_evidence_policy_source_v1/selected_policy.json"
OUTPUT_DIR = ROOT / "experiments/caucafall_to_oops_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    for path in (PREDICTIONS, MANIFEST, POLICY):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = read_csv(PREDICTIONS)
    manifest = read_csv(MANIFEST)
    if len(manifest) != 572:
        raise RuntimeError(f"Expected 572 OOPS events, found {len(manifest)}")

    sequences = {}
    events = {}
    for row in manifest:
        sequence_id = row["omnifall_path"]
        onset_ms = float(row["event_start_sec"]) * 1000.0
        duration_ms = float(row["clip_duration_sec"]) * 1000.0
        sequences[sequence_id] = {
            "sequence_id": sequence_id,
            "scene": "OOPS",
            "is_fall": True,
            "negative_exposure_seconds": onset_ms / 1000.0,
            "duration_seconds": duration_ms / 1000.0,
        }
        events[sequence_id] = {
            "sequence_id": sequence_id,
            "onset_ms": onset_ms,
            "end_ms": duration_ms,
        }

    prediction_sequences = {row["sequence_id"] for row in rows}
    if prediction_sequences != set(sequences):
        missing = sorted(set(sequences) - prediction_sequences)
        extra = sorted(prediction_sequences - set(sequences))
        raise RuntimeError(f"Prediction/manifest mismatch: missing={missing[:5]}, extra={extra[:5]}")

    probabilities = np.asarray(
        [
            [float(row["prob_adl"]), float(row["prob_falling"]), float(row["prob_fallen"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    policy = evaluator.load_policy(POLICY)
    alarms = evaluator.generate_candidate_alarms(rows, probabilities, policy, sequences)
    negative_hours = sum(
        item["negative_exposure_seconds"] for item in sequences.values()
    ) / 3600.0
    metrics, event_rows, classified = evaluator.evaluate_events(
        alarms, sequences, events, negative_hours
    )

    summary = {
        "protocol": {
            "dataset": "OOPS fall-only test subset",
            "source_training_dataset": "CAUCAFall",
            "sequences": len(sequences),
            "all_sequences_are_fall_positive": True,
            "false_alarm_exposure": "pre-onset portions only; no ADL-only OOPS videos in this subset",
            "labels_read_during_inference": False,
            "policy": str(POLICY),
        },
        "metrics": metrics,
        "event_rows": event_rows,
        "classified_alarms": classified,
    }
    (OUTPUT_DIR / "event_level_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fields = [
        "total_fall_events", "detected_fall_events", "missed_fall_events",
        "event_recall", "candidate_alarm_count", "false_alarm_count",
        "negative_exposure_hours", "false_alarms_per_hour",
        "duplicate_alarms_after_onset", "mean_detection_delay_ms",
        "median_detection_delay_ms", "p95_detection_delay_ms",
    ]
    with (OUTPUT_DIR / "event_level_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: metrics.get(field, "") for field in fields})

    print(json.dumps({"protocol": summary["protocol"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
