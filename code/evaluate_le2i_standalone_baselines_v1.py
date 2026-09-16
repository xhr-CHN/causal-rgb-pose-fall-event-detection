#!/usr/bin/env python3
"""Evaluate standalone Le2i baseline predictions with the locked event policy."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import external_fusion_ablation_urfd_le2i_v1 as evaluator


ROOT = Path("/home/data/yoloA27")
PREDICTION_DIR = ROOT / "experiments/le2i_standalone_baselines_v2"
OUTPUT_DIR = ROOT / "experiments/le2i_standalone_baselines_v2"
INVENTORY = ROOT / "experiments/le2i_inventory_v2/sequence_inventory.csv"
EVENTS = ROOT / "experiments/le2i_inventory_v2/fall_events.csv"
POLICY = ROOT / "experiments/w16_sequential_evidence_policy_source_v1/selected_policy.json"


def evaluate_one(name: str, path: Path) -> dict:
    rows = evaluator.read_csv(path)
    sequences, events, _, negative_hours = evaluator.load_le2i_ground_truth(
        rows, INVENTORY, EVENTS
    )
    probabilities = np.asarray(
        [
            [
                float(row["prob_adl"]),
                float(row["prob_falling"]),
                float(row["prob_fallen"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    policy = evaluator.load_policy(POLICY)
    alarms = evaluator.generate_candidate_alarms(
        rows, probabilities, policy, sequences
    )
    metrics, event_rows, classified = evaluator.evaluate_events(
        alarms, sequences, events, negative_hours
    )
    return {
        "baseline": name,
        "prediction_file": str(path),
        "windows": len(rows),
        "sequences": len(sequences),
        "metrics": metrics,
        "alarms": len(alarms),
        "event_rows": event_rows,
        "classified_alarms": classified,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [
        evaluate_one(
            "pose_tcn",
            PREDICTION_DIR / "pose_tcn_frame_predictions.csv",
        ),
        evaluate_one(
            "rgb_tcn",
            PREDICTION_DIR / "rgb_tcn_frame_predictions.csv",
        ),
    ]
    summary = {
        "protocol": {
            "dataset": "Le2i",
            "evaluation": "same locked causal sequential evidence policy",
            "labels_used_only_for_evaluation": True,
            "inventory": str(INVENTORY),
            "events": str(EVENTS),
            "policy": str(POLICY),
        },
        "baselines": results,
    }
    (OUTPUT_DIR / "event_level_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = [
        "baseline",
        "windows",
        "sequences",
        "total_fall_events",
        "detected_fall_events",
        "event_recall",
        "candidate_alarm_count",
        "false_alarm_count",
        "negative_exposure_hours",
        "false_alarms_per_hour",
        "duplicate_alarms_after_onset",
        "mean_detection_delay_ms",
    ]
    with (OUTPUT_DIR / "event_level_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {
                "baseline": result["baseline"],
                "windows": result["windows"],
                "sequences": result["sequences"],
            }
            row.update(result["metrics"])
            writer.writerow({field: row.get(field) for field in fields})
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
