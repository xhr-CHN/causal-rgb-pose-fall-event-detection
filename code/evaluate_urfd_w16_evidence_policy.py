import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/home/data/yoloA27")

PREDICTIONS = (
    ROOT
    / "experiments/urfd_quality_gated_logit_fusion_v3_w16_external/"
    / "frame_predictions.csv"
)

EVENTS = (
    ROOT
    / "experiments/urfd_quality_gated_logit_fusion_v3_w16_external/"
    / "event_results.csv"
)

ORIGINAL_SUMMARY = (
    ROOT
    / "experiments/urfd_quality_gated_logit_fusion_v3_w16_external/"
    / "summary.json"
)

POLICY_PATH = (
    ROOT
    / "experiments/w16_sequential_evidence_policy_source_v1/"
    / "selected_policy.json"
)

OUTPUT_DIR = (
    ROOT
    / "experiments/urfd_w16_sequential_evidence_policy_v1"
)

FPS = 20.0


def read_csv(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames):
    with path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def generate_alarms(rows, policy):
    alpha = float(policy["alpha"])
    threshold = float(policy["alarm_threshold"])
    required = int(
        policy["consecutive_evidence_frames"]
    )
    reset_threshold = float(
        policy["reset_threshold"]
    )
    reset_frames = int(policy["reset_frames"])

    evidence = 0.0
    above_count = 0
    below_count = 0
    latched = False
    alarms = []

    for row in rows:
        falling = float(row["prob_falling"])
        fallen = float(row["prob_fallen"])

        fall_score = max(
            0.0,
            min(1.0, falling + fallen),
        )

        evidence = (
            alpha * evidence
            + (1.0 - alpha) * fall_score
        )

        if not latched:
            if evidence >= threshold:
                above_count += 1
            else:
                above_count = 0

            if above_count >= required:
                alarms.append({
                    "sequence_id": row["sequence_id"],
                    "category": row["category"],
                    "sample_index":
                        int(float(row["sample_index"])),
                    "sample_timestamp_ms":
                        int(float(row["sample_timestamp_ms"])),
                    "source_frame_number":
                        int(float(row["source_frame_number"])),
                    "source_timestamp_ms":
                        int(float(row["source_timestamp_ms"])),
                    "fall_score": fall_score,
                    "accumulated_evidence": evidence,
                    "pose_gate_weight":
                        float(row["pose_gate_weight"]),
                    "branch_disagreement":
                        float(row["branch_disagreement"]),
                })

                latched = True
                above_count = 0
                below_count = 0

        else:
            if evidence <= reset_threshold:
                below_count += 1
            else:
                below_count = 0

            if below_count >= reset_frames:
                latched = False
                below_count = 0

    return alarms


def main():
    required_files = [
        PREDICTIONS,
        EVENTS,
        ORIGINAL_SUMMARY,
        POLICY_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with POLICY_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        policy_document = json.load(file)

    policy = policy_document["selected_policy"]

    with ORIGINAL_SUMMARY.open(
        "r", encoding="utf-8"
    ) as file:
        original_summary = json.load(file)

    prediction_rows = read_csv(PREDICTIONS)
    event_rows = read_csv(EVENTS)

    sequences = defaultdict(list)

    for row in prediction_rows:
        sequences[row["sequence_id"]].append(row)

    for rows in sequences.values():
        rows.sort(
            key=lambda row: int(
                float(row["sample_index"])
            )
        )

    event_metadata = {
        row["sequence_id"]: row
        for row in event_rows
    }

    output_events = []
    output_alarms = []

    detected_events = 0
    false_alarm_count = 0
    duplicate_alarms = 0
    delays = []

    for sequence_id, rows in sequences.items():
        category = rows[0]["category"]
        alarms = generate_alarms(rows, policy)

        if category == "adl":
            false_alarm_count += len(alarms)

            for alarm in alarms:
                alarm["alarm_type"] = "false_alarm_adl"
                output_alarms.append(alarm)

            continue

        if sequence_id not in event_metadata:
            raise KeyError(
                f"Missing event metadata: {sequence_id}"
            )

        metadata = event_metadata[sequence_id]
        onset_timestamp = int(
            float(metadata["onset_timestamp_ms"])
        )

        early_alarms = [
            alarm for alarm in alarms
            if alarm["source_timestamp_ms"]
            < onset_timestamp
        ]

        event_alarms = [
            alarm for alarm in alarms
            if alarm["source_timestamp_ms"]
            >= onset_timestamp
        ]

        false_alarm_count += len(early_alarms)

        for alarm in early_alarms:
            alarm["alarm_type"] = "false_alarm_early"
            output_alarms.append(alarm)

        if event_alarms:
            detected_events += 1

            first_alarm = event_alarms[0]
            delay = (
                first_alarm["source_timestamp_ms"]
                - onset_timestamp
            )
            delays.append(delay)

            duplicates = max(
                0, len(event_alarms) - 1
            )
            duplicate_alarms += duplicates

            for index, alarm in enumerate(event_alarms):
                alarm["alarm_type"] = (
                    "event_alarm"
                    if index == 0
                    else "duplicate_event_alarm"
                )
                output_alarms.append(alarm)

            first_alarm_timestamp = (
                first_alarm["source_timestamp_ms"]
            )
            first_alarm_frame = (
                first_alarm["source_frame_number"]
            )
        else:
            delay = None
            first_alarm_timestamp = None
            first_alarm_frame = None
            duplicates = 0

        output_events.append({
            "sequence_id": sequence_id,
            "onset_timestamp_ms": onset_timestamp,
            "detected": int(bool(event_alarms)),
            "first_alarm_source_frame":
                first_alarm_frame,
            "first_alarm_timestamp_ms":
                first_alarm_timestamp,
            "detection_delay_ms": delay,
            "early_false_alarms":
                len(early_alarms),
            "alarms_after_onset":
                len(event_alarms),
            "duplicate_alarms":
                duplicates,
        })

    total_events = len(event_metadata)
    missed_events = total_events - detected_events

    negative_windows = sum(
        int(float(row["gt_fall"])) == 0
        for row in prediction_rows
    )

    negative_exposure_hours = (
        negative_windows / FPS / 3600.0
    )

    if delays:
        mean_delay = float(np.mean(delays))
        median_delay = float(np.median(delays))
        p95_delay = float(np.percentile(delays, 95))
    else:
        mean_delay = None
        median_delay = None
        p95_delay = None

    result = {
        "protocol": {
            "source_training_dataset": "CAUCAFall",
            "external_development_dataset":
                "URFD Camera 0 RGB",
            "window_length": 16,
            "fine_tuning": False,
            "policy_selection_dataset":
                "CAUCAFall validation only",
            "urfd_used_for_policy_selection": False,
            "score":
                "P(Falling) + P(Fallen)",
            "policy_parameters_locked": True,
        },
        "locked_policy": policy,
        "event_level_sequential_evidence": {
            "total_fall_events": total_events,
            "detected_fall_events": detected_events,
            "missed_fall_events": missed_events,
            "event_recall":
                detected_events / total_events,
            "false_alarm_count": false_alarm_count,
            "negative_exposure_hours":
                negative_exposure_hours,
            "false_alarms_per_hour": (
                false_alarm_count
                / negative_exposure_hours
                if negative_exposure_hours > 0
                else None
            ),
            "duplicate_alarms_after_onset":
                duplicate_alarms,
            "mean_detection_delay_ms": mean_delay,
            "median_detection_delay_ms": median_delay,
            "p95_detection_delay_ms": p95_delay,
        },
        "original_locked_state_machine":
            original_summary[
                "event_level_locked_state_machine"
            ],
    }

    with (
        OUTPUT_DIR / "summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    write_csv(
        OUTPUT_DIR / "event_results.csv",
        output_events,
        [
            "sequence_id",
            "onset_timestamp_ms",
            "detected",
            "first_alarm_source_frame",
            "first_alarm_timestamp_ms",
            "detection_delay_ms",
            "early_false_alarms",
            "alarms_after_onset",
            "duplicate_alarms",
        ],
    )

    write_csv(
        OUTPUT_DIR / "alarm_results.csv",
        output_alarms,
        [
            "sequence_id",
            "category",
            "sample_index",
            "sample_timestamp_ms",
            "source_frame_number",
            "source_timestamp_ms",
            "fall_score",
            "accumulated_evidence",
            "pose_gate_weight",
            "branch_disagreement",
            "alarm_type",
        ],
    )

    print(json.dumps(result, indent=2))
    print("\nOutputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
