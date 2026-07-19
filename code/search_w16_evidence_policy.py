import csv
import json
from collections import defaultdict
from pathlib import Path

PREDICTIONS = Path(
    "/home/data/yoloA27/experiments/"
    "quality_gated_logit_fusion_v3_w16_fixed_seed42/"
    "val_predictions.csv"
)

TRAINING_SUMMARY = Path(
    "/home/data/yoloA27/experiments/"
    "quality_gated_logit_fusion_v3_w16_fixed_seed42/"
    "summary.json"
)

OUTPUT_DIR = Path(
    "/home/data/yoloA27/experiments/"
    "w16_sequential_evidence_policy_source_v1"
)

FPS = 20.0
RESET_THRESHOLD = 0.10
RESET_FRAMES = 5

ALPHAS = [0.25, 0.50, 0.65, 0.75, 0.80, 0.85, 0.90]
THRESHOLDS = [value / 100 for value in range(20, 96, 5)]
CONSECUTIVE_VALUES = range(1, 6)


def read_predictions():
    sequences = defaultdict(list)

    with PREDICTIONS.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        for row in csv.DictReader(file):
            falling = float(row["prob_falling_fusion_v3"])
            fallen = float(row["prob_fallen_fusion_v3"])

            row["frame"] = int(float(row["target_frame_index"]))
            row["fall_score"] = max(
                0.0, min(1.0, falling + fallen)
            )

            onset = row.get("onset_frame", "")
            row["onset"] = (
                int(float(onset)) if onset else None
            )

            sequences[row["sequence_id"]].append(row)

    for rows in sequences.values():
        rows.sort(key=lambda item: item["frame"])

    return sequences


def generate_alarms(
    rows,
    alpha,
    threshold,
    consecutive_required,
):
    evidence = 0.0
    above_count = 0
    below_count = 0
    latched = False
    alarms = []

    for row in rows:
        score = row["fall_score"]

        evidence = (
            alpha * evidence
            + (1.0 - alpha) * score
        )

        if not latched:
            if evidence >= threshold:
                above_count += 1
            else:
                above_count = 0

            if above_count >= consecutive_required:
                alarms.append(row["frame"])
                latched = True
                above_count = 0
                below_count = 0

        else:
            if evidence <= RESET_THRESHOLD:
                below_count += 1
            else:
                below_count = 0

            if below_count >= RESET_FRAMES:
                latched = False
                below_count = 0

    return alarms


def evaluate_policy(
    sequences,
    alpha,
    threshold,
    consecutive_required,
):
    detected = 0
    false_alarms = 0
    delays = []
    event_rows = []

    for sequence_id, rows in sequences.items():
        category = rows[0]["category"]
        onset = rows[0]["onset"]

        alarms = generate_alarms(
            rows,
            alpha,
            threshold,
            consecutive_required,
        )

        if category == "adl":
            false_alarms += len(alarms)
            continue

        early = [
            frame for frame in alarms
            if frame < onset
        ]

        after_onset = [
            frame for frame in alarms
            if frame >= onset
        ]

        false_alarms += len(early)

        if after_onset:
            detected += 1
            delay = after_onset[0] - onset
            delays.append(delay)
            first_alarm = after_onset[0]
        else:
            delay = None
            first_alarm = None

        event_rows.append({
            "sequence_id": sequence_id,
            "onset_frame": onset,
            "detected": int(bool(after_onset)),
            "first_alarm_frame": first_alarm,
            "delay_frames": delay,
            "delay_seconds": (
                delay / FPS if delay is not None else None
            ),
            "early_false_alarms": len(early),
        })

    mean_delay = (
        sum(delays) / len(delays)
        if delays else float("inf")
    )

    return {
        "detected_events": detected,
        "total_events": sum(
            rows[0]["category"] == "fall"
            for rows in sequences.values()
        ),
        "false_alarm_count": false_alarms,
        "mean_delay_frames": mean_delay,
        "mean_delay_seconds": mean_delay / FPS,
        "event_rows": event_rows,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sequences = read_predictions()
    search_rows = []

    for alpha in ALPHAS:
        for threshold in THRESHOLDS:
            for consecutive in CONSECUTIVE_VALUES:
                result = evaluate_policy(
                    sequences,
                    alpha,
                    threshold,
                    consecutive,
                )

                search_rows.append({
                    "alpha": alpha,
                    "threshold": threshold,
                    "consecutive_frames": consecutive,
                    "detected_events":
                        result["detected_events"],
                    "total_events":
                        result["total_events"],
                    "false_alarm_count":
                        result["false_alarm_count"],
                    "mean_delay_frames":
                        result["mean_delay_frames"],
                    "mean_delay_seconds":
                        result["mean_delay_seconds"],
                })

    zero_false_alarm = [
        row for row in search_rows
        if row["false_alarm_count"] == 0
    ]

    candidates = (
        zero_false_alarm
        if zero_false_alarm
        else search_rows
    )

    candidates.sort(
        key=lambda row: (
            -row["detected_events"],
            row["false_alarm_count"],
            row["mean_delay_frames"],
            row["consecutive_frames"],
        )
    )

    best = candidates[0]

    best_result = evaluate_policy(
        sequences,
        best["alpha"],
        best["threshold"],
        best["consecutive_frames"],
    )

    with TRAINING_SUMMARY.open(
        "r", encoding="utf-8"
    ) as file:
        training_summary = json.load(file)

    output = {
        "protocol": {
            "selection_dataset":
                "CAUCAFall validation only",
            "external_data_used_for_selection": False,
            "window_length": 16,
            "score":
                "P(Falling) + P(Fallen)",
            "evidence_update":
                "E_t = alpha*E_(t-1) + "
                "(1-alpha)*fall_score",
        },
        "selected_policy": {
            "alpha": best["alpha"],
            "alarm_threshold": best["threshold"],
            "consecutive_evidence_frames":
                best["consecutive_frames"],
            "reset_threshold": RESET_THRESHOLD,
            "reset_frames": RESET_FRAMES,
        },
        "source_validation_result": {
            "detected_events":
                best_result["detected_events"],
            "total_events":
                best_result["total_events"],
            "event_recall":
                best_result["detected_events"]
                / best_result["total_events"],
            "false_alarm_count":
                best_result["false_alarm_count"],
            "mean_delay_frames":
                best_result["mean_delay_frames"],
            "mean_delay_seconds":
                best_result["mean_delay_seconds"],
        },
        "original_state_machine_result":
            training_summary[
                "clean_validation_event_metrics"
            ],
    }

    with (
        OUTPUT_DIR / "selected_policy.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    with (
        OUTPUT_DIR / "policy_search.csv"
    ).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(search_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(search_rows)

    event_rows = best_result["event_rows"]

    with (
        OUTPUT_DIR / "source_event_results.csv"
    ).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(event_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(event_rows)

    print(json.dumps(output, indent=2))
    print("\nOutputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
