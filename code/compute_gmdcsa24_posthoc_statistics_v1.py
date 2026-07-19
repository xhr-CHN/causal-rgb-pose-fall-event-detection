#!/usr/bin/env python3
"""Compute post-hoc uncertainty for the frozen GMDCSA24 blind evaluation.

This script never changes a model, threshold, alarm policy, prediction, or
label. It reads the completed one-time evaluation and reports Wilson intervals,
exact Poisson rate intervals, event-delay bootstrap intervals, and paired
stage comparisons at the complete-video level.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/data/yoloA27")
INPUT_DIR = ROOT / "experiments/gmdcsa24_final_blind_evaluation_v1"
EVENT_RESULTS = INPUT_DIR / "event_results.csv"
EVALUATION_SUMMARY = INPUT_DIR / "final_evaluation_summary.json"
OUTPUT_DIR = ROOT / "experiments/gmdcsa24_posthoc_statistics_v1"

STAGES = ("base_candidate", "primary_verifier", "final_causal_rescue")
COMPARISONS = (
    ("base_candidate", "primary_verifier"),
    ("base_candidate", "final_causal_rescue"),
    ("primary_verifier", "final_causal_rescue"),
)
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260718
ALPHA = 0.05


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        fail(f"Empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        fail(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def poisson_cdf(k: int, mean: float) -> float:
    if k < 0:
        return 0.0
    if mean == 0.0:
        return 1.0
    term = math.exp(-mean)
    total = term
    for value in range(1, k + 1):
        term *= mean / value
        total += term
    return min(1.0, max(0.0, total))


def invert_poisson_cdf(k: int, target: float) -> float:
    low = 0.0
    high = max(1.0, float(k + 1))
    while poisson_cdf(k, high) > target:
        high *= 2.0
        if high > 1e6:
            fail("Poisson interval bracketing failed")
    for _ in range(120):
        middle = (low + high) / 2.0
        if poisson_cdf(k, middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def poisson_rate_interval(
    count: int, exposure_hours: float
) -> tuple[float, float]:
    if exposure_hours <= 0.0:
        return math.nan, math.nan
    lower_mean = (
        0.0
        if count == 0
        else invert_poisson_cdf(count - 1, 1.0 - ALPHA / 2.0)
    )
    upper_mean = invert_poisson_cdf(count, ALPHA / 2.0)
    return lower_mean / exposure_hours, upper_mean / exposure_hours


def exact_two_sided_binomial_p(first: int, second: int) -> float:
    total = first + second
    if total == 0:
        return 1.0
    smaller = min(first, second)
    tail = sum(math.comb(total, value) for value in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2.0**total))


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return (
        float(np.quantile(values, ALPHA / 2.0)),
        float(np.quantile(values, 1.0 - ALPHA / 2.0)),
    )


def bootstrap_delay(
    delays: np.ndarray, rng: np.random.Generator
) -> dict[str, object]:
    if not len(delays):
        return {
            "mean_ms": math.nan,
            "mean_95ci_ms": [math.nan, math.nan],
            "median_ms": math.nan,
            "median_95ci_ms": [math.nan, math.nan],
        }
    sampled_means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    sampled_medians = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sample = delays[rng.integers(0, len(delays), len(delays))]
        sampled_means[replicate] = sample.mean()
        sampled_medians[replicate] = np.median(sample)
    return {
        "mean_ms": float(delays.mean()),
        "mean_95ci_ms": list(percentile_interval(sampled_means)),
        "median_ms": float(np.median(delays)),
        "median_95ci_ms": list(percentile_interval(sampled_medians)),
    }


def stage_rows_by_sequence(
    rows: list[dict[str, str]], stage: str
) -> dict[str, dict[str, str]]:
    selected = [row for row in rows if row["stage"] == stage]
    mapping = {row["sequence_id"]: row for row in selected}
    if len(selected) != 160 or len(mapping) != 160:
        fail(f"Expected 160 unique sequences for {stage}")
    return mapping


def main() -> None:
    for path in (EVENT_RESULTS, EVALUATION_SUMMARY):
        if not path.is_file():
            fail(f"Required frozen evaluation artifact missing: {path}")
    if OUTPUT_DIR.exists():
        fail(f"Output already exists; refusing overwrite: {OUTPUT_DIR}")

    rows = read_csv(EVENT_RESULTS)
    source_summary = json.loads(EVALUATION_SUMMARY.read_text(encoding="utf-8"))
    if set(row["stage"] for row in rows) != set(STAGES):
        fail("Unexpected GMDCSA24 evaluation stages")
    if len(rows) != 160 * len(STAGES):
        fail("Expected exactly 480 stage-by-sequence rows")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    stage_output_rows = []
    stage_details = {}
    mappings = {}

    for stage in STAGES:
        mapping = stage_rows_by_sequence(rows, stage)
        mappings[stage] = mapping
        stage_rows = list(mapping.values())
        fall_rows = [row for row in stage_rows if row["category"] == "Fall"]
        adl_rows = [row for row in stage_rows if row["category"] == "ADL"]
        if len(fall_rows) != 79 or len(adl_rows) != 81:
            fail(f"Unexpected Fall/ADL counts for {stage}")

        detected = sum(int(row["detected"]) for row in fall_rows)
        false_alarms = sum(
            int(row["false_alarms_before_onset_or_in_adl"])
            for row in stage_rows
        )
        duplicates = sum(
            int(row["duplicate_alarms_after_detection"])
            for row in stage_rows
        )
        delays = np.array(
            [
                float(row["detection_delay_ms"])
                for row in fall_rows
                if row["detected"] == "1"
            ],
            dtype=np.float64,
        )

        frozen = source_summary["event_level"][stage]
        exposure = float(frozen["negative_exposure_hours"])
        if detected != int(frozen["detected_fall_events"]):
            fail(f"Detected-event count mismatch for {stage}")
        if false_alarms != int(frozen["false_alarm_count"]):
            fail(f"False-alarm count mismatch for {stage}")
        if duplicates != int(frozen["duplicate_alarms_after_event_detection"]):
            fail(f"Duplicate-alarm count mismatch for {stage}")

        recall_interval = wilson_interval(detected, len(fall_rows))
        rate_interval = poisson_rate_interval(false_alarms, exposure)
        precision_total = detected + false_alarms
        precision_interval = wilson_interval(detected, precision_total)
        delay_summary = bootstrap_delay(delays, rng)

        detail = {
            "fall_events": len(fall_rows),
            "detected_events": detected,
            "event_recall": divide(detected, len(fall_rows)),
            "event_recall_wilson_95ci": list(recall_interval),
            "false_alarm_count": false_alarms,
            "negative_exposure_hours": exposure,
            "false_alarms_per_hour": divide(false_alarms, exposure),
            "false_alarms_per_hour_exact_poisson_95ci": list(rate_interval),
            "event_alarm_precision_excluding_duplicates": divide(
                detected, precision_total
            ),
            "event_alarm_precision_wilson_95ci": list(precision_interval),
            "duplicate_alarms": duplicates,
            "delay_bootstrap": delay_summary,
        }
        stage_details[stage] = detail
        stage_output_rows.append(
            {
                "stage": stage,
                "fall_events": len(fall_rows),
                "detected_events": detected,
                "event_recall": detail["event_recall"],
                "event_recall_ci_low": recall_interval[0],
                "event_recall_ci_high": recall_interval[1],
                "false_alarm_count": false_alarms,
                "negative_exposure_hours": exposure,
                "false_alarms_per_hour": detail["false_alarms_per_hour"],
                "false_alarms_per_hour_ci_low": rate_interval[0],
                "false_alarms_per_hour_ci_high": rate_interval[1],
                "event_alarm_precision": detail[
                    "event_alarm_precision_excluding_duplicates"
                ],
                "event_alarm_precision_ci_low": precision_interval[0],
                "event_alarm_precision_ci_high": precision_interval[1],
                "median_delay_ms": delay_summary["median_ms"],
                "median_delay_ci_low_ms": delay_summary["median_95ci_ms"][0],
                "median_delay_ci_high_ms": delay_summary["median_95ci_ms"][1],
            }
        )

    comparison_rows = []
    comparison_details = {}
    all_sequences = sorted(mappings[STAGES[0]])
    fall_sequences = [
        sequence
        for sequence in all_sequences
        if mappings[STAGES[0]][sequence]["category"] == "Fall"
    ]

    for reference, target in COMPARISONS:
        reference_detected = np.array(
            [int(mappings[reference][s]["detected"]) for s in fall_sequences],
            dtype=np.int64,
        )
        target_detected = np.array(
            [int(mappings[target][s]["detected"]) for s in fall_sequences],
            dtype=np.int64,
        )
        reference_only = int(
            np.sum((reference_detected == 1) & (target_detected == 0))
        )
        target_only = int(
            np.sum((reference_detected == 0) & (target_detected == 1))
        )
        both_detected = int(
            np.sum((reference_detected == 1) & (target_detected == 1))
        )
        neither_detected = int(
            np.sum((reference_detected == 0) & (target_detected == 0))
        )

        event_differences = target_detected - reference_detected
        event_bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        for replicate in range(BOOTSTRAP_REPLICATES):
            indices = rng.integers(0, len(fall_sequences), len(fall_sequences))
            event_bootstrap[replicate] = event_differences[indices].mean()
        event_difference_ci = percentile_interval(event_bootstrap)

        reference_false = np.array(
            [
                int(
                    mappings[reference][s][
                        "false_alarms_before_onset_or_in_adl"
                    ]
                )
                for s in all_sequences
            ],
            dtype=np.int64,
        )
        target_false = np.array(
            [
                int(
                    mappings[target][s][
                        "false_alarms_before_onset_or_in_adl"
                    ]
                )
                for s in all_sequences
            ],
            dtype=np.int64,
        )
        false_differences = target_false - reference_false
        false_bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        for replicate in range(BOOTSTRAP_REPLICATES):
            indices = rng.integers(0, len(all_sequences), len(all_sequences))
            false_bootstrap[replicate] = false_differences[indices].sum()
        false_difference_ci = percentile_interval(false_bootstrap)
        reference_more_false = int(np.sum(false_differences < 0))
        target_more_false = int(np.sum(false_differences > 0))

        name = f"{reference}_vs_{target}"
        detail = {
            "reference": reference,
            "target": target,
            "paired_event_detection": {
                "both_detected": both_detected,
                "reference_only_detected": reference_only,
                "target_only_detected": target_only,
                "neither_detected": neither_detected,
                "event_recall_difference_target_minus_reference": float(
                    event_differences.mean()
                ),
                "event_recall_difference_bootstrap_95ci": list(
                    event_difference_ci
                ),
                "exact_mcnemar_p": exact_two_sided_binomial_p(
                    reference_only, target_only
                ),
            },
            "paired_false_alarm_counts": {
                "reference_false_alarms": int(reference_false.sum()),
                "target_false_alarms": int(target_false.sum()),
                "total_difference_target_minus_reference": int(
                    false_differences.sum()
                ),
                "total_difference_bootstrap_95ci": list(false_difference_ci),
                "sequences_with_fewer_false_alarms_in_target": reference_more_false,
                "sequences_with_more_false_alarms_in_target": target_more_false,
                "exact_sign_test_p": exact_two_sided_binomial_p(
                    reference_more_false, target_more_false
                ),
            },
        }
        comparison_details[name] = detail
        comparison_rows.append(
            {
                "comparison": name,
                "event_recall_difference": float(event_differences.mean()),
                "event_difference_ci_low": event_difference_ci[0],
                "event_difference_ci_high": event_difference_ci[1],
                "mcnemar_p": detail["paired_event_detection"][
                    "exact_mcnemar_p"
                ],
                "false_alarm_count_difference": int(false_differences.sum()),
                "false_alarm_difference_ci_low": false_difference_ci[0],
                "false_alarm_difference_ci_high": false_difference_ci[1],
                "false_alarm_sign_test_p": detail[
                    "paired_false_alarm_counts"
                ]["exact_sign_test_p"],
            }
        )

    OUTPUT_DIR.mkdir(parents=True)
    stage_path = OUTPUT_DIR / "stage_intervals.csv"
    comparison_path = OUTPUT_DIR / "paired_comparisons.csv"
    write_csv(stage_path, stage_output_rows)
    write_csv(comparison_path, comparison_rows)

    summary = {
        "protocol": {
            "dataset": "GMDCSA24 v2.1",
            "role": "post-hoc statistics on frozen one-time blind evaluation",
            "model_or_policy_changed": False,
            "predictions_changed": False,
            "bootstrap_unit": "complete video sequence",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage_intervals": stage_details,
        "paired_comparisons": comparison_details,
        "input_sha256": {
            str(EVENT_RESULTS): sha256(EVENT_RESULTS),
            str(EVALUATION_SUMMARY): sha256(EVALUATION_SUMMARY),
        },
        "script_sha256": sha256(Path(__file__)),
    }
    summary_path = OUTPUT_DIR / "statistical_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "output_sha256.txt").write_text(
        f"{sha256(stage_path)}  stage_intervals.csv\n"
        f"{sha256(comparison_path)}  paired_comparisons.csv\n"
        f"{sha256(summary_path)}  statistical_summary.json\n",
        encoding="utf-8",
    )

    for row in stage_output_rows:
        print(
            f"{row['stage']}: recall={row['event_recall']:.4f} "
            f"[{row['event_recall_ci_low']:.4f}, "
            f"{row['event_recall_ci_high']:.4f}], "
            f"FA/h={row['false_alarms_per_hour']:.2f} "
            f"[{row['false_alarms_per_hour_ci_low']:.2f}, "
            f"{row['false_alarms_per_hour_ci_high']:.2f}]",
            flush=True,
        )
    print(f"Completed: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
