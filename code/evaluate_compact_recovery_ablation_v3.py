#!/usr/bin/env python3
"""Evaluate compact recovery-feature ablations with the frozen V2 CV folds.

This development-only script uses URFD and Le2i candidates. It reuses the
complete-sequence fold assignment from the V2 verifier, performs fold-local
imputation/standardization, and fits weighted L2 logistic regression using
NumPy only. GMDCSA24 and CAUCAFall are not used.
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


SEED = 42
ROOT = Path("/home/data/yoloA27")
INPUT_TABLE = (
    ROOT
    / "features/multidataset_alarm_recovery_features_v3/"
    "alarm_recovery_features.csv"
)
V2_OOF = (
    ROOT
    / "experiments/multidataset_alarm_verifier_v2_seed42/oof_predictions.csv"
)
OUTPUT_DIR = ROOT / "experiments/compact_recovery_ablation_v3_seed42"

L2_REGULARIZATION = 0.10
GLOBAL_MINIMUM_RECALL = 0.95
PER_DATASET_MINIMUM_RECALL = 0.90
MATERIAL_MACRO_FPR_REDUCTION = 0.03
MAXIMUM_ACCEPTABLE_AP_LOSS = 0.02

BASE_FEATURES = [
    "pre_person_conf_max",
    "pre_bbox_aspect_ratio_std",
    "pre_branch_disagreement_mean",
    "pre_bbox_cy_delta",
    "pre_bbox_aspect_ratio_slope",
    "pre_rgb_fall_score_delta",
    "pre_torso_keypoint_conf_max",
    "pre_bbox_height_delta",
    "confirm_fused_fall_score_mean",
    "confirm_fused_fall_score_min",
    "confirm_fused_fall_score_delta",
    "confirm_fall_support_fraction_70",
    "confirm_fall_longest_run_70",
    "confirm_modal_agreement_fraction_50",
    "confirm_recovery_drop_from_candidate",
    "confirm_fallen_rise",
    "confirm_branch_disagreement_mean",
    "confirm_pose_found_ratio_mean",
]


def terminal_features(tag: str) -> list[str]:
    return [
        f"{tag}_torso_horizontalness_last",
        f"{tag}_bbox_aspect_ratio_last",
        f"{tag}_bbox_cy_last",
        f"{tag}_hip_y_last",
    ]


def recovery_features(tag: str) -> list[str]:
    return terminal_features(tag) + [
        f"{tag}_torso_horizontalness_recovery",
        f"{tag}_bbox_aspect_ratio_recovery",
        f"{tag}_bbox_cy_recovery",
        f"{tag}_hip_y_recovery",
        f"{tag}_joint_speed_late_mean",
        f"{tag}_joint_speed_late_to_early_ratio",
        f"{tag}_static_transition_fraction",
        f"{tag}_pose_found_mean",
        f"{tag}_coverage_fraction",
        "history_750_coverage_fraction",
    ]


CONFIGURATIONS = {
    "v2_base_200ms": {
        "features": BASE_FEATURES,
        "maximum_confirmation_ms": 200,
    },
    "v2_plus_terminal_500ms": {
        "features": BASE_FEATURES + terminal_features("h500"),
        "maximum_confirmation_ms": 500,
    },
    "v2_plus_terminal_1000ms": {
        "features": BASE_FEATURES + terminal_features("h1000"),
        "maximum_confirmation_ms": 1000,
    },
    "v2_plus_terminal_1500ms": {
        "features": BASE_FEATURES + terminal_features("h1500"),
        "maximum_confirmation_ms": 1500,
    },
    "v2_plus_recovery_1000ms": {
        "features": BASE_FEATURES + recovery_features("h1000"),
        "maximum_confirmation_ms": 1000,
    },
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        fail(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def prepare_fold(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    medians = np.zeros(x_train.shape[1], dtype=np.float64)
    for column in range(x_train.shape[1]):
        clean = x_train[np.isfinite(x_train[:, column]), column]
        medians[column] = float(np.median(clean)) if clean.size else 0.0
    missing_columns = np.any(~np.isfinite(x_train), axis=0) | np.any(
        ~np.isfinite(x_validation), axis=0
    )
    train_missing = (~np.isfinite(x_train[:, missing_columns])).astype(np.float64)
    validation_missing = (
        ~np.isfinite(x_validation[:, missing_columns])
    ).astype(np.float64)
    train_filled = np.where(np.isfinite(x_train), x_train, medians)
    validation_filled = np.where(
        np.isfinite(x_validation), x_validation, medians
    )
    if np.any(missing_columns):
        train_filled = np.column_stack([train_filled, train_missing])
        validation_filled = np.column_stack(
            [validation_filled, validation_missing]
        )
    weight_sum = float(sample_weights.sum())
    means = (
        train_filled * sample_weights[:, None]
    ).sum(axis=0) / weight_sum
    variances = (
        (train_filled - means) ** 2 * sample_weights[:, None]
    ).sum(axis=0) / weight_sum
    scales = np.sqrt(np.maximum(variances, 0.0))
    scales[scales < 1e-8] = 1.0
    train_scaled = (train_filled - means) / scales
    validation_scaled = (validation_filled - means) / scales
    return train_scaled, validation_scaled, {
        "medians": medians.tolist(),
        "missing_indicator_columns": np.where(missing_columns)[0].tolist(),
        "means": means.tolist(),
        "scales": scales.tolist(),
    }


def cell_balanced_weights(
    datasets: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    counts = Counter(zip(datasets.tolist(), labels.tolist()))
    weights = np.array(
        [1.0 / counts[(dataset, int(label))] for dataset, label in zip(datasets, labels)],
        dtype=np.float64,
    )
    return weights / weights.mean()


def fit_logistic_newton(
    x: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    l2_regularization: float,
    maximum_iterations: int = 100,
) -> tuple[np.ndarray, int]:
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.concatenate(
        [[0.0], np.full(x.shape[1], l2_regularization)]
    )
    weight_sum = float(sample_weights.sum())
    for iteration in range(1, maximum_iterations + 1):
        probabilities = stable_sigmoid(design @ coefficients)
        curvature = np.clip(
            probabilities * (1.0 - probabilities), 1e-7, None
        )
        gradient = (
            design.T @ (sample_weights * (probabilities - y)) / weight_sum
            + penalty * coefficients
        )
        weighted_curvature = sample_weights * curvature
        hessian = (
            design.T @ (design * weighted_curvature[:, None]) / weight_sum
            + np.diag(penalty)
            + np.eye(design.shape[1]) * 1e-10
        )
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            update = np.linalg.pinv(hessian) @ gradient
        coefficients -= update
        if np.max(np.abs(update)) < 1e-10:
            return coefficients, iteration
    return coefficients, maximum_iterations


def predict_probability(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return stable_sigmoid(design @ coefficients)


def auc_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if not len(positives) or not len(negatives):
        return math.nan
    comparisons = 0.0
    for positive in positives:
        comparisons += np.sum(positive > negatives)
        comparisons += 0.5 * np.sum(positive == negatives)
    return float(comparisons / (len(positives) * len(negatives)))


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    positive_total = int(np.sum(sorted_labels == 1))
    if positive_total == 0:
        return math.nan
    true_positives = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, len(sorted_labels) + 1)
    return float(np.sum((true_positives / ranks) * (sorted_labels == 1)) / positive_total)


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "f1": 2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def evaluate_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    datasets: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    output: dict[str, object] = {
        "overall": binary_metrics(labels, predictions),
        "by_dataset": {},
    }
    for dataset in sorted(set(datasets.tolist())):
        selected = datasets == dataset
        metrics = binary_metrics(labels[selected], predictions[selected])
        metrics["candidates"] = int(np.sum(selected))
        output["by_dataset"][dataset] = metrics
    output["macro_fpr"] = float(
        np.mean(
            [
                value["false_positive_rate"]
                for value in output["by_dataset"].values()
            ]
        )
    )
    output["auroc"] = auc_score(labels, probabilities)
    output["average_precision"] = average_precision(labels, probabilities)
    output["macro_average_precision"] = float(
        np.mean(
            [
                average_precision(
                    labels[datasets == dataset], probabilities[datasets == dataset]
                )
                for dataset in sorted(set(datasets.tolist()))
            ]
        )
    )
    return output


def select_threshold(
    labels: np.ndarray, probabilities: np.ndarray, datasets: np.ndarray
) -> tuple[float, dict[str, object]]:
    unique = np.unique(probabilities)
    candidates = [0.0]
    candidates.extend(
        float((left + right) / 2.0)
        for left, right in zip(unique[:-1], unique[1:])
    )
    candidates.append(1.0)
    eligible: list[tuple[float, dict[str, object]]] = []
    for threshold in candidates:
        metrics = evaluate_threshold(labels, probabilities, datasets, threshold)
        recalls = [
            value["recall_sensitivity"]
            for value in metrics["by_dataset"].values()
        ]
        if (
            metrics["overall"]["recall_sensitivity"] >= GLOBAL_MINIMUM_RECALL
            and min(recalls) >= PER_DATASET_MINIMUM_RECALL
        ):
            eligible.append((threshold, metrics))
    if not eligible:
        fail("No threshold satisfies the recall constraints")

    def threshold_margin(threshold: float) -> float:
        below = probabilities[probabilities < threshold]
        above = probabilities[probabilities >= threshold]
        if len(below) == 0 or len(above) == 0:
            return 0.0
        return float(above.min() - below.max())

    threshold, metrics = min(
        eligible,
        key=lambda item: (
            float(item[1]["macro_fpr"]),
            -float(item[1]["macro_average_precision"]),
            -float(item[1]["overall"]["recall_sensitivity"]),
            -threshold_margin(item[0]),
            -item[0],
        ),
    )
    metrics["threshold_margin"] = threshold_margin(threshold)
    return float(threshold), metrics


def main() -> None:
    for path in (INPUT_TABLE, V2_OOF):
        if not path.is_file():
            fail(f"Required input missing: {path}")
    if OUTPUT_DIR.exists():
        fail(f"Output already exists; refusing overwrite: {OUTPUT_DIR}")

    rows = read_csv(INPUT_TABLE)
    fold_rows = read_csv(V2_OOF)
    fold_by_key = {row["candidate_key"]: int(row["fold"]) for row in fold_rows}
    if len(rows) != 281 or len(fold_by_key) != len(rows):
        fail("Expected 281 candidates and complete V2 fold mapping")
    candidate_keys = [row["candidate_key"] for row in rows]
    if set(candidate_keys) != set(fold_by_key):
        fail("Recovery table and V2 OOF candidate keys do not match")

    labels = np.array([int(row["label_true_alarm"]) for row in rows])
    datasets = np.array([row["dataset"] for row in rows])
    sequences = np.array([row["sequence_id"] for row in rows])
    folds = np.array([fold_by_key[key] for key in candidate_keys])
    if set(datasets.tolist()) != {"Le2i", "URFD"}:
        fail("Only Le2i and URFD are allowed in this development ablation")
    if len(set(folds.tolist())) != 5:
        fail("Expected five frozen V2 folds")
    for sequence in set(sequences.tolist()):
        if len(set(folds[sequences == sequence].tolist())) != 1:
            fail(f"Sequence leakage across folds: {sequence}")

    all_input_columns = set(rows[0])
    for name, configuration in CONFIGURATIONS.items():
        missing = set(configuration["features"]).difference(all_input_columns)
        if missing:
            fail(f"{name} missing features: {sorted(missing)}")

    comparison_rows: list[dict[str, object]] = []
    configuration_results: dict[str, object] = {}
    oof_output_rows: list[dict[str, object]] = [
        {
            "candidate_key": key,
            "dataset": dataset,
            "sequence_id": sequence,
            "label_true_alarm": int(label),
            "fold": int(fold),
        }
        for key, dataset, sequence, label, fold in zip(
            candidate_keys, datasets, sequences, labels, folds
        )
    ]

    for configuration_name, configuration in CONFIGURATIONS.items():
        feature_names = configuration["features"]
        matrix = np.array(
            [[numeric(row[name]) for name in feature_names] for row in rows],
            dtype=np.float64,
        )
        probabilities = np.zeros(len(rows), dtype=np.float64)
        fold_details = []
        for fold in sorted(set(folds.tolist())):
            validation_indices = np.where(folds == fold)[0]
            training_indices = np.where(folds != fold)[0]
            sample_weights = cell_balanced_weights(
                datasets[training_indices], labels[training_indices]
            )
            train_x, validation_x, _ = prepare_fold(
                matrix[training_indices],
                matrix[validation_indices],
                sample_weights,
            )
            coefficients, iterations = fit_logistic_newton(
                train_x,
                labels[training_indices].astype(np.float64),
                sample_weights,
                L2_REGULARIZATION,
            )
            probabilities[validation_indices] = predict_probability(
                validation_x, coefficients
            )
            fold_details.append(
                {
                    "fold": int(fold),
                    "train_candidates": len(training_indices),
                    "validation_candidates": len(validation_indices),
                    "newton_iterations": iterations,
                }
            )
        threshold, metrics = select_threshold(labels, probabilities, datasets)
        selected = {
            "l2_regularization": L2_REGULARIZATION,
            "selected_threshold": threshold,
            "metrics": metrics,
            "probabilities": probabilities,
            "folds": fold_details,
        }
        predictions = (
            selected["probabilities"] >= selected["selected_threshold"]
        ).astype(np.int64)
        for index, (probability, prediction) in enumerate(
            zip(selected["probabilities"], predictions)
        ):
            oof_output_rows[index][
                f"{configuration_name}_oof_probability"
            ] = probability
            oof_output_rows[index][
                f"{configuration_name}_oof_prediction"
            ] = int(prediction)

        result = {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "maximum_confirmation_ms": configuration[
                "maximum_confirmation_ms"
            ],
            "l2_regularization": selected["l2_regularization"],
            "selected_threshold": selected["selected_threshold"],
            "grouped_oof": selected["metrics"],
            "folds": selected["folds"],
        }
        configuration_results[configuration_name] = result
        comparison_rows.append(
            {
                "configuration": configuration_name,
                "feature_count": len(feature_names),
                "maximum_confirmation_ms": configuration[
                    "maximum_confirmation_ms"
                ],
                "l2_regularization": selected["l2_regularization"],
                "threshold": selected["selected_threshold"],
                "recall": selected["metrics"]["overall"][
                    "recall_sensitivity"
                ],
                "precision": selected["metrics"]["overall"]["precision"],
                "macro_fpr": selected["metrics"]["macro_fpr"],
                "auroc": selected["metrics"]["auroc"],
                "average_precision": selected["metrics"]["average_precision"],
                "macro_average_precision": selected["metrics"][
                    "macro_average_precision"
                ],
                "le2i_recall": selected["metrics"]["by_dataset"]["Le2i"][
                    "recall_sensitivity"
                ],
                "urfd_recall": selected["metrics"]["by_dataset"]["URFD"][
                    "recall_sensitivity"
                ],
            }
        )
        print(
            f"{configuration_name}: macro FPR="
            f"{selected['metrics']['macro_fpr']:.4f}, "
            f"recall={selected['metrics']['overall']['recall_sensitivity']:.4f}",
            flush=True,
        )

    baseline = configuration_results["v2_base_200ms"]
    alternatives = [
        (name, result)
        for name, result in configuration_results.items()
        if name != "v2_base_200ms"
    ]
    best_alternative_name, best_alternative = min(
        alternatives,
        key=lambda item: (
            item[1]["grouped_oof"]["macro_fpr"],
            -item[1]["grouped_oof"]["macro_average_precision"],
            item[1]["maximum_confirmation_ms"],
        ),
    )
    fpr_reduction = (
        baseline["grouped_oof"]["macro_fpr"]
        - best_alternative["grouped_oof"]["macro_fpr"]
    )
    ap_change = (
        best_alternative["grouped_oof"]["macro_average_precision"]
        - baseline["grouped_oof"]["macro_average_precision"]
    )
    promote = (
        fpr_reduction >= MATERIAL_MACRO_FPR_REDUCTION
        and ap_change >= -MAXIMUM_ACCEPTABLE_AP_LOSS
    )
    recommendation = {
        "best_recovery_configuration": best_alternative_name,
        "macro_fpr_reduction_vs_base": fpr_reduction,
        "macro_average_precision_change_vs_base": ap_change,
        "material_improvement_required": MATERIAL_MACRO_FPR_REDUCTION,
        "promote_recovery_v3": promote,
        "decision": (
            "Promote compact recovery V3 for further external testing."
            if promote
            else "Do not promote V3; retain recovery features as a negative/diagnostic ablation."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True)
    write_csv(OUTPUT_DIR / "comparison.csv", comparison_rows)
    write_csv(OUTPUT_DIR / "oof_predictions.csv", oof_output_rows)
    summary = {
        "protocol": {
            "development_datasets": ["URFD", "Le2i"],
            "gmdcsa24_used": False,
            "caucafall_used": False,
            "cross_validation": "frozen V2 five-fold complete-sequence grouped CV",
            "fold_local_preprocessing": True,
            "sample_weighting": "equal total weight per dataset x class cell",
            "global_minimum_recall": GLOBAL_MINIMUM_RECALL,
            "per_dataset_minimum_recall": PER_DATASET_MINIMUM_RECALL,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "candidate_count": len(rows),
        "positive_candidates": int(np.sum(labels == 1)),
        "negative_candidates": int(np.sum(labels == 0)),
        "sequence_groups": len(set(sequences.tolist())),
        "dataset_counts": dict(Counter(datasets.tolist())),
        "configurations": configuration_results,
        "recommendation": recommendation,
        "input_sha256": sha256(INPUT_TABLE),
        "v2_oof_sha256": sha256(V2_OOF),
        "script_sha256": sha256(Path(__file__)),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "output_sha256.txt").write_text(
        f"{sha256(OUTPUT_DIR / 'summary.json')}  summary.json\n"
        f"{sha256(OUTPUT_DIR / 'comparison.csv')}  comparison.csv\n"
        f"{sha256(OUTPUT_DIR / 'oof_predictions.csv')}  oof_predictions.csv\n",
        encoding="utf-8",
    )
    print(json.dumps(recommendation, ensure_ascii=False, indent=2), flush=True)
    print(f"Completed: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
