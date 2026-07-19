#!/usr/bin/env python3
"""Train a compact causal alarm verifier on URFD + revealed Le2i.

Selection uses deterministic five-fold group CV with complete sequences kept
inside one fold. Sample weights balance dataset and class. GMDCSA24 is never
read. Three feature sets are compared under one predeclared operating rule:
global true-alarm recall >= 0.95 and per-dataset recall >= 0.90, then minimize
macro false-positive rate.
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

import numpy as np


ROOT = Path("/home/data/yoloA27")
INPUT = (
    ROOT
    / "features/multidataset_alarm_confirmation_features_v2/"
    "alarm_confirmation_features.csv"
)
OUTPUT_DIR = ROOT / "experiments/multidataset_alarm_verifier_v2_seed42"

SEED = 42
NUM_FOLDS = 5
L2_REGULARIZATION = 0.10
GLOBAL_TARGET_RECALL = 0.95
PER_DATASET_MINIMUM_RECALL = 0.90
MAX_NEWTON_ITERATIONS = 100
NEWTON_TOLERANCE = 1e-10
CONFIRMATION_FUTURE_SAMPLES = 4
MAXIMUM_ADDED_DELAY_MS = 200.0

PRE_V1_FEATURES = [
    "pre_person_conf_max",
    "pre_bbox_aspect_ratio_std",
    "pre_branch_disagreement_mean",
    "pre_bbox_cy_delta",
    "pre_bbox_aspect_ratio_slope",
    "pre_rgb_fall_score_delta",
    "pre_torso_keypoint_conf_max",
    "pre_bbox_height_delta",
]

CONFIRMATION_FEATURES = [
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

FEATURE_CONFIGS = {
    "pre_only_v1_compatible": PRE_V1_FEATURES,
    "confirmation_only_compact": CONFIRMATION_FEATURES,
    "pre_plus_confirmation_v2": PRE_V1_FEATURES + CONFIRMATION_FEATURES,
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
        fail(f"Input table is empty: {path}")
    return rows, fields


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        fail(f"Cannot write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    nonnegative = values >= 0.0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    output[~nonnegative] = exponential / (1.0 + exponential)
    return output


def cell_balanced_weights(dataset: np.ndarray, y: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(y), dtype=np.float64)
    cells = sorted({(str(d), int(label)) for d, label in zip(dataset, y)})
    expected_cells = len(set(dataset.tolist())) * 2
    if set(y.tolist()) != {0, 1} or len(cells) != expected_cells:
        fail(
            "Expected both classes inside every included dataset, "
            f"found cells={cells}"
        )
    for cell in cells:
        mask = (dataset == cell[0]) & (y == cell[1])
        count = int(mask.sum())
        if count == 0:
            fail(f"Empty dataset/class cell: {cell}")
        weights[mask] = 1.0 / (len(cells) * count)
    weights *= len(weights) / weights.sum()
    return weights


def weighted_standardize_fit(
    x: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weight_sum = float(weights.sum())
    mean = (x * weights[:, None]).sum(axis=0) / weight_sum
    variance = ((x - mean) ** 2 * weights[:, None]).sum(axis=0) / weight_sum
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-8] = 1.0
    return mean, scale


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    regularization: float = L2_REGULARIZATION,
) -> tuple[np.ndarray, int]:
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.concatenate([[0.0], np.full(x.shape[1], regularization)])
    weight_sum = float(weights.sum())

    for iteration in range(1, MAX_NEWTON_ITERATIONS + 1):
        probability = sigmoid(design @ beta)
        curvature = np.clip(probability * (1.0 - probability), 1e-7, None)
        gradient = (
            design.T @ (weights * (probability - y)) / weight_sum
            + penalty * beta
        )
        weighted_curvature = weights * curvature
        hessian = (
            design.T @ (design * weighted_curvature[:, None]) / weight_sum
            + np.diag(penalty)
            + np.eye(design.shape[1]) * 1e-10
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta -= step
        if float(np.max(np.abs(step))) < NEWTON_TOLERANCE:
            return beta, iteration
    return beta, MAX_NEWTON_ITERATIONS


def predict_probability(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    return sigmoid(design @ beta)


def divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    y = np.asarray(y, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    tp = int(np.sum((y == 1) & (prediction == 1)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    f1 = divide(2.0 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "false_positive_rate": divide(fp, fp + tn),
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    for value in np.unique(score):
        indices = np.flatnonzero(score == value)
        if len(indices) > 1:
            ranks[indices] = ranks[indices].mean()
    rank_sum = float(ranks[y == 1].sum())
    return float(
        (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    )


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
    ordered = y[order]
    positives = int(ordered.sum())
    if positives == 0:
        return float("nan")
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(y) + 1)
    return float(precision[ordered == 1].sum() / positives)


def evaluate(
    y: np.ndarray,
    score: np.ndarray,
    dataset: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    prediction = (score >= threshold).astype(np.int64)
    overall = confusion_metrics(y, prediction)
    overall["auroc"] = auroc(y, score)
    overall["average_precision"] = average_precision(y, score)
    by_dataset: dict[str, dict[str, object]] = {}
    for name in sorted(set(dataset.tolist())):
        mask = dataset == name
        metrics = confusion_metrics(y[mask], prediction[mask])
        metrics["auroc"] = auroc(y[mask], score[mask])
        metrics["average_precision"] = average_precision(y[mask], score[mask])
        metrics["candidates"] = int(mask.sum())
        by_dataset[str(name)] = metrics
    macro_fpr = float(
        np.mean([float(value["false_positive_rate"]) for value in by_dataset.values()])
    )
    macro_ap = float(
        np.mean([float(value["average_precision"]) for value in by_dataset.values()])
    )
    return {"overall": overall, "by_dataset": by_dataset, "macro_fpr": macro_fpr, "macro_average_precision": macro_ap}


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def make_group_folds(
    dataset: np.ndarray,
    groups: np.ndarray,
    y: np.ndarray,
    num_folds: int = NUM_FOLDS,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    group_rows: dict[str, list[int]] = defaultdict(list)
    for index, (dataset_name, group_name) in enumerate(zip(dataset, groups)):
        group_rows[f"{dataset_name}::{group_name}"].append(index)

    fold_assignment: dict[str, int] = {}
    fold_stats = [defaultdict(int) for _ in range(num_folds)]
    for dataset_name in sorted(set(dataset.tolist())):
        keys = [key for key in group_rows if key.startswith(f"{dataset_name}::")]
        total_candidates = sum(len(group_rows[key]) for key in keys)
        total_positive = sum(int(y[group_rows[key]].sum()) for key in keys)
        total_negative = total_candidates - total_positive
        targets = {
            "candidates": max(total_candidates / num_folds, 1.0),
            "positive": max(total_positive / num_folds, 1.0),
            "negative": max(total_negative / num_folds, 1.0),
            "groups": max(len(keys) / num_folds, 1.0),
        }
        keys.sort(
            key=lambda key: (
                -len(group_rows[key]),
                -abs(int(y[group_rows[key]].sum()) - (len(group_rows[key]) - int(y[group_rows[key]].sum()))),
                stable_hash(f"{SEED}:{key}"),
            )
        )
        for key in keys:
            indices = group_rows[key]
            positive = int(y[indices].sum())
            negative = len(indices) - positive
            costs = []
            for fold in range(num_folds):
                stats = fold_stats[fold]
                values = {
                    "candidates": stats[f"{dataset_name}_candidates"] + len(indices),
                    "positive": stats[f"{dataset_name}_positive"] + positive,
                    "negative": stats[f"{dataset_name}_negative"] + negative,
                    "groups": stats[f"{dataset_name}_groups"] + 1,
                }
                cost = sum((values[name] / targets[name]) ** 2 for name in values)
                costs.append((cost, fold))
            _, chosen = min(costs)
            fold_assignment[key] = chosen
            fold_stats[chosen][f"{dataset_name}_candidates"] += len(indices)
            fold_stats[chosen][f"{dataset_name}_positive"] += positive
            fold_stats[chosen][f"{dataset_name}_negative"] += negative
            fold_stats[chosen][f"{dataset_name}_groups"] += 1

    fold_index = np.asarray(
        [fold_assignment[f"{d}::{g}"] for d, g in zip(dataset, groups)],
        dtype=np.int64,
    )
    audit = []
    for key, indices in sorted(group_rows.items()):
        dataset_name, sequence_id = key.split("::", 1)
        audit.append(
            {
                "dataset": dataset_name,
                "sequence_id": sequence_id,
                "fold": fold_assignment[key] + 1,
                "candidates": len(indices),
                "positive": int(y[indices].sum()),
                "negative": len(indices) - int(y[indices].sum()),
            }
        )
    return fold_index, audit


def grouped_oof(
    x: np.ndarray,
    y: np.ndarray,
    dataset: np.ndarray,
    fold_index: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    probability = np.zeros(len(y), dtype=np.float64)
    fold_rows = []
    for fold in sorted(np.unique(fold_index).tolist()):
        test_mask = fold_index == fold
        train_mask = ~test_mask
        if not test_mask.any() or len(np.unique(y[train_mask])) != 2:
            fail(f"Invalid CV fold {fold + 1}")
        train_weights = cell_balanced_weights(dataset[train_mask], y[train_mask])
        mean, scale = weighted_standardize_fit(x[train_mask], train_weights)
        beta, iterations = fit_logistic(
            (x[train_mask] - mean) / scale,
            y[train_mask].astype(np.float64),
            train_weights,
        )
        probability[test_mask] = predict_probability(
            (x[test_mask] - mean) / scale, beta
        )
        fold_rows.append(
            {
                "fold": int(fold + 1),
                "train_candidates": int(train_mask.sum()),
                "validation_candidates": int(test_mask.sum()),
                "validation_positive": int(y[test_mask].sum()),
                "validation_negative": int(test_mask.sum() - y[test_mask].sum()),
                "newton_iterations": iterations,
            }
        )
    return probability, fold_rows


def threshold_margin(score: np.ndarray, threshold: float) -> float:
    below = score[score < threshold]
    above = score[score >= threshold]
    if len(below) == 0 or len(above) == 0:
        return 0.0
    return float(above.min() - below.max())


def select_threshold(
    y: np.ndarray,
    score: np.ndarray,
    dataset: np.ndarray,
    require_per_dataset: bool = True,
) -> tuple[float, dict[str, object]]:
    unique = np.unique(score)
    candidates = [0.0]
    candidates.extend(
        float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:])
    )
    candidates.append(1.0)
    feasible = []
    for threshold in candidates:
        metrics = evaluate(y, score, dataset, threshold)
        global_recall = float(metrics["overall"]["recall_sensitivity"])
        dataset_recalls = [
            float(value["recall_sensitivity"])
            for value in metrics["by_dataset"].values()
        ]
        if global_recall + 1e-12 < GLOBAL_TARGET_RECALL:
            continue
        if require_per_dataset and any(
            recall + 1e-12 < PER_DATASET_MINIMUM_RECALL for recall in dataset_recalls
        ):
            continue
        feasible.append((threshold, metrics))
    if not feasible:
        fail("No threshold satisfies the predeclared recall constraints")
    threshold, metrics = min(
        feasible,
        key=lambda item: (
            float(item[1]["macro_fpr"]),
            -float(item[1]["macro_average_precision"]),
            -float(item[1]["overall"]["recall_sensitivity"]),
            -threshold_margin(score, item[0]),
            -item[0],
        ),
    )
    metrics["threshold_margin"] = threshold_margin(score, threshold)
    return float(threshold), metrics


def fit_final_model(
    x: np.ndarray, y: np.ndarray, dataset: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    weights = cell_balanced_weights(dataset, y)
    mean, scale = weighted_standardize_fit(x, weights)
    beta, iterations = fit_logistic(
        (x - mean) / scale, y.astype(np.float64), weights
    )
    return mean, scale, beta, iterations


def leave_one_dataset_out(
    x: np.ndarray,
    y: np.ndarray,
    dataset: np.ndarray,
    groups: np.ndarray,
) -> dict[str, object]:
    results: dict[str, object] = {}
    for target_name in sorted(set(dataset.tolist())):
        train_mask = dataset != target_name
        test_mask = dataset == target_name
        source_dataset = dataset[train_mask]
        source_groups = groups[train_mask]
        source_y = y[train_mask]
        source_x = x[train_mask]
        source_fold, _ = make_group_folds(
            source_dataset, source_groups, source_y, num_folds=NUM_FOLDS
        )
        source_oof, _ = grouped_oof(
            source_x, source_y, source_dataset, source_fold
        )
        threshold, source_metrics = select_threshold(
            source_y, source_oof, source_dataset, require_per_dataset=False
        )

        source_weights = cell_balanced_weights(source_dataset, source_y)
        mean, scale = weighted_standardize_fit(source_x, source_weights)
        beta, _ = fit_logistic(
            (source_x - mean) / scale,
            source_y.astype(np.float64),
            source_weights,
        )
        target_score = predict_probability((x[test_mask] - mean) / scale, beta)
        target_metrics = evaluate(
            y[test_mask], target_score, dataset[test_mask], threshold
        )
        results[f"train_other_test_{target_name}"] = {
            "source_dataset": sorted(set(source_dataset.tolist())),
            "target_dataset": target_name,
            "threshold_selected_from_source_oof": threshold,
            "source_oof": source_metrics,
            "target_evaluation": target_metrics,
        }
    return results


def main() -> None:
    if not INPUT.exists():
        fail(f"Input does not exist: {INPUT}")
    if OUTPUT_DIR.exists():
        fail(f"Output directory already exists; refusing to overwrite: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    try:
        rows, fields = read_csv(INPUT)
        all_features = sorted({name for names in FEATURE_CONFIGS.values() for name in names})
        missing = [name for name in all_features if name not in fields]
        if missing:
            fail(f"Required features are missing: {missing}")

        y = np.asarray([int(row["label_true_alarm"]) for row in rows], dtype=np.int64)
        dataset = np.asarray([row["dataset"] for row in rows], dtype=object)
        groups = np.asarray([row["sequence_id"] for row in rows], dtype=object)
        if set(y.tolist()) != {0, 1}:
            fail("Expected binary labels 0 and 1")
        if set(dataset.tolist()) != {"URFD", "Le2i"}:
            fail(f"Unexpected development datasets: {set(dataset.tolist())}")

        fold_index, fold_audit = make_group_folds(dataset, groups, y)
        write_csv(OUTPUT_DIR / "group_fold_assignments.csv", fold_audit)

        comparison: dict[str, dict[str, object]] = {}
        config_probabilities: dict[str, np.ndarray] = {}
        config_thresholds: dict[str, float] = {}
        config_matrices: dict[str, np.ndarray] = {}
        for config_name, feature_names in FEATURE_CONFIGS.items():
            x = np.asarray(
                [[float(row[name]) for name in feature_names] for row in rows],
                dtype=np.float64,
            )
            if not np.isfinite(x).all():
                fail(f"Non-finite feature value in {config_name}")
            probability, fold_rows = grouped_oof(x, y, dataset, fold_index)
            threshold, metrics = select_threshold(y, probability, dataset)
            comparison[config_name] = {
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "l2_regularization": L2_REGULARIZATION,
                "selected_threshold": threshold,
                "grouped_oof": metrics,
                "folds": fold_rows,
            }
            config_probabilities[config_name] = probability
            config_thresholds[config_name] = threshold
            config_matrices[config_name] = x

        selected_name = min(
            FEATURE_CONFIGS,
            key=lambda name: (
                float(comparison[name]["grouped_oof"]["macro_fpr"]),
                -float(
                    comparison[name]["grouped_oof"]["macro_average_precision"]
                ),
                len(FEATURE_CONFIGS[name]),
                name,
            ),
        )
        selected_features = FEATURE_CONFIGS[selected_name]
        selected_x = config_matrices[selected_name]
        selected_probability = config_probabilities[selected_name]
        selected_threshold = config_thresholds[selected_name]

        mean, scale, beta, final_iterations = fit_final_model(
            selected_x, y, dataset
        )
        model = {
            "model_type": "dataset_class_balanced_standardized_l2_logistic_regression",
            "model_version": "multidataset_alarm_verifier_v2",
            "selected_feature_configuration": selected_name,
            "feature_names": selected_features,
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            "intercept": float(beta[0]),
            "coefficients": beta[1:].tolist(),
            "decision_threshold": selected_threshold,
            "l2_regularization": L2_REGULARIZATION,
            "confirmation_future_samples": CONFIRMATION_FUTURE_SAMPLES,
            "maximum_added_delay_ms": MAXIMUM_ADDED_DELAY_MS,
            "positive_class": "first true fall event alarm",
            "negative_class": "false, early, or duplicate alarm",
        }
        model_path = OUTPUT_DIR / "alarm_verifier_v2_model.json"
        model_path.write_text(
            json.dumps(model, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        oof_rows = []
        for index, row in enumerate(rows):
            output: dict[str, object] = {
                "candidate_key": row["candidate_key"],
                "dataset": row["dataset"],
                "sequence_id": row["sequence_id"],
                "sample_index": row["sample_index"],
                "label_true_alarm": int(y[index]),
                "fold": int(fold_index[index] + 1),
            }
            for config_name in FEATURE_CONFIGS:
                probability = float(config_probabilities[config_name][index])
                output[f"{config_name}_oof_probability"] = probability
                output[f"{config_name}_oof_prediction"] = int(
                    probability >= config_thresholds[config_name]
                )
            oof_rows.append(output)
        write_csv(OUTPUT_DIR / "oof_predictions.csv", oof_rows)

        cross_dataset = leave_one_dataset_out(selected_x, y, dataset, groups)
        summary = {
            "protocol": {
                "development_datasets": ["URFD Camera 0 RGB", "Le2i revealed V1 test"],
                "untouched_final_blind_dataset": "GMDCSA24 v2.1",
                "gmdcsa24_read": False,
                "cross_validation": "deterministic 5-fold complete-sequence grouped CV",
                "sample_weighting": "equal total weight per dataset x class cell",
                "global_target_recall": GLOBAL_TARGET_RECALL,
                "per_dataset_minimum_recall": PER_DATASET_MINIMUM_RECALL,
                "selection_order": [
                    "satisfy global and per-dataset recall constraints",
                    "minimize macro dataset false-positive rate",
                    "maximize macro average precision",
                    "prefer fewer features",
                ],
                "future_samples_after_candidate": CONFIRMATION_FUTURE_SAMPLES,
                "maximum_added_decision_delay_ms": MAXIMUM_ADDED_DELAY_MS,
                "post_decision_samples_used": 0,
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "candidate_count": len(rows),
            "positive_candidates": int(y.sum()),
            "negative_candidates": int(len(y) - y.sum()),
            "sequence_groups": len(set(zip(dataset.tolist(), groups.tolist()))),
            "dataset_counts": dict(Counter(dataset.tolist())),
            "feature_configuration_comparison": comparison,
            "selected_feature_configuration": selected_name,
            "selected_threshold": selected_threshold,
            "selected_grouped_oof": comparison[selected_name]["grouped_oof"],
            "leave_one_dataset_out_diagnostics": cross_dataset,
            "final_model_newton_iterations": final_iterations,
            "input": str(INPUT),
            "input_sha256": sha256_file(INPUT),
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
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
