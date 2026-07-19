#!/usr/bin/env python3
"""Evaluate the frozen YOLO binary detector on the held-out CAUCAFall test split.

This script performs no training and no threshold selection. It reports:

1. Ultralytics' standard object-detection metrics on split=test.
2. Frame-level fall/non-fall metrics at the source-validation-locked fall
   threshold (0.623): precision, recall, specificity, F1, AUROC, AUPRC,
   balanced accuracy, macro F1, and macro AUPRC.

Frame-level fall score definition:
    maximum confidence among class-1 (fall) detections in an image;
    zero when no class-1 detection is returned at the 0.001 inference floor.

Every CAUCAFall image is expected to have exactly one YOLO label row whose
class is 0 (non_fall) or 1 (fall).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Locked paths and evaluation settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/data/yoloA27")
MODEL_PATH = (
    PROJECT_ROOT
    / "experiments/yolo26n_baseline_seed42/weights/best.pt"
)
DATA_YAML = PROJECT_ROOT / "CAUCAFall_YOLO/data.yaml"
IMAGE_ROOT = PROJECT_ROOT / "CAUCAFall_YOLO/images/test"
LABEL_ROOT = PROJECT_ROOT / "CAUCAFall_YOLO/labels/test"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/caucafall_heldout_binary_test_v1"
)

CLASS_NAMES = {0: "non_fall", 1: "fall"}
FALL_CLASS_ID = 1

# This threshold was selected from source validation before external testing.
FALL_THRESHOLD = 0.623

IMAGE_SIZE = 640
BATCH_SIZE = 32
WORKERS = 4
DEVICE = 0
INFERENCE_CONFIDENCE_FLOOR = 0.001
NMS_IOU = 0.7
MAX_DETECTIONS = 300
SOURCE_CHUNK_SIZE = 512

# This adds a second inference pass but produces the standard Ultralytics
# split=test results and plots, which are useful for the manuscript record.
RUN_OFFICIAL_YOLO_TEST = True

IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def finite_or_none(value: Any) -> Any:
    """Convert NumPy/Torch objects to JSON-safe built-in values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            value = value.item()
        else:
            return [finite_or_none(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def list_test_images() -> List[Path]:
    images = sorted(
        path
        for path in IMAGE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No test images found under {IMAGE_ROOT}")
    return images


def label_path_for_image(image_path: Path) -> Path:
    relative = image_path.relative_to(IMAGE_ROOT)
    return LABEL_ROOT / relative.with_suffix(".txt")


def read_single_frame_label(image_path: Path) -> int:
    label_path = label_path_for_image(image_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")

    rows = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one label row for {image_path}, found {len(rows)}"
        )

    fields = rows[0].split()
    if len(fields) != 5:
        raise RuntimeError(
            f"Expected five-column YOLO label in {label_path}, found {len(fields)}"
        )

    try:
        class_id = int(float(fields[0]))
    except ValueError as exc:
        raise RuntimeError(f"Invalid class value in {label_path}: {fields[0]}") from exc

    if class_id not in CLASS_NAMES:
        raise RuntimeError(f"Unexpected class {class_id} in {label_path}")
    return class_id


def validate_inventory(images: Sequence[Path]) -> List[int]:
    labels = [read_single_frame_label(path) for path in images]
    label_files = sorted(path for path in LABEL_ROOT.rglob("*.txt") if path.is_file())

    if len(label_files) != len(images):
        raise RuntimeError(
            "Image/label inventory mismatch: "
            f"{len(images)} images versus {len(label_files)} label files"
        )

    negatives = sum(label == 0 for label in labels)
    positives = sum(label == 1 for label in labels)
    print(
        f"Verified inventory: {len(images)} images, "
        f"{negatives} non_fall, {positives} fall"
    )
    return labels


def chunks(items: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks, assigning equal values their mean rank."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    ranks = average_ranks(scores)
    positive_rank_sum = float(ranks[positives].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def grouped_precision_recall_curve(
    y_true: np.ndarray, scores: np.ndarray
) -> List[Dict[str, float]]:
    """Create threshold points after grouping identical scores."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_true = y_true[order]
    total_positives = int((y_true == 1).sum())

    points: List[Dict[str, float]] = []
    true_positives = 0
    false_positives = 0
    start = 0

    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1

        group = sorted_true[start:end]
        true_positives += int((group == 1).sum())
        false_positives += int((group == 0).sum())
        precision = safe_divide(true_positives, true_positives + false_positives)
        recall = safe_divide(true_positives, total_positives)
        points.append(
            {
                "threshold": float(sorted_scores[start]),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "precision": precision,
                "recall": recall,
            }
        )
        start = end

    return points


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    total_positives = int((y_true == 1).sum())
    if total_positives == 0:
        return float("nan")

    points = grouped_precision_recall_curve(y_true, scores)
    average_precision_value = 0.0
    previous_recall = 0.0
    for point in points:
        recall = float(point["recall"])
        precision = float(point["precision"])
        average_precision_value += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision_value


def calculate_frame_metrics(
    y_true: np.ndarray, fall_scores: np.ndarray, threshold: float
) -> Dict[str, Any]:
    predicted = (fall_scores >= threshold).astype(np.int64)

    true_positive = int(((y_true == 1) & (predicted == 1)).sum())
    false_positive = int(((y_true == 0) & (predicted == 1)).sum())
    true_negative = int(((y_true == 0) & (predicted == 0)).sum())
    false_negative = int(((y_true == 1) & (predicted == 0)).sum())

    positive_precision = safe_divide(true_positive, true_positive + false_positive)
    positive_recall = safe_divide(true_positive, true_positive + false_negative)
    specificity = safe_divide(true_negative, true_negative + false_positive)
    positive_f1 = safe_divide(
        2.0 * positive_precision * positive_recall,
        positive_precision + positive_recall,
    )

    negative_precision = safe_divide(true_negative, true_negative + false_negative)
    negative_recall = specificity
    negative_f1 = safe_divide(
        2.0 * negative_precision * negative_recall,
        negative_precision + negative_recall,
    )

    positive_auprc = average_precision(y_true, fall_scores)
    negative_auprc = average_precision(1 - y_true, 1.0 - fall_scores)

    return {
        "threshold": threshold,
        "samples": int(len(y_true)),
        "positive_samples": int((y_true == 1).sum()),
        "negative_samples": int((y_true == 0).sum()),
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
        "accuracy": safe_divide(true_positive + true_negative, len(y_true)),
        "positive_precision": positive_precision,
        "positive_recall_sensitivity": positive_recall,
        "specificity": specificity,
        "positive_f1": positive_f1,
        "negative_f1": negative_f1,
        "binary_macro_f1": (positive_f1 + negative_f1) / 2.0,
        "balanced_accuracy": (positive_recall + specificity) / 2.0,
        "auroc": binary_auroc(y_true, fall_scores),
        "positive_auprc_average_precision": positive_auprc,
        "negative_auprc_average_precision": negative_auprc,
        "binary_macro_auprc": (positive_auprc + negative_auprc) / 2.0,
    }


def run_official_yolo_test(model: YOLO) -> Dict[str, Any]:
    if not RUN_OFFICIAL_YOLO_TEST:
        return {"status": "skipped"}

    print("\nRunning official Ultralytics split=test evaluation...")
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=INFERENCE_CONFIDENCE_FLOOR,
        iou=NMS_IOU,
        max_det=MAX_DETECTIONS,
        device=DEVICE,
        workers=WORKERS,
        plots=True,
        project=str(OUTPUT_DIR),
        name="official_yolo_test",
        exist_ok=True,
        verbose=True,
    )

    result: Dict[str, Any] = {
        "status": "completed",
        "save_dir": str(getattr(metrics, "save_dir", "")),
        "results_dict": finite_or_none(getattr(metrics, "results_dict", {})),
        "speed_ms_per_image": finite_or_none(getattr(metrics, "speed", {})),
    }

    box = getattr(metrics, "box", None)
    if box is not None:
        for attribute in ("mp", "mr", "map50", "map75", "map"):
            if hasattr(box, attribute):
                result[f"box_{attribute}"] = finite_or_none(getattr(box, attribute))
        if hasattr(box, "maps"):
            result["box_map50_95_by_class"] = finite_or_none(box.maps)
    return result


def run_frame_inference(
    model: YOLO, images: Sequence[Path], true_labels: Sequence[int]
) -> List[Dict[str, Any]]:
    print("\nRunning frame-level held-out inference...")
    rows: List[Dict[str, Any]] = []
    processed = 0

    for chunk_index, image_chunk in enumerate(
        chunks(images, SOURCE_CHUNK_SIZE), start=1
    ):
        source = [str(path) for path in image_chunk]
        results = model.predict(
            source=source,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            conf=INFERENCE_CONFIDENCE_FLOOR,
            iou=NMS_IOU,
            max_det=MAX_DETECTIONS,
            device=DEVICE,
            stream=True,
            save=False,
            save_txt=False,
            save_conf=False,
            verbose=False,
        )

        chunk_result_count = 0
        for image_path, result in zip(image_chunk, results):
            global_index = processed + chunk_result_count
            true_label = int(true_labels[global_index])

            if result.boxes is None or len(result.boxes) == 0:
                classes = np.empty(0, dtype=np.int64)
                confidences = np.empty(0, dtype=np.float64)
            else:
                classes = (
                    result.boxes.cls.detach().cpu().numpy().astype(np.int64)
                )
                confidences = (
                    result.boxes.conf.detach().cpu().numpy().astype(np.float64)
                )

            fall_confidences = confidences[classes == FALL_CLASS_ID]
            adl_confidences = confidences[classes == 0]
            fall_score = (
                float(fall_confidences.max()) if fall_confidences.size else 0.0
            )
            adl_score = float(adl_confidences.max()) if adl_confidences.size else 0.0

            if confidences.size:
                best_index = int(np.argmax(confidences))
                best_class = int(classes[best_index])
                best_confidence = float(confidences[best_index])
            else:
                best_class = -1
                best_confidence = 0.0

            predicted_label = int(fall_score >= FALL_THRESHOLD)
            rows.append(
                {
                    "relative_image_path": str(image_path.relative_to(IMAGE_ROOT)),
                    "true_label": true_label,
                    "true_name": CLASS_NAMES[true_label],
                    "fall_score": fall_score,
                    "adl_score": adl_score,
                    "predicted_label_at_locked_threshold": predicted_label,
                    "predicted_name_at_locked_threshold": CLASS_NAMES[predicted_label],
                    "correct_at_locked_threshold": int(predicted_label == true_label),
                    "number_of_detections": int(len(confidences)),
                    "highest_confidence_detection_class": best_class,
                    "highest_confidence_detection_confidence": best_confidence,
                }
            )
            chunk_result_count += 1

        if chunk_result_count != len(image_chunk):
            raise RuntimeError(
                f"Chunk {chunk_index} returned {chunk_result_count} results "
                f"for {len(image_chunk)} images"
            )

        processed += chunk_result_count
        print(f"Processed {processed}/{len(images)} test images")

    if len(rows) != len(images):
        raise RuntimeError(f"Expected {len(images)} rows, produced {len(rows)}")
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, metrics: Dict[str, Any]) -> None:
    ordered_keys = [
        "threshold",
        "samples",
        "positive_samples",
        "negative_samples",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "positive_precision",
        "positive_recall_sensitivity",
        "specificity",
        "positive_f1",
        "negative_f1",
        "binary_macro_f1",
        "balanced_accuracy",
        "auroc",
        "positive_auprc_average_precision",
        "negative_auprc_average_precision",
        "binary_macro_auprc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_keys)
        writer.writeheader()
        writer.writerow({key: metrics[key] for key in ordered_keys})


def print_metrics(metrics: Dict[str, Any]) -> None:
    print("\n===== HELD-OUT FRAME-LEVEL BINARY RESULTS =====")
    print(
        f"samples={metrics['samples']}  "
        f"positive={metrics['positive_samples']}  "
        f"negative={metrics['negative_samples']}"
    )
    print(
        f"tp={metrics['tp']}  fp={metrics['fp']}  "
        f"tn={metrics['tn']}  fn={metrics['fn']}"
    )
    display = [
        ("threshold", "threshold"),
        ("accuracy", "accuracy"),
        ("precision", "positive_precision"),
        ("recall/sensitivity", "positive_recall_sensitivity"),
        ("specificity", "specificity"),
        ("positive F1", "positive_f1"),
        ("binary macro F1", "binary_macro_f1"),
        ("balanced accuracy", "balanced_accuracy"),
        ("AUROC", "auroc"),
        ("positive AUPRC/AP", "positive_auprc_average_precision"),
        ("binary macro AUPRC", "binary_macro_auprc"),
    ]
    for label, key in display:
        value = metrics[key]
        print(f"{label:22s}: {value:.6f}")


def main() -> None:
    for required_path in (MODEL_PATH, DATA_YAML, IMAGE_ROOT, LABEL_ROOT):
        if not required_path.exists():
            raise FileNotFoundError(f"Required path does not exist: {required_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = list_test_images()
    true_labels = validate_inventory(images)

    print(f"Loading frozen detector: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    official_yolo_metrics = run_official_yolo_test(model)
    prediction_rows = run_frame_inference(model, images, true_labels)

    y_true = np.asarray(
        [row["true_label"] for row in prediction_rows], dtype=np.int64
    )
    fall_scores = np.asarray(
        [row["fall_score"] for row in prediction_rows], dtype=np.float64
    )
    frame_metrics = calculate_frame_metrics(y_true, fall_scores, FALL_THRESHOLD)

    per_frame_path = OUTPUT_DIR / "per_frame_predictions.csv"
    summary_csv_path = OUTPUT_DIR / "heldout_binary_summary.csv"
    summary_json_path = OUTPUT_DIR / "heldout_binary_summary.json"
    curve_path = OUTPUT_DIR / "positive_precision_recall_curve.csv"

    write_csv(per_frame_path, prediction_rows)
    write_summary_csv(summary_csv_path, frame_metrics)
    write_csv(curve_path, grouped_precision_recall_curve(y_true, fall_scores))

    complete_summary = {
        "protocol": {
            "dataset": "CAUCAFall",
            "split": "held-out source test (Subjects 2 and 10)",
            "task": "binary frame-level non_fall/fall detector evaluation",
            "training_performed": False,
            "threshold_selected_on_test": False,
            "fall_score_definition": (
                "maximum confidence among class-1 detections; zero when no "
                "class-1 detection is returned at the inference floor"
            ),
        },
        "configuration": {
            "model_path": str(MODEL_PATH),
            "model_sha256": sha256_file(MODEL_PATH),
            "data_yaml": str(DATA_YAML),
            "data_yaml_sha256": sha256_file(DATA_YAML),
            "image_root": str(IMAGE_ROOT),
            "label_root": str(LABEL_ROOT),
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "device": DEVICE,
            "inference_confidence_floor": INFERENCE_CONFIDENCE_FLOOR,
            "nms_iou": NMS_IOU,
            "max_detections": MAX_DETECTIONS,
            "fall_threshold_locked_from_source_validation": FALL_THRESHOLD,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
        },
        "frame_level_binary_metrics": frame_metrics,
        "official_ultralytics_test_metrics": official_yolo_metrics,
        "outputs": {
            "per_frame_predictions_csv": str(per_frame_path),
            "summary_csv": str(summary_csv_path),
            "precision_recall_curve_csv": str(curve_path),
        },
    }
    summary_json_path.write_text(
        json.dumps(finite_or_none(complete_summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_metrics(frame_metrics)
    print("\nCompleted without training or test-set threshold tuning.")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
