from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path("/home/data/yoloA27/URFD")
MODEL_PATH = Path(
    "/home/data/yoloA27/experiments/"
    "yolo26n_baseline_seed42/weights/best.pt"
)
FRAME_LABELS = ROOT / "metadata/rgb_frame_labels.csv"
MANUAL_EVENTS = ROOT / "metadata/manual_event_annotations.csv"
OUTPUT_DIR = Path("/home/data/yoloA27/experiments/urfd_external_baseline")

IMAGE_SIZE = 640
BATCH_SIZE = 1
DEVICE = 0
INFERENCE_CONF = 0.001
NMS_IOU = 0.7

# Both values are locked before looking at URFD results.
FALL_THRESHOLD = 0.623
CONSECUTIVE_FRAMES = 5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def image_path_from_row(row: dict[str, str]) -> Path:
    category_dir = "falls" if row["category"].lower() == "fall" else "adl"
    sequence_dir = f'{row["sequence_id"]}-cam0-rgb'
    filename = PureWindowsPath(row["image_path"]).name
    return ROOT / category_dir / sequence_dir / filename


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return 0.0, 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y_true[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)

    threshold_ends = np.r_[np.where(np.diff(sorted_scores) != 0)[0], len(scores) - 1]
    tpr = np.r_[0.0, tp[threshold_ends] / positives]
    fpr = np.r_[0.0, fp[threshold_ends] / negatives]
    auroc = float(np.trapz(tpr, fpr))

    precision = tp / np.arange(1, len(y_true) + 1)
    recall = tp / positives
    recall_at_thresholds = recall[threshold_ends]
    precision_at_thresholds = precision[threshold_ends]
    auprc = float(
        np.sum(
            np.diff(np.r_[0.0, recall_at_thresholds])
            * precision_at_thresholds
        )
    )
    return auroc, auprc


def expected_calibration_error(
    y_true: np.ndarray, scores: np.ndarray, bins: int = 10
) -> float:
    total = len(y_true)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            mask = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if not np.any(mask):
            continue
        confidence = float(np.mean(scores[mask]))
        positive_rate = float(np.mean(y_true[mask]))
        ece += (int(np.sum(mask)) / total) * abs(positive_rate - confidence)
    return float(ece)


def get_alarm_triggers(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    triggers = []
    consecutive = 0
    active = False
    for row in rows:
        if int(row["pred_fall"]) == 1:
            consecutive += 1
            if consecutive >= CONSECUTIVE_FRAMES and not active:
                triggers.append(row)
                active = True
        else:
            consecutive = 0
            active = False
    return triggers


def main() -> None:
    for required in (MODEL_PATH, FRAME_LABELS, MANUAL_EVENTS):
        if not required.is_file():
            raise FileNotFoundError(f"Missing required file: {required}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_rows = read_csv(FRAME_LABELS)
    manual_rows = read_csv(MANUAL_EVENTS)

    if len(frame_rows) != 11936:
        raise RuntimeError(f"Expected 11936 frames, found {len(frame_rows)}")
    if len(manual_rows) != 30:
        raise RuntimeError(f"Expected 30 fall events, found {len(manual_rows)}")

    image_paths = [image_path_from_row(row) for row in frame_rows]
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} images. First missing image: {missing[0]}"
        )

    print(f"Model: {MODEL_PATH}")
    print(f"Frames: {len(image_paths)}")
    print(f"Locked fall threshold: {FALL_THRESHOLD}")
    print(f"Locked consecutive frames: {CONSECUTIVE_FRAMES}")
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    model = YOLO(str(MODEL_PATH))
    predictions = (
        result
        for path in image_paths
        for result in model.predict(
            source=str(path),
            imgsz=IMAGE_SIZE,
            batch=1,
            device=DEVICE,
            conf=INFERENCE_CONF,
            iou=NMS_IOU,
            stream=True,
            verbose=False,
        )
    )

    output_rows: list[dict[str, object]] = []
    for index, (metadata, path, result) in enumerate(
        zip(frame_rows, image_paths, predictions), start=1
    ):
        non_fall_conf = 0.0
        fall_conf = 0.0
        detection_count = 0

        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            detection_count = len(classes)
            if np.any(classes == 0):
                non_fall_conf = float(np.max(confidences[classes == 0]))
            if np.any(classes == 1):
                fall_conf = float(np.max(confidences[classes == 1]))

        if max(non_fall_conf, fall_conf) == 0.0:
            top_class = "no_detection"
            top_conf = 0.0
        elif fall_conf > non_fall_conf:
            top_class = "fall"
            top_conf = fall_conf
        else:
            top_class = "non_fall"
            top_conf = non_fall_conf

        gt_fall = int(metadata["event_label"] in {"Falling", "Fallen"})
        pred_fall = int(
            fall_conf >= FALL_THRESHOLD and fall_conf > non_fall_conf
        )

        output_rows.append(
            {
                "sequence_id": metadata["sequence_id"],
                "category": metadata["category"],
                "frame_number": int(metadata["frame_number"]),
                "timestamp_ms": int(metadata["timestamp_ms"]),
                "timestamp_source": metadata["timestamp_source"],
                "event_label": metadata["event_label"],
                "gt_fall": gt_fall,
                "non_fall_conf": non_fall_conf,
                "fall_conf": fall_conf,
                "top_class": top_class,
                "top_conf": top_conf,
                "pred_fall": pred_fall,
                "correct": int(gt_fall == pred_fall),
                "detection_count": detection_count,
                "image_path": str(path),
            }
        )
        if index % 50 == 0 or index == len(frame_rows):
            print(f"Processed {index}/{len(frame_rows)} frames")

    if len(output_rows) != len(frame_rows):
        raise RuntimeError(
            f"Inference returned {len(output_rows)} results for "
            f"{len(frame_rows)} input frames"
        )

    prediction_csv = OUTPUT_DIR / "urfd_frame_predictions.csv"
    with prediction_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    y_true = np.asarray([row["gt_fall"] for row in output_rows], dtype=int)
    y_pred = np.asarray([row["pred_fall"] for row in output_rows], dtype=int)
    scores = np.asarray([row["fall_conf"] for row in output_rows], dtype=float)

    frame_summary = binary_metrics(y_true, y_pred)
    auroc, auprc = ranking_metrics(y_true, scores)
    frame_summary.update(
        {
            "auroc": auroc,
            "auprc_average_precision": auprc,
            "brier_score": float(np.mean((scores - y_true) ** 2)),
            "ece_10_bins": expected_calibration_error(y_true, scores),
        }
    )

    rows_by_sequence: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in output_rows:
        rows_by_sequence[str(row["sequence_id"])].append(row)
    for rows in rows_by_sequence.values():
        rows.sort(key=lambda item: int(item["frame_number"]))

    annotations = {row["sequence_id"]: row for row in manual_rows}
    event_rows = []
    false_alarm_count = 0
    duplicate_alarm_count = 0
    delays_ms = []

    for sequence_id, rows in sorted(rows_by_sequence.items()):
        triggers = get_alarm_triggers(rows)
        if sequence_id.startswith("fall-"):
            annotation = annotations[sequence_id]
            onset = int(annotation["onset_frame"])
            impact = int(annotation["impact_frame"])
            stable = int(annotation["stable_fallen_frame"])
            valid_triggers = [
                row for row in triggers if int(row["frame_number"]) >= onset
            ]
            early_triggers = [
                row for row in triggers if int(row["frame_number"]) < onset
            ]
            false_alarm_count += len(early_triggers)
            duplicate_alarm_count += max(0, len(valid_triggers) - 1)

            onset_row = next(
                row for row in rows if int(row["frame_number"]) == onset
            )
            if valid_triggers:
                first_alarm = valid_triggers[0]
                first_alarm_frame = int(first_alarm["frame_number"])
                first_alarm_ms = int(first_alarm["timestamp_ms"])
                delay_ms = first_alarm_ms - int(onset_row["timestamp_ms"])
                delays_ms.append(delay_ms)
                detected = 1
            else:
                first_alarm_frame = ""
                first_alarm_ms = ""
                delay_ms = ""
                detected = 0

            event_rows.append(
                {
                    "sequence_id": sequence_id,
                    "onset_frame": onset,
                    "impact_frame": impact,
                    "stable_fallen_frame": stable,
                    "detected": detected,
                    "first_alarm_frame": first_alarm_frame,
                    "first_alarm_ms": first_alarm_ms,
                    "detection_delay_ms_from_onset": delay_ms,
                    "early_false_alarms": len(early_triggers),
                    "alarms_after_onset": len(valid_triggers),
                }
            )
        else:
            false_alarm_count += len(triggers)

    event_csv = OUTPUT_DIR / "urfd_event_results.csv"
    with event_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(event_rows[0].keys()))
        writer.writeheader()
        writer.writerows(event_rows)

    negative_frames = int(np.sum(y_true == 0))
    negative_exposure_hours = negative_frames / 30.0 / 3600.0
    detected_events = int(sum(int(row["detected"]) for row in event_rows))
    event_summary = {
        "total_fall_events": len(event_rows),
        "detected_fall_events": detected_events,
        "missed_fall_events": len(event_rows) - detected_events,
        "event_recall": safe_div(detected_events, len(event_rows)),
        "false_alarm_count": false_alarm_count,
        "negative_exposure_hours_assuming_30fps": negative_exposure_hours,
        "false_alarms_per_hour": safe_div(false_alarm_count, negative_exposure_hours),
        "duplicate_alarms_after_onset": duplicate_alarm_count,
        "mean_detection_delay_ms": float(np.mean(delays_ms)) if delays_ms else None,
        "median_detection_delay_ms": float(np.median(delays_ms)) if delays_ms else None,
        "p95_detection_delay_ms": float(np.percentile(delays_ms, 95)) if delays_ms else None,
    }

    summary = {
        "protocol": {
            "source_training_dataset": "CAUCAFall",
            "external_test_dataset": "URFD Camera 0 RGB",
            "fine_tuning": False,
            "fall_threshold_locked_from_source_validation": FALL_THRESHOLD,
            "consecutive_frames_locked_before_external_test": CONSECUTIVE_FRAMES,
            "positive_frame_definition": "Falling or Fallen",
            "image_size": IMAGE_SIZE,
            "inference_confidence_floor": INFERENCE_CONF,
            "nms_iou": NMS_IOU,
        },
        "frame_level": frame_summary,
        "event_level_simple_sequential_baseline": event_summary,
    }

    summary_json = OUTPUT_DIR / "urfd_external_summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_txt = OUTPUT_DIR / "urfd_external_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as file:
        file.write(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nExternal test completed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
