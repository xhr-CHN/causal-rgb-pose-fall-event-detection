from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
from collections import defaultdict
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


DATASET_ROOT = Path("/home/data/yoloA27/CAUCAFall_YOLO")
MANIFEST_PATH = DATASET_ROOT / "prepared_dataset_manifest.csv"
POSE_MODEL_PATH = Path("/home/data/yoloA27/yolo26n-pose.pt")
OUTPUT_DIR = Path("/home/data/yoloA27/features/caucafall_pose_v1")

# Method development uses only train and val. Test remains untouched until frozen.
SPLITS = ("train", "val")
IMAGE_SIZE = 640
DEVICE = 0
POSE_CONFIDENCE_FLOOR = 0.01
NMS_IOU = 0.7
VISIBLE_KEYPOINT_THRESHOLD = 0.25
BOUNDARY_MARGIN = 0.02
REPORT_EVERY = 50

KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
TORSO_INDICES = (5, 6, 11, 12)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_image_name(row: dict[str, str]) -> str:
    return PureWindowsPath(row["destination_image"]).name


def get_sequence_id(row: dict[str, str]) -> str:
    return f'S{int(row["subject_id"])}_{slugify(row["activity_name"])}'


def raw_frame_code(image_name: str) -> int:
    match = re.search(r"(\d+)$", Path(image_name).stem)
    if not match:
        raise ValueError(f"Cannot parse frame code from {image_name}")
    return int(match.group(1))


def read_frame_label(label_path: Path) -> int:
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label: {label_path}")
    lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one object in {label_path}, found {len(lines)}")
    parts = lines[0].split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label in {label_path}: {lines[0]}")
    class_id = int(parts[0])
    if class_id not in (0, 1):
        raise ValueError(f"Invalid class {class_id} in {label_path}")
    return class_id


def output_fieldnames() -> list[str]:
    fields = [
        "split",
        "subject_id",
        "activity_name",
        "category",
        "sequence_id",
        "frame_index",
        "raw_frame_code",
        "image_name",
        "image_path",
        "frame_label",
        "pose_found",
        "person_count",
        "person_conf",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_cx",
        "bbox_cy",
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "boundary_left",
        "boundary_top",
        "boundary_right",
        "boundary_bottom",
        "boundary_touch_count",
        "visible_keypoint_count",
        "mean_keypoint_conf",
        "torso_keypoint_conf",
    ]
    for name in KEYPOINT_NAMES:
        fields.extend(
            [
                f"{name}_x_global",
                f"{name}_y_global",
                f"{name}_x_bbox",
                f"{name}_y_bbox",
                f"{name}_conf",
            ]
        )
    return fields


def missing_pose_values() -> dict[str, float | int]:
    values: dict[str, float | int] = {
        "pose_found": 0,
        "person_count": 0,
        "person_conf": 0.0,
        "bbox_x1": 0.0,
        "bbox_y1": 0.0,
        "bbox_x2": 0.0,
        "bbox_y2": 0.0,
        "bbox_cx": 0.0,
        "bbox_cy": 0.0,
        "bbox_w": 0.0,
        "bbox_h": 0.0,
        "bbox_area": 0.0,
        "boundary_left": 0,
        "boundary_top": 0,
        "boundary_right": 0,
        "boundary_bottom": 0,
        "boundary_touch_count": 0,
        "visible_keypoint_count": 0,
        "mean_keypoint_conf": 0.0,
        "torso_keypoint_conf": 0.0,
    }
    for name in KEYPOINT_NAMES:
        values[f"{name}_x_global"] = 0.0
        values[f"{name}_y_global"] = 0.0
        values[f"{name}_x_bbox"] = 0.0
        values[f"{name}_y_bbox"] = 0.0
        values[f"{name}_conf"] = 0.0
    return values


def pose_values(result) -> dict[str, float | int]:
    values = missing_pose_values()
    if (
        result.boxes is None
        or result.keypoints is None
        or len(result.boxes) == 0
        or len(result.keypoints) == 0
    ):
        return values

    box_confidences = result.boxes.conf.detach().cpu().numpy()
    selected = int(np.argmax(box_confidences))
    person_count = int(len(result.boxes))
    person_confidence = float(box_confidences[selected])

    xyxy = result.boxes.xyxy[selected].detach().cpu().numpy().astype(float)
    keypoint_xy = result.keypoints.xy[selected].detach().cpu().numpy().astype(float)
    if result.keypoints.conf is None:
        keypoint_conf = np.ones(len(KEYPOINT_NAMES), dtype=float)
    else:
        keypoint_conf = (
            result.keypoints.conf[selected].detach().cpu().numpy().astype(float)
        )

    image_height, image_width = result.orig_shape
    x1 = float(np.clip(xyxy[0] / image_width, 0.0, 1.0))
    y1 = float(np.clip(xyxy[1] / image_height, 0.0, 1.0))
    x2 = float(np.clip(xyxy[2] / image_width, 0.0, 1.0))
    y2 = float(np.clip(xyxy[3] / image_height, 0.0, 1.0))
    bbox_width = max(x2 - x1, 1e-6)
    bbox_height = max(y2 - y1, 1e-6)

    left = int(x1 <= BOUNDARY_MARGIN)
    top = int(y1 <= BOUNDARY_MARGIN)
    right = int(x2 >= 1.0 - BOUNDARY_MARGIN)
    bottom = int(y2 >= 1.0 - BOUNDARY_MARGIN)

    valid_confidences = keypoint_conf[np.isfinite(keypoint_conf)]
    mean_confidence = (
        float(np.mean(valid_confidences)) if len(valid_confidences) else 0.0
    )
    torso_confidence = float(np.mean(keypoint_conf[list(TORSO_INDICES)]))

    values.update(
        {
            "pose_found": 1,
            "person_count": person_count,
            "person_conf": person_confidence,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_cx": (x1 + x2) / 2,
            "bbox_cy": (y1 + y2) / 2,
            "bbox_w": bbox_width,
            "bbox_h": bbox_height,
            "bbox_area": bbox_width * bbox_height,
            "boundary_left": left,
            "boundary_top": top,
            "boundary_right": right,
            "boundary_bottom": bottom,
            "boundary_touch_count": left + top + right + bottom,
            "visible_keypoint_count": int(
                np.sum(keypoint_conf >= VISIBLE_KEYPOINT_THRESHOLD)
            ),
            "mean_keypoint_conf": mean_confidence,
            "torso_keypoint_conf": torso_confidence,
        }
    )

    for index, name in enumerate(KEYPOINT_NAMES):
        x_global = float(np.clip(keypoint_xy[index, 0] / image_width, 0.0, 1.0))
        y_global = float(np.clip(keypoint_xy[index, 1] / image_height, 0.0, 1.0))
        confidence = float(np.clip(keypoint_conf[index], 0.0, 1.0))
        values[f"{name}_x_global"] = x_global
        values[f"{name}_y_global"] = y_global
        values[f"{name}_x_bbox"] = float(np.clip((x_global - x1) / bbox_width, 0.0, 1.0))
        values[f"{name}_y_bbox"] = float(np.clip((y_global - y1) / bbox_height, 0.0, 1.0))
        values[f"{name}_conf"] = confidence
    return values


def processed_image_names(output_path: Path) -> set[str]:
    if not output_path.is_file():
        return set()
    with output_path.open("r", encoding="utf-8", newline="") as file:
        return {row["image_name"] for row in csv.DictReader(file)}


def extract_split(model: YOLO, rows: list[dict[str, str]], split: str) -> None:
    output_path = OUTPUT_DIR / f"pose_features_{split}.csv"
    processed = processed_image_names(output_path)
    split_rows = [row for row in rows if row["split"] == split]
    split_rows.sort(
        key=lambda row: (
            int(row["subject_id"]),
            row["activity_name"],
            get_image_name(row),
        )
    )

    frame_indices: dict[str, int] = defaultdict(int)
    for row in split_rows:
        frame_indices[get_sequence_id(row)] += 1
        row["_frame_index"] = str(frame_indices[get_sequence_id(row)])

    remaining = [row for row in split_rows if get_image_name(row) not in processed]
    print(
        f"{split}: total={len(split_rows)}, already_done={len(processed)}, "
        f"remaining={len(remaining)}",
        flush=True,
    )
    if not remaining:
        return

    fieldnames = output_fieldnames()
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for number, row in enumerate(remaining, start=1):
            image_name = get_image_name(row)
            image_path = DATASET_ROOT / "images" / split / image_name
            label_path = DATASET_ROOT / "labels" / split / f"{Path(image_name).stem}.txt"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image: {image_path}")

            result = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                conf=POSE_CONFIDENCE_FLOOR,
                iou=NMS_IOU,
                verbose=False,
            )[0]

            output_row: dict[str, object] = {
                "split": split,
                "subject_id": int(row["subject_id"]),
                "activity_name": row["activity_name"],
                "category": row["category"],
                "sequence_id": get_sequence_id(row),
                "frame_index": int(row["_frame_index"]),
                "raw_frame_code": raw_frame_code(image_name),
                "image_name": image_name,
                "image_path": str(image_path),
                "frame_label": read_frame_label(label_path),
            }
            output_row.update(pose_values(result))
            writer.writerow(output_row)

            if number % REPORT_EVERY == 0 or number == len(remaining):
                file.flush()
                print(
                    f"{split}: processed {number}/{len(remaining)} new frames",
                    flush=True,
                )


def summarize() -> dict[str, object]:
    groups: dict[str, dict[str, float | int]] = {}
    total_frames = 0
    total_pose_found = 0
    for split in SPLITS:
        path = OUTPUT_DIR / f"pose_features_{split}.csv"
        rows = read_csv(path)
        total_frames += len(rows)
        total_pose_found += sum(int(row["pose_found"]) for row in rows)
        for label in (0, 1):
            selected = [row for row in rows if int(row["frame_label"]) == label]
            if not selected:
                continue
            found = sum(int(row["pose_found"]) for row in selected)
            key = f"{split}_class_{label}"
            groups[key] = {
                "frames": len(selected),
                "pose_found": found,
                "pose_detection_rate": found / len(selected),
                "mean_visible_keypoints": float(
                    np.mean([int(row["visible_keypoint_count"]) for row in selected])
                ),
                "mean_keypoint_conf": float(
                    np.mean([float(row["mean_keypoint_conf"]) for row in selected])
                ),
                "boundary_touch_rate": float(
                    np.mean([int(row["boundary_touch_count"]) > 0 for row in selected])
                ),
            }
    return {
        "protocol": {
            "dataset": "CAUCAFall_YOLO",
            "splits_extracted": list(SPLITS),
            "test_extracted": False,
            "pose_model": str(POSE_MODEL_PATH),
            "pose_model_sha256": sha256(POSE_MODEL_PATH),
            "pose_confidence_floor": POSE_CONFIDENCE_FLOOR,
            "visible_keypoint_threshold": VISIBLE_KEYPOINT_THRESHOLD,
            "image_size": IMAGE_SIZE,
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
        },
        "total_frames": total_frames,
        "total_pose_found": total_pose_found,
        "overall_pose_detection_rate": total_pose_found / total_frames,
        "groups": groups,
    }


def main() -> None:
    for required in (DATASET_ROOT, MANIFEST_PATH, POSE_MODEL_PATH):
        if not required.exists():
            raise FileNotFoundError(f"Missing required input: {required}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_csv(MANIFEST_PATH)
    expected_counts = {"train": 11818, "val": 4070}
    for split, expected in expected_counts.items():
        actual = sum(row["split"] == split for row in rows)
        if actual != expected:
            raise RuntimeError(f"{split}: expected {expected} rows, found {actual}")

    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}", flush=True)
    print(f"Pose model: {POSE_MODEL_PATH}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    model = YOLO(str(POSE_MODEL_PATH))

    for split in SPLITS:
        extract_split(model, rows, split)

    summary = summarize()
    summary_path = OUTPUT_DIR / "pose_extraction_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Completed. Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
