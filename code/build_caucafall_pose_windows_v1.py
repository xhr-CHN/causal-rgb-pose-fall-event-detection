from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


INPUT_DIR = Path("/home/data/yoloA27/features/caucafall_pose_v1")
OUTPUT_DIR = Path("/home/data/yoloA27/features/caucafall_pose_windows_v1")

WINDOW_LENGTH = 32
TRAIN_STRIDE = 4
VAL_STRIDE = 1
DELTA_CONFIDENCE_THRESHOLD = 0.05

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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def feature_names() -> list[str]:
    names = [
        "pose_found",
        "person_conf",
        "bbox_cx",
        "bbox_cy",
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "boundary_left",
        "boundary_top",
        "boundary_right",
        "boundary_bottom",
        "visible_keypoint_ratio",
        "mean_keypoint_conf",
        "torso_keypoint_conf",
    ]
    for keypoint in KEYPOINT_NAMES:
        names.extend(
            [
                f"{keypoint}_x_bbox",
                f"{keypoint}_y_bbox",
                f"{keypoint}_conf",
            ]
        )
    names.extend(["delta_bbox_cx", "delta_bbox_cy", "delta_bbox_w", "delta_bbox_h"])
    for keypoint in KEYPOINT_NAMES:
        names.extend([f"delta_{keypoint}_x_bbox", f"delta_{keypoint}_y_bbox"])
    return names


def base_feature(row: dict[str, str]) -> list[float]:
    values = [
        float(row["pose_found"]),
        float(row["person_conf"]),
        float(row["bbox_cx"]),
        float(row["bbox_cy"]),
        float(row["bbox_w"]),
        float(row["bbox_h"]),
        float(row["bbox_area"]),
        float(row["boundary_left"]),
        float(row["boundary_top"]),
        float(row["boundary_right"]),
        float(row["boundary_bottom"]),
        float(row["visible_keypoint_count"]) / len(KEYPOINT_NAMES),
        float(row["mean_keypoint_conf"]),
        float(row["torso_keypoint_conf"]),
    ]
    for keypoint in KEYPOINT_NAMES:
        values.extend(
            [
                float(row[f"{keypoint}_x_bbox"]),
                float(row[f"{keypoint}_y_bbox"]),
                float(row[f"{keypoint}_conf"]),
            ]
        )
    return values


def delta_feature(
    current: dict[str, str], previous: dict[str, str] | None
) -> list[float]:
    if previous is None:
        return [0.0] * (4 + 2 * len(KEYPOINT_NAMES))

    both_poses = int(current["pose_found"]) == 1 and int(previous["pose_found"]) == 1
    if both_poses:
        bbox_delta = [
            float(current[name]) - float(previous[name])
            for name in ("bbox_cx", "bbox_cy", "bbox_w", "bbox_h")
        ]
    else:
        bbox_delta = [0.0] * 4

    keypoint_delta: list[float] = []
    for keypoint in KEYPOINT_NAMES:
        current_conf = float(current[f"{keypoint}_conf"])
        previous_conf = float(previous[f"{keypoint}_conf"])
        if (
            current_conf >= DELTA_CONFIDENCE_THRESHOLD
            and previous_conf >= DELTA_CONFIDENCE_THRESHOLD
        ):
            keypoint_delta.extend(
                [
                    float(current[f"{keypoint}_x_bbox"])
                    - float(previous[f"{keypoint}_x_bbox"]),
                    float(current[f"{keypoint}_y_bbox"])
                    - float(previous[f"{keypoint}_y_bbox"]),
                ]
            )
        else:
            keypoint_delta.extend([0.0, 0.0])
    return bbox_delta + keypoint_delta


def vectorize_sequence(rows: list[dict[str, str]]) -> np.ndarray:
    vectors = []
    previous = None
    for row in rows:
        vectors.append(base_feature(row) + delta_feature(row, previous))
        previous = row
    array = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Feature array contains NaN or infinity")
    return array


def group_sequences(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["sequence_id"], []).append(row)
    sequences = []
    for sequence_id in sorted(groups):
        sequence = sorted(groups[sequence_id], key=lambda row: int(row["frame_index"]))
        expected = list(range(1, len(sequence) + 1))
        actual = [int(row["frame_index"]) for row in sequence]
        if actual != expected:
            raise ValueError(f"Non-contiguous sequence: {sequence_id}")
        sequences.append(sequence)
    return sequences


def build_split(split: str, stride: int) -> dict[str, object]:
    input_path = INPUT_DIR / f"pose_features_{split}.csv"
    rows = read_csv(input_path)
    sequences = group_sequences(rows)
    names = feature_names()

    windows: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, object]] = []

    for sequence in sequences:
        features = vectorize_sequence(sequence)
        for target_index in range(WINDOW_LENGTH - 1, len(sequence), stride):
            start = target_index - WINDOW_LENGTH + 1
            window = features[start : target_index + 1]
            if window.shape != (WINDOW_LENGTH, len(names)):
                raise ValueError(f"Unexpected window shape: {window.shape}")

            target = sequence[target_index]
            window_rows = sequence[start : target_index + 1]
            label = int(target["frame_label"])
            windows.append(window)
            labels.append(label)
            metadata.append(
                {
                    "window_id": len(windows) - 1,
                    "split": split,
                    "sequence_id": target["sequence_id"],
                    "subject_id": int(target["subject_id"]),
                    "activity_name": target["activity_name"],
                    "category": target["category"],
                    "start_frame_index": int(window_rows[0]["frame_index"]),
                    "target_frame_index": int(target["frame_index"]),
                    "target_raw_frame_code": int(target["raw_frame_code"]),
                    "target_image_name": target["image_name"],
                    "target_label": label,
                    "positive_frames_in_window": sum(
                        int(row["frame_label"]) for row in window_rows
                    ),
                    "pose_found_ratio": float(
                        np.mean([int(row["pose_found"]) for row in window_rows])
                    ),
                    "mean_keypoint_conf": float(
                        np.mean(
                            [float(row["mean_keypoint_conf"]) for row in window_rows]
                        )
                    ),
                    "boundary_touch_ratio": float(
                        np.mean(
                            [
                                int(row["boundary_touch_count"]) > 0
                                for row in window_rows
                            ]
                        )
                    ),
                }
            )

    x = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if not np.isfinite(x).all():
        raise ValueError(f"{split}: window tensor contains NaN or infinity")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT_DIR / f"{split}_windows.npz"
    np.savez_compressed(
        npz_path,
        x=x,
        y=y,
        feature_names=np.asarray(names),
        window_length=np.asarray(WINDOW_LENGTH),
        stride=np.asarray(stride),
    )

    metadata_path = OUTPUT_DIR / f"{split}_window_metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metadata[0].keys()))
        writer.writeheader()
        writer.writerows(metadata)

    counts = Counter(labels)
    return {
        "split": split,
        "source_frames": len(rows),
        "sequences": len(sequences),
        "subjects": sorted({int(row["subject_id"]) for row in rows}),
        "stride": stride,
        "windows": len(windows),
        "class_0_windows": counts[0],
        "class_1_windows": counts[1],
        "positive_window_ratio": counts[1] / len(windows),
        "tensor_shape": list(x.shape),
        "feature_dimension": x.shape[-1],
        "npz_path": str(npz_path),
        "metadata_path": str(metadata_path),
    }


def main() -> None:
    required = [
        INPUT_DIR / "pose_features_train.csv",
        INPUT_DIR / "pose_features_val.csv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    train_summary = build_split("train", TRAIN_STRIDE)
    print(json.dumps(train_summary, indent=2), flush=True)
    val_summary = build_split("val", VAL_STRIDE)
    print(json.dumps(val_summary, indent=2), flush=True)

    summary = {
        "protocol": {
            "causal_windows": True,
            "window_label": "label of final frame",
            "window_length": WINDOW_LENGTH,
            "train_stride": TRAIN_STRIDE,
            "val_stride": VAL_STRIDE,
            "test_used": False,
            "delta_confidence_threshold": DELTA_CONFIDENCE_THRESHOLD,
        },
        "feature_names": feature_names(),
        "train": train_summary,
        "val": val_summary,
    }
    summary_path = OUTPUT_DIR / "window_build_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Completed. Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
