from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


INPUT_DIR = Path("/home/data/yoloA27/features/caucafall_pose_windows_v1")
ANNOTATION_PATH = Path("/home/data/yoloA27/caucafall_state_annotations.csv")
OUTPUT_DIR = Path("/home/data/yoloA27/features/caucafall_pose_windows_3state_v1")

STATE_NAMES = {0: "ADL", 1: "Falling", 2: "Fallen"}
EXPECTED_SPLIT_SUBJECTS = {
    "train": {1, 3, 6, 7, 8, 9},
    "val": {4, 5},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot save empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_annotations() -> dict[str, dict[str, str]]:
    rows = read_csv(ANNOTATION_PATH)
    if len(rows) != 39:
        raise RuntimeError(f"Expected 39 annotations, found {len(rows)}")
    if len({row["sequence_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate sequence_id in annotations")
    if any(row["split"] == "test" for row in rows):
        raise RuntimeError("Test annotations must not be present")
    if any(row["status"] != "checked" for row in rows):
        pending = [row["sequence_id"] for row in rows if row["status"] != "checked"]
        raise RuntimeError(f"Unchecked annotations: {pending}")

    for row in rows:
        onset = int(row["onset_frame"])
        stable = int(row["stable_fallen_frame"])
        total = int(row["total_frames"])
        if not (1 <= onset <= stable <= total):
            raise RuntimeError(f"Invalid markers: {row['sequence_id']}")
        if float(row["fps"]) != 20.0:
            raise RuntimeError(f"Unexpected FPS: {row['sequence_id']}")
    return {row["sequence_id"]: row for row in rows}


def state_for_frame(
    category: str,
    frame_index: int,
    annotation: dict[str, str] | None,
) -> int:
    if category == "adl":
        return 0
    if category != "fall" or annotation is None:
        raise ValueError(f"Missing fall annotation for category={category}")
    onset = int(annotation["onset_frame"])
    stable = int(annotation["stable_fallen_frame"])
    if frame_index < onset:
        return 0
    if frame_index < stable:
        return 1
    return 2


def build_split(split: str, annotations: dict[str, dict[str, str]]) -> dict:
    input_npz = np.load(INPUT_DIR / f"{split}_windows.npz")
    metadata = read_csv(INPUT_DIR / f"{split}_window_metadata.csv")
    x = input_npz["x"].astype(np.float32)
    old_y = input_npz["y"].astype(np.int64)
    feature_names = input_npz["feature_names"]
    window_length = int(input_npz["window_length"])
    stride = int(input_npz["stride"])

    if len(x) != len(old_y) or len(x) != len(metadata):
        raise RuntimeError(f"Length mismatch in {split}")
    if x.shape[1:] != (32, 103) or window_length != 32:
        raise RuntimeError(f"Unexpected window shape in {split}: {x.shape}")
    if [int(row["window_id"]) for row in metadata] != list(range(len(metadata))):
        raise RuntimeError(f"Non-sequential window IDs in {split}")

    fall_sequences_in_metadata = {
        row["sequence_id"] for row in metadata if row["category"] == "fall"
    }
    annotation_sequences = {
        sequence_id
        for sequence_id, row in annotations.items()
        if row["split"] == split
    }
    if fall_sequences_in_metadata != annotation_sequences:
        missing = sorted(fall_sequences_in_metadata - annotation_sequences)
        extra = sorted(annotation_sequences - fall_sequences_in_metadata)
        raise RuntimeError(
            f"Annotation/metadata mismatch in {split}: missing={missing}, extra={extra}"
        )

    subjects = {int(row["subject_id"]) for row in metadata}
    if subjects != EXPECTED_SPLIT_SUBJECTS[split]:
        raise RuntimeError(f"Unexpected {split} subjects: {sorted(subjects)}")

    new_y = np.zeros(len(metadata), dtype=np.int64)
    output_metadata = []
    sequence_state_counts: dict[str, Counter] = {}
    for index, row in enumerate(metadata):
        sequence_id = row["sequence_id"]
        category = row["category"]
        annotation = annotations.get(sequence_id)
        start = int(row["start_frame_index"])
        target = int(row["target_frame_index"])
        target_state = state_for_frame(category, target, annotation)
        expected_binary = int(target_state in (1, 2))
        if expected_binary != int(old_y[index]):
            raise RuntimeError(
                f"Binary/three-state onset mismatch: {sequence_id}, frame={target}, "
                f"binary={old_y[index]}, state={target_state}"
            )

        window_states = [
            state_for_frame(category, frame, annotation)
            for frame in range(start, target + 1)
        ]
        counts = Counter(window_states)
        new_y[index] = target_state
        sequence_state_counts.setdefault(sequence_id, Counter())[target_state] += 1

        item = dict(row)
        item["binary_target_label"] = int(row["target_label"])
        item["target_state_id"] = target_state
        item["target_state"] = STATE_NAMES[target_state]
        item["adl_frames_in_window"] = counts[0]
        item["falling_frames_in_window"] = counts[1]
        item["fallen_frames_in_window"] = counts[2]
        item["onset_frame"] = int(annotation["onset_frame"]) if annotation else ""
        item["stable_fallen_frame"] = (
            int(annotation["stable_fallen_frame"]) if annotation else ""
        )
        output_metadata.append(item)

    if not np.isfinite(x).all():
        raise RuntimeError(f"Non-finite features in {split}")
    if set(np.unique(new_y)) != {0, 1, 2}:
        raise RuntimeError(f"Not all three states exist in {split}: {np.unique(new_y)}")

    np.savez_compressed(
        OUTPUT_DIR / f"{split}_windows.npz",
        x=x,
        y=new_y,
        feature_names=feature_names,
        window_length=np.asarray(window_length, dtype=np.int64),
        stride=np.asarray(stride, dtype=np.int64),
        state_names=np.asarray(["ADL", "Falling", "Fallen"]),
    )
    save_csv(OUTPUT_DIR / f"{split}_window_metadata.csv", output_metadata)

    per_sequence = []
    sequence_info = {
        row["sequence_id"]: row for row in metadata
    }
    for sequence_id in sorted(sequence_state_counts):
        counts = sequence_state_counts[sequence_id]
        info = sequence_info[sequence_id]
        per_sequence.append(
            {
                "split": split,
                "sequence_id": sequence_id,
                "subject_id": int(info["subject_id"]),
                "activity_name": info["activity_name"],
                "category": info["category"],
                "adl_windows": counts[0],
                "falling_windows": counts[1],
                "fallen_windows": counts[2],
                "total_windows": sum(counts.values()),
            }
        )
    save_csv(OUTPUT_DIR / f"{split}_sequence_state_counts.csv", per_sequence)

    class_counts = Counter(new_y.tolist())
    return {
        "split": split,
        "windows": len(new_y),
        "shape": list(x.shape),
        "stride": stride,
        "subjects": sorted(subjects),
        "sequences": len(sequence_state_counts),
        "fall_sequences": len(fall_sequences_in_metadata),
        "state_counts": {
            STATE_NAMES[class_id]: class_counts[class_id] for class_id in range(3)
        },
        "state_percentages": {
            STATE_NAMES[class_id]: class_counts[class_id] / len(new_y)
            for class_id in range(3)
        },
    }


def main() -> None:
    required = [
        ANNOTATION_PATH,
        INPUT_DIR / "train_windows.npz",
        INPUT_DIR / "val_windows.npz",
        INPUT_DIR / "train_window_metadata.csv",
        INPUT_DIR / "val_window_metadata.csv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    annotations = load_annotations()
    shutil.copy2(ANNOTATION_PATH, OUTPUT_DIR / ANNOTATION_PATH.name)
    summaries = {
        split: build_split(split, annotations) for split in ("train", "val")
    }
    train_subjects = set(summaries["train"]["subjects"])
    val_subjects = set(summaries["val"]["subjects"])
    if train_subjects & val_subjects:
        raise RuntimeError("Train/validation subject leakage")

    summary = {
        "protocol": {
            "task": "three-state causal pose-window classification",
            "states": STATE_NAMES,
            "window_length": 32,
            "source_binary_windows_reused": True,
            "pose_features_recomputed": False,
            "test_used": False,
            "annotation_rows": len(annotations),
        },
        "splits": summaries,
    }
    (OUTPUT_DIR / "window_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Completed. Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
