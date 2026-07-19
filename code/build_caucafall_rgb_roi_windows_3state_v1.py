from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


EMBEDDING_DIR = Path(
    "/home/data/yoloA27/features/caucafall_rgb_roi_embeddings_v1"
)
POSE_WINDOW_DIR = Path(
    "/home/data/yoloA27/features/"
    "caucafall_pose_windows_3state_kinematic_v2"
)
OUTPUT_DIR = Path(
    "/home/data/yoloA27/features/caucafall_rgb_roi_windows_3state_v1"
)
EXPECTED_WINDOWS = {"train": 2513, "val": 3450}
EXPECTED_FRAMES = {"train": 11818, "val": 4070}
EXPECTED_SEQUENCES = {"train": 59, "val": 20}
EXPECTED_SUBJECTS = {"train": {1, 3, 6, 7, 8, 9}, "val": {4, 5}}
WINDOW_LENGTH = 32
EMBEDDING_DIMENSION = 256
STATE_NAMES = np.asarray(["ADL", "Falling", "Fallen"])


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


def frame_groups(rows: list[dict[str, str]]) -> dict[str, list[int]]:
    groups: dict[str, list[tuple[int, int]]] = {}
    for global_index, row in enumerate(rows):
        groups.setdefault(row["sequence_id"], []).append(
            (int(row["frame_index"]), global_index)
        )
    output = {}
    for sequence_id, items in groups.items():
        items.sort()
        frame_indices = [item[0] for item in items]
        if frame_indices != list(range(1, len(items) + 1)):
            raise RuntimeError(f"Non-contiguous embedding sequence: {sequence_id}")
        output[sequence_id] = [item[1] for item in items]
    return output


def build_split(split: str) -> dict:
    embeddings = np.load(
        EMBEDDING_DIR / f"embeddings_{split}.npy", mmap_mode="r"
    )
    frame_rows = read_csv(EMBEDDING_DIR / f"metadata_{split}.csv")
    window_rows = read_csv(POSE_WINDOW_DIR / f"{split}_window_metadata.csv")
    pose_npz = np.load(POSE_WINDOW_DIR / f"{split}_windows.npz")
    pose_y = pose_npz["y"].astype(np.int64)

    if embeddings.shape != (EXPECTED_FRAMES[split], EMBEDDING_DIMENSION):
        raise RuntimeError(f"Unexpected {split} embeddings: {embeddings.shape}")
    if len(frame_rows) != EXPECTED_FRAMES[split]:
        raise RuntimeError(f"Unexpected {split} frame metadata count")
    if len(window_rows) != EXPECTED_WINDOWS[split]:
        raise RuntimeError(f"Unexpected {split} window metadata count")
    if len(pose_y) != len(window_rows):
        raise RuntimeError(f"Pose label/window mismatch in {split}")
    if [int(row["window_id"]) for row in window_rows] != list(
        range(len(window_rows))
    ):
        raise RuntimeError(f"Non-sequential window IDs in {split}")

    subjects = {int(row["subject_id"]) for row in frame_rows}
    if subjects != EXPECTED_SUBJECTS[split]:
        raise RuntimeError(f"Unexpected {split} subjects: {sorted(subjects)}")
    groups = frame_groups(frame_rows)
    if len(groups) != EXPECTED_SEQUENCES[split]:
        raise RuntimeError(f"Unexpected {split} sequence count: {len(groups)}")

    x = np.empty(
        (len(window_rows), WINDOW_LENGTH, EMBEDDING_DIMENSION),
        dtype=np.float32,
    )
    y = np.empty(len(window_rows), dtype=np.int64)
    output_metadata = []
    for index, row in enumerate(window_rows):
        sequence_id = row["sequence_id"]
        if sequence_id not in groups:
            raise RuntimeError(f"Missing RGB sequence: {sequence_id}")
        start = int(row["start_frame_index"])
        target = int(row["target_frame_index"])
        if target - start + 1 != WINDOW_LENGTH:
            raise RuntimeError(f"Unexpected window length: {sequence_id}/{target}")
        global_indices = groups[sequence_id][start - 1 : target]
        if len(global_indices) != WINDOW_LENGTH:
            raise RuntimeError(f"Incomplete RGB window: {sequence_id}/{target}")
        x[index] = embeddings[global_indices]
        target_state = int(row["target_state_id"])
        if target_state != int(pose_y[index]):
            raise RuntimeError(f"RGB/Pose state mismatch at window {index}")
        y[index] = target_state

        crop_flags = [int(frame_rows[item]["roi_crop_used"]) for item in global_indices]
        item = dict(row)
        item["rgb_roi_crop_frames_in_window"] = sum(crop_flags)
        item["rgb_full_image_fallback_frames_in_window"] = (
            WINDOW_LENGTH - sum(crop_flags)
        )
        output_metadata.append(item)

    if not np.isfinite(x).all():
        raise RuntimeError(f"Non-finite RGB windows in {split}")
    if set(np.unique(y)) != {0, 1, 2}:
        raise RuntimeError(f"Not all states occur in {split}: {np.unique(y)}")

    np.savez_compressed(
        OUTPUT_DIR / f"{split}_windows.npz",
        x=x,
        y=y,
        feature_names=np.asarray(
            [f"rgb_embedding_{index:03d}" for index in range(EMBEDDING_DIMENSION)]
        ),
        window_length=np.asarray(WINDOW_LENGTH, dtype=np.int64),
        stride=np.asarray(int(pose_npz["stride"]), dtype=np.int64),
        state_names=STATE_NAMES,
    )
    save_csv(OUTPUT_DIR / f"{split}_window_metadata.csv", output_metadata)

    counts = Counter(y.tolist())
    fallback_windows = sum(
        int(row["rgb_full_image_fallback_frames_in_window"]) > 0
        for row in output_metadata
    )
    return {
        "split": split,
        "frames": len(frame_rows),
        "sequences": len(groups),
        "subjects": sorted(subjects),
        "windows": len(window_rows),
        "shape": list(x.shape),
        "stride": int(pose_npz["stride"]),
        "state_counts": {
            "ADL": counts[0],
            "Falling": counts[1],
            "Fallen": counts[2],
        },
        "windows_with_full_image_fallback": fallback_windows,
        "fallback_window_rate": fallback_windows / len(output_metadata),
    }


def main() -> None:
    required = []
    for split in ("train", "val"):
        required.extend(
            [
                EMBEDDING_DIR / f"embeddings_{split}.npy",
                EMBEDDING_DIR / f"metadata_{split}.csv",
                POSE_WINDOW_DIR / f"{split}_windows.npz",
                POSE_WINDOW_DIR / f"{split}_window_metadata.csv",
            ]
        )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    summaries = {split: build_split(split) for split in ("train", "val")}
    if set(summaries["train"]["subjects"]) & set(summaries["val"]["subjects"]):
        raise RuntimeError("Train/validation subject leakage")
    summary = {
        "protocol": {
            "task": "three-state causal RGB-ROI embedding windows",
            "source_embeddings": str(EMBEDDING_DIR),
            "alignment_reference": str(POSE_WINDOW_DIR),
            "window_length": WINDOW_LENGTH,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "test_used": False,
            "external_data_used": False,
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
