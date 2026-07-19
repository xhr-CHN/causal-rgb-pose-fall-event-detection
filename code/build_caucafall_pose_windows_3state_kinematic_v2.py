from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


INPUT_DIR = Path(
    "/home/data/yoloA27/features/caucafall_pose_windows_3state_v1"
)
OUTPUT_DIR = Path(
    "/home/data/yoloA27/features/"
    "caucafall_pose_windows_3state_kinematic_v2"
)

# Remove camera/framing-dependent channels from the original 103 features.
REMOVED_INDICES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 65, 66, 67, 68]
EXPECTED_REMOVED_NAMES = [
    "bbox_cx",
    "bbox_cy",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "boundary_left",
    "boundary_top",
    "boundary_right",
    "boundary_bottom",
    "delta_bbox_cx",
    "delta_bbox_cy",
    "delta_bbox_w",
    "delta_bbox_h",
]
DERIVED_NAMES = [
    "bbox_aspect_ratio",
    "delta_bbox_cx_over_w",
    "delta_bbox_cy_over_h",
    "delta_bbox_w_over_w",
    "delta_bbox_h_over_h",
]


def derived_kinematics(x: np.ndarray) -> np.ndarray:
    pose_found = x[:, :, 0] > 0.5
    bbox_w = x[:, :, 4]
    bbox_h = x[:, :, 5]
    delta_cx = x[:, :, 65]
    delta_cy = x[:, :, 66]
    delta_w = x[:, :, 67]
    delta_h = x[:, :, 68]

    valid = pose_found & (bbox_w > 1e-5) & (bbox_h > 1e-5)
    aspect = np.zeros_like(bbox_w, dtype=np.float32)
    normalized_cx = np.zeros_like(bbox_w, dtype=np.float32)
    normalized_cy = np.zeros_like(bbox_w, dtype=np.float32)
    normalized_w = np.zeros_like(bbox_w, dtype=np.float32)
    normalized_h = np.zeros_like(bbox_w, dtype=np.float32)

    aspect[valid] = bbox_w[valid] / bbox_h[valid]
    normalized_cx[valid] = delta_cx[valid] / bbox_w[valid]
    normalized_cy[valid] = delta_cy[valid] / bbox_h[valid]
    normalized_w[valid] = delta_w[valid] / bbox_w[valid]
    normalized_h[valid] = delta_h[valid] / bbox_h[valid]

    # Fixed physical plausibility bounds prevent rare detection glitches from
    # dominating source-train statistics. They are not selected on target data.
    aspect = np.clip(aspect, 0.0, 5.0)
    normalized_cx = np.clip(normalized_cx, -2.0, 2.0)
    normalized_cy = np.clip(normalized_cy, -2.0, 2.0)
    normalized_w = np.clip(normalized_w, -2.0, 2.0)
    normalized_h = np.clip(normalized_h, -2.0, 2.0)
    return np.stack(
        [aspect, normalized_cx, normalized_cy, normalized_w, normalized_h],
        axis=2,
    ).astype(np.float32)


def transform_split(split: str) -> dict:
    source = np.load(INPUT_DIR / f"{split}_windows.npz")
    x = source["x"].astype(np.float32)
    y = source["y"].astype(np.int64)
    names = source["feature_names"].astype(str)
    states = source["state_names"].astype(str)
    window_length = int(source["window_length"])
    stride = int(source["stride"])

    if x.ndim != 3 or x.shape[1:] != (32, 103):
        raise ValueError(f"Unexpected {split} shape: {x.shape}")
    if names.shape != (103,):
        raise ValueError(f"Unexpected feature names in {split}: {names.shape}")
    removed_names = names[REMOVED_INDICES].tolist()
    if removed_names != EXPECTED_REMOVED_NAMES:
        raise ValueError(f"Unexpected removed names: {removed_names}")
    if tuple(states.tolist()) != ("ADL", "Falling", "Fallen"):
        raise ValueError(f"Unexpected states: {states.tolist()}")
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError(f"Missing state in {split}: {np.unique(y)}")

    keep_mask = np.ones(103, dtype=bool)
    keep_mask[REMOVED_INDICES] = False
    kept = x[:, :, keep_mask]
    derived = derived_kinematics(x)
    transformed = np.concatenate([kept, derived], axis=2).astype(np.float32)
    transformed_names = np.concatenate(
        [names[keep_mask], np.asarray(DERIVED_NAMES, dtype=str)]
    )

    if transformed.shape != (len(x), 32, 95):
        raise RuntimeError(f"Unexpected transformed shape: {transformed.shape}")
    if transformed_names.shape != (95,):
        raise RuntimeError("Unexpected transformed feature-name count")
    if not np.isfinite(transformed).all():
        raise RuntimeError(f"Non-finite transformed values in {split}")

    np.savez_compressed(
        OUTPUT_DIR / f"{split}_windows.npz",
        x=transformed,
        y=y,
        feature_names=transformed_names,
        window_length=np.asarray(window_length, dtype=np.int64),
        stride=np.asarray(stride, dtype=np.int64),
        state_names=states,
    )
    shutil.copy2(
        INPUT_DIR / f"{split}_window_metadata.csv",
        OUTPUT_DIR / f"{split}_window_metadata.csv",
    )
    state_count = np.bincount(y, minlength=3)
    return {
        "split": split,
        "source_shape": list(x.shape),
        "output_shape": list(transformed.shape),
        "stride": stride,
        "state_counts": {
            "ADL": int(state_count[0]),
            "Falling": int(state_count[1]),
            "Fallen": int(state_count[2]),
        },
        "derived_feature_ranges": {
            name: {
                "minimum": float(derived[:, :, index].min()),
                "maximum": float(derived[:, :, index].max()),
                "mean": float(derived[:, :, index].mean()),
            }
            for index, name in enumerate(DERIVED_NAMES)
        },
    }


def main() -> None:
    required = [
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

    for optional_name in (
        "caucafall_state_annotations.csv",
        "train_sequence_state_counts.csv",
        "val_sequence_state_counts.csv",
    ):
        source = INPUT_DIR / optional_name
        if source.is_file():
            shutil.copy2(source, OUTPUT_DIR / optional_name)

    summaries = {
        split: transform_split(split) for split in ("train", "val")
    }
    summary = {
        "protocol": {
            "task": "three-state causal pose-window classification",
            "source_windows": str(INPUT_DIR),
            "test_used": False,
            "external_data_used": False,
            "window_length": 32,
        },
        "transformation": {
            "removed_feature_indices": REMOVED_INDICES,
            "removed_feature_names": EXPECTED_REMOVED_NAMES,
            "derived_feature_names": DERIVED_NAMES,
            "description": (
                "Replace absolute camera/framing geometry with bbox-scale-"
                "normalized global motion and aspect ratio"
            ),
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
