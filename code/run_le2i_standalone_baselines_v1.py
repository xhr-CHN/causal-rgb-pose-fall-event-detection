#!/usr/bin/env python3
"""Run independently trained Pose-TCN and RGB-TCN checkpoints on Le2i.

This script uses the existing label-free Le2i frozen feature manifest and
does not read Le2i annotations. It produces frame/window probabilities only;
event-level scoring is performed by a separate locked evaluation step.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import run_le2i_frozen_inference_v1 as le2i
from train_pose_tcn_baseline_v1 import PoseTCN


ROOT = Path("/home/data/yoloA27")
FEATURE_DIR = ROOT / "features/le2i_frozen_features_v1"
FEATURE_MANIFEST = FEATURE_DIR / "sequence_feature_manifest.csv"
DEFAULT_OUTPUT = ROOT / "experiments/le2i_standalone_baselines_v2"
POSE_CHECKPOINT = ROOT / "experiments/pose_tcn_baseline_v1_seed42/pose_tcn_baseline_best.pt"
RGB_CHECKPOINT = ROOT / "experiments/rgb_roi_tcn_3state_v1_seed42/rgb_roi_tcn_3state_best.pt"
BATCH_SIZE = 256


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def transform_pose_baseline(
    resampled: list[tuple[int, float, dict[str, str]]],
    expected_names: list[str],
) -> np.ndarray:
    """Reconstruct the 103-feature Pose-TCN representation."""
    vectors = []
    previous = None
    for _, _, row in resampled:
        values = {
            "pose_found": le2i.number(row, "pose_found"),
            "person_conf": le2i.number(row, "person_conf"),
            "bbox_cx": le2i.number(row, "bbox_cx"),
            "bbox_cy": le2i.number(row, "bbox_cy"),
            "bbox_w": le2i.number(row, "bbox_w"),
            "bbox_h": le2i.number(row, "bbox_h"),
            "bbox_area": le2i.number(row, "bbox_area"),
            "boundary_left": le2i.number(row, "boundary_left"),
            "boundary_top": le2i.number(row, "boundary_top"),
            "boundary_right": le2i.number(row, "boundary_right"),
            "boundary_bottom": le2i.number(row, "boundary_bottom"),
            "visible_keypoint_ratio": le2i.number(row, "visible_keypoint_count") / 17.0,
            "mean_keypoint_conf": le2i.number(row, "mean_keypoint_conf"),
            "torso_keypoint_conf": le2i.number(row, "torso_keypoint_conf"),
        }
        for joint in le2i.JOINT_NAMES:
            values[f"{joint}_x_bbox"] = le2i.number(row, f"{joint}_x_bbox")
            values[f"{joint}_y_bbox"] = le2i.number(row, f"{joint}_y_bbox")
            values[f"{joint}_conf"] = le2i.number(row, f"{joint}_conf")

        for name in ("bbox_cx", "bbox_cy", "bbox_w", "bbox_h"):
            values[f"delta_{name}"] = 0.0 if previous is None else (
                values[name] - le2i.number(previous, name)
                if int(values["pose_found"]) == 1 and int(le2i.number(previous, "pose_found")) == 1
                else 0.0
            )
        for joint in le2i.JOINT_NAMES:
            for axis in ("x", "y"):
                name = f"{joint}_{axis}_bbox"
                conf = values[f"{joint}_conf"]
                prev_conf = 0.0 if previous is None else le2i.number(previous, f"{joint}_conf")
                values[f"delta_{name}"] = (
                    values[name] - le2i.number(previous, name)
                    if previous is not None and conf >= 0.05 and prev_conf >= 0.05
                    else 0.0
                )

        missing = [name for name in expected_names if name not in values]
        if missing:
            raise ValueError(f"Cannot construct Pose-TCN features: {missing[:10]}")
        vectors.append([values[name] for name in expected_names])
        previous = row
    return np.asarray(vectors, dtype=np.float32)


def make_windows(
    pose_rows: list[dict[str, str]],
    rgb_rows: list[dict[str, str]],
    embeddings: np.ndarray,
    feature_names: list[str],
    window_length: int,
    sequence_id: str,
    scene: str,
    need_pose: bool,
) -> tuple[np.ndarray | None, np.ndarray, list[dict[str, str]]]:
    aligned = le2i.validate_and_attach(pose_rows, rgb_rows, embeddings)
    resampled = le2i.nearest_resample(aligned)
    pose = transform_pose_baseline(resampled, feature_names) if need_pose else None
    rgb = np.stack(
        [embeddings[int(row["rgb_index"])] for _, _, row in resampled]
    ).astype(np.float32)

    pose_windows = []
    rgb_windows = []
    metadata = []
    for end in range(window_length - 1, len(resampled)):
        start = end - window_length + 1
        sample_index, sample_timestamp, row = resampled[end]
        if pose is not None:
            pose_windows.append(pose[start : end + 1])
        rgb_windows.append(rgb[start : end + 1])
        metadata.append(
            {
                "sequence_id": sequence_id,
                "scene": scene,
                "sample_index": str(sample_index),
                "sample_timestamp_ms": str(int(round(sample_timestamp))),
                "source_frame_number": str(
                    le2i.integer(row, "frame_number")
                ),
                "source_timestamp_ms": str(
                    int(round(le2i.number(row, "timestamp_ms")))
                ),
            }
        )
    if not metadata:
        raise RuntimeError(f"No windows for {sequence_id}")
    return (
        None if pose is None else np.stack(pose_windows).astype(np.float32),
        np.stack(rgb_windows).astype(np.float32),
        metadata,
    )


def load_model(checkpoint_path: Path, input_features: int, classes: int) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = PoseTCN(input_features=input_features)
    model.classifier = nn.Linear(model.classifier.in_features, classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


@torch.inference_mode()
def predict(model: torch.nn.Module, x: np.ndarray, mean, std, device: torch.device) -> np.ndarray:
    x = (x - np.asarray(mean)[None, None, :]) / np.asarray(std)[None, None, :]
    out = []
    for start in range(0, len(x), BATCH_SIZE):
        batch = torch.from_numpy(x[start : start + BATCH_SIZE]).to(device)
        out.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(out)


def run_baseline(name: str, checkpoint_path: Path, output_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    input_features = int(config["input_features"])
    classes = 2 if name == "pose_tcn" else 3
    model, checkpoint = load_model(checkpoint_path, input_features, classes)
    model.to(device).eval()

    manifest = read_rows(FEATURE_MANIFEST)
    all_rows = []
    for index, item in enumerate(manifest, 1):
        pose_rows = read_rows(Path(item["pose_features_csv"]))
        rgb_rows = read_rows(Path(item["rgb_metadata_csv"]))
        embeddings = np.load(item["rgb_embeddings_npy"], mmap_mode="r")
        pose, rgb, metadata = make_windows(
            pose_rows,
            rgb_rows,
            embeddings,
            list(checkpoint["feature_names"]),
            int(config["window_length"]),
            item["sequence_id"],
            item.get("scene", ""),
            name == "pose_tcn",
        )
        x = pose if name == "pose_tcn" else rgb
        probabilities = predict(
            model,
            x,
            checkpoint["input_mean"].numpy(),
            checkpoint["input_std"].numpy(),
            device,
        )
        for meta, prob in zip(metadata, probabilities):
            row = dict(meta)
            if name == "pose_tcn":
                row.update(
                    {
                        "prob_adl": float(prob[0]),
                        "prob_falling": float(prob[1]),
                        "prob_fallen": 0.0,
                    }
                )
            else:
                row.update(
                    {
                        "prob_adl": float(prob[0]),
                        "prob_falling": float(prob[1]),
                        "prob_fallen": float(prob[2]),
                    }
                )
            all_rows.append(row)
        if index % 25 == 0:
            print(f"{name}: sequences={index}/{len(manifest)}", flush=True)

    fields = list(all_rows[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    return {"baseline": name, "rows": len(all_rows), "sequences": len(manifest), "checkpoint": str(checkpoint_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {args.output_dir}"
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    summaries = [
        run_baseline("pose_tcn", POSE_CHECKPOINT, args.output_dir / "pose_tcn_frame_predictions.csv", device),
        run_baseline("rgb_tcn", RGB_CHECKPOINT, args.output_dir / "rgb_tcn_frame_predictions.csv", device),
    ]
    (args.output_dir / "summary.json").write_text(
        json.dumps({"device": str(device), "baselines": summaries}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"device": str(device), "baselines": summaries}, indent=2))


if __name__ == "__main__":
    main()
