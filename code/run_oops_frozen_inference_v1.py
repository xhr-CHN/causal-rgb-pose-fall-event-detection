#!/usr/bin/env python3
"""Run the frozen CAUCAFall-trained RGB/pose fusion model on OOPS test videos."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

import run_le2i_frozen_inference_v1 as base
from train_quality_gated_logit_fusion_v2 import ReliabilityGatedLogitFusion


ROOT = Path("/home/data/yoloA27")
FEATURE_DIR = ROOT / "features/oops_test_frozen_features_v1"
FEATURE_MANIFEST = FEATURE_DIR / "sequence_feature_manifest.csv"
MODEL_PATH = ROOT / "experiments/quality_gated_logit_fusion_v3_w16_fixed_seed42/quality_gated_logit_fusion_v3_best.pt"
OUTPUT_DIR = ROOT / "experiments/caucafall_to_oops_v1"
BATCH_SIZE = 128


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@torch.inference_mode()
def predict(model, pose, rgb, quality, device):
    outputs = []
    for start in range(0, len(pose), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(pose))
        out = model(
            torch.from_numpy(pose[start:end]).to(device),
            torch.from_numpy(rgb[start:end]).to(device),
            torch.from_numpy(quality[start:end]).to(device),
        )
        fused = torch.softmax(out["fused"], dim=1).cpu().numpy()
        outputs.append(fused)
    return np.concatenate(outputs)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT_DIR}")
    if not FEATURE_MANIFEST.is_file() or not MODEL_PATH.is_file():
        raise FileNotFoundError("OOPS feature manifest or fusion checkpoint missing")

    manifest = read_csv(FEATURE_MANIFEST)
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["model_config"]
    model = ReliabilityGatedLogitFusion(
        config["pose_dim"],
        config["rgb_dim"],
        config["quality_dim"],
        config["hidden_dim"],
        config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    all_rows = []
    for index, item in enumerate(manifest, 1):
        pose_rows = read_csv(Path(item["pose_features_csv"]))
        rgb_rows = read_csv(Path(item["rgb_metadata_csv"]))
        embeddings = np.load(item["rgb_embeddings_npy"], mmap_mode="r")
        for row in pose_rows:
            row.setdefault("scene", "OOPS")

        aligned = base.validate_and_attach(pose_rows, rgb_rows, embeddings)
        pose, rgb, quality, metadata, summary = base.make_windows(
            aligned, embeddings, checkpoint
        )
        pose = (
            (pose - np.asarray(checkpoint["pose_mean"])[None, None, :])
            / np.asarray(checkpoint["pose_std"])[None, None, :]
        ).astype(np.float32)
        rgb = (
            (rgb - np.asarray(checkpoint["rgb_mean"])[None, None, :])
            / np.asarray(checkpoint["rgb_std"])[None, None, :]
        ).astype(np.float32)
        probabilities = predict(model, pose, rgb, quality, device)
        all_rows.extend(
            base.prediction_rows(
                metadata,
                {
                    "fused": probabilities,
                    "pose": probabilities,
                    "rgb": probabilities,
                    "pose_weight": np.zeros(len(probabilities)),
                    "disagreement": np.zeros(len(probabilities)),
                },
            )
        )

        if index % 25 == 0 or index == len(manifest):
            print(f"OOPS sequences={index}/{len(manifest)}", flush=True)

    OUTPUT_DIR.mkdir(parents=True)
    prediction_path = OUTPUT_DIR / "frame_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "dataset": "OOPS test subset",
        "source_training_dataset": "CAUCAFall",
        "sequences": len(manifest),
        "prediction_rows": len(all_rows),
        "model": str(MODEL_PATH),
        "labels_read_during_inference": False,
        "output": str(prediction_path),
    }
    (OUTPUT_DIR / "inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
