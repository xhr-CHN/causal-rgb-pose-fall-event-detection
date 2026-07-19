from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import urfd_pose_tcn_3state_external_v1 as external
from build_caucafall_pose_windows_3state_kinematic_v2 import (
    DERIVED_NAMES,
    REMOVED_INDICES,
    derived_kinematics,
)
from train_pose_tcn_baseline_v1 import PoseTCN


CHECKPOINT = Path(
    "/home/data/yoloA27/experiments/pose_tcn_3state_kinematic_v2_seed42/"
    "pose_tcn_3state_kinematic_v2_best.pt"
)
OUTPUT_DIR = Path(
    "/home/data/yoloA27/experiments/"
    "urfd_pose_tcn_3state_kinematic_v2_external"
)
INPUT_FEATURES = 95


def transform_external_windows(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3 or x.shape[1:] != (32, 103):
        raise ValueError(f"Unexpected raw external shape: {x.shape}")
    keep_mask = np.ones(103, dtype=bool)
    keep_mask[REMOVED_INDICES] = False
    transformed = np.concatenate(
        [x[:, :, keep_mask], derived_kinematics(x)], axis=2
    ).astype(np.float32)
    if transformed.shape != (len(x), 32, INPUT_FEATURES):
        raise RuntimeError(f"Unexpected transformed shape: {transformed.shape}")
    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite transformed URFD features")
    return transformed


def make_model() -> PoseTCN:
    model = PoseTCN(input_features=INPUT_FEATURES)
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    return model


@torch.no_grad()
def predict_kinematic_windows(x: np.ndarray, checkpoint: dict) -> np.ndarray:
    transformed = transform_external_windows(x)
    mean = checkpoint["input_mean"].numpy().reshape(1, 1, -1)
    std = checkpoint["input_std"].numpy().reshape(1, 1, -1)
    standardized = (transformed - mean) / std
    if not np.isfinite(standardized).all():
        raise ValueError("Non-finite standardized URFD features")

    model = make_model().to(external.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(standardized.astype(np.float32))),
        batch_size=external.TCN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    outputs = []
    for batch_index, (batch,) in enumerate(loader, start=1):
        logits = model(batch.to(external.DEVICE, non_blocking=True))
        outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        if batch_index % 10 == 0 or batch_index == len(loader):
            print(
                f"Kinematic-v2 inference: {batch_index}/{len(loader)} batches",
                flush=True,
            )
    return np.concatenate(outputs)


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    if checkpoint.get("feature_transform") != (
        "bbox-scale-normalized-global-kinematics-v2"
    ):
        raise ValueError("Unexpected checkpoint feature transform")
    if int(checkpoint["config"]["input_features"]) != INPUT_FEATURES:
        raise ValueError("Unexpected checkpoint input feature count")

    raw_names = list(external.urfd.feature_names())
    keep_mask = np.ones(103, dtype=bool)
    keep_mask[REMOVED_INDICES] = False
    expected_names = [
        *np.asarray(raw_names, dtype=str)[keep_mask].tolist(),
        *DERIVED_NAMES,
    ]
    if list(checkpoint["feature_names"]) != expected_names:
        raise ValueError("Checkpoint kinematic feature names do not match")

    external.CHECKPOINT = CHECKPOINT
    external.OUTPUT_DIR = OUTPUT_DIR
    external.predict_windows = predict_kinematic_windows
    # The shared evaluator checks the checkpoint schema before inference. The
    # raw 103-D windows are transformed inside predict_kinematic_windows.
    external.urfd.feature_names = lambda: expected_names
    external.main()

    summary_path = OUTPUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["protocol"]["model"] = "kinematic-v2 three-state causal Pose-TCN"
    summary["protocol"]["raw_external_features"] = 103
    summary["protocol"]["transformed_model_features"] = INPUT_FEATURES
    summary["protocol"]["feature_transform"] = checkpoint["feature_transform"]
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nCorrected kinematic-v2 summary:", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
