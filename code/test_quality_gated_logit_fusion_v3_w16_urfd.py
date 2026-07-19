#!/usr/bin/env python3
"""Evaluate quality-gated late-logit fusion V3 on URFD.

URFD is treated as an external development benchmark. The V3 checkpoint,
source normalization, and alarm state machine remain locked from CAUCAFall.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from train_quality_gated_fusion_tcn import (
    STATE_NAMES,
    json_ready,
    multiclass_metrics,
    write_csv_rows,
)
from train_quality_gated_logit_fusion_v2 import ReliabilityGatedLogitFusion
from test_quality_gated_fusion_urfd import (
    BATCH_SIZE,
    POSE_CSV,
    RGB_EMBEDDINGS,
    RGB_METADATA,
    ROOT,
    calibration_metrics,
    collapsed_metrics,
    construct_windows,
    number,
    read_rows,
    run_state_machine,
    validate_and_attach_embeddings,
)


MODEL_PATH = (
    ROOT
    / "experiments/quality_gated_logit_fusion_v3_w16_fixed_seed42/quality_gated_logit_fusion_v3_best.pt"
)
MANUAL_EVENTS = ROOT / "URFD/metadata/manual_event_annotations.csv"
OUT_DIR = ROOT / "experiments/urfd_quality_gated_logit_fusion_v3_w16_external"


@torch.no_grad()
def predict(model, pose, rgb, quality, device):
    fused_probabilities = []
    pose_probabilities = []
    rgb_probabilities = []
    pose_weights = []
    disagreements = []
    model.eval()

    for start in range(0, len(pose), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(pose))
        output = model(
            torch.from_numpy(pose[start:end]).to(device),
            torch.from_numpy(rgb[start:end]).to(device),
            torch.from_numpy(quality[start:end]).to(device),
        )
        fused_probabilities.append(torch.softmax(output["fused"], 1).cpu().numpy())
        pose_probabilities.append(output["pose_probabilities"].cpu().numpy())
        rgb_probabilities.append(output["rgb_probabilities"].cpu().numpy())
        pose_weights.append(output["pose_weight"].cpu().numpy())
        disagreements.append(output["branch_disagreement"].cpu().numpy())
        if end % 1000 < BATCH_SIZE or end == len(pose):
            print(f"Inference: {end}/{len(pose)}", flush=True)

    return {
        "fused": np.concatenate(fused_probabilities),
        "pose": np.concatenate(pose_probabilities),
        "rgb": np.concatenate(rgb_probabilities),
        "pose_weight": np.concatenate(pose_weights),
        "disagreement": np.concatenate(disagreements),
    }


def build_manual_event_boundaries(aligned_rows):
    annotations = read_rows(MANUAL_EVENTS)
    grouped = defaultdict(list)
    for row in aligned_rows:
        grouped[row["sequence_id"]].append(row)

    boundaries = {}
    for annotation in annotations:
        sequence_id = annotation["sequence_id"]
        if sequence_id not in grouped:
            raise ValueError(f"Manual event sequence missing from features: {sequence_id}")
        frame_rows = {
            int(float(row["frame_number"])): row for row in grouped[sequence_id]
        }
        onset_frame = int(float(annotation["onset_frame"]))
        stable_frame = int(float(annotation["stable_fallen_frame"]))
        if onset_frame not in frame_rows or stable_frame not in frame_rows:
            raise ValueError(f"Manual event frame missing in {sequence_id}")
        boundaries[sequence_id] = {
            "onset_frame": onset_frame,
            "onset_timestamp_ms": int(
                round(number(frame_rows[onset_frame], "timestamp_ms"))
            ),
            "stable_fallen_frame": stable_frame,
            "stable_fallen_timestamp_ms": int(
                round(number(frame_rows[stable_frame], "timestamp_ms"))
            ),
        }
    if len(boundaries) != 30:
        raise ValueError(f"Expected 30 manual fall events, found {len(boundaries)}")
    return boundaries


def safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std() < 1e-12 or right.std() < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    config = checkpoint["model_config"]
    model = ReliabilityGatedLogitFusion(
        config["pose_dim"],
        config["rgb_dim"],
        config["quality_dim"],
        config["hidden_dim"],
        config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model: {MODEL_PATH}")
    print(f"GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print("Target data used for parameter tuning: NO", flush=True)

    pose_rows = read_rows(POSE_CSV)
    rgb_rows = read_rows(RGB_METADATA)
    embeddings = np.load(RGB_EMBEDDINGS, mmap_mode="r")
    aligned = validate_and_attach_embeddings(pose_rows, rgb_rows, embeddings)
    pose, rgb, quality, labels, metadata, resampling_rows = construct_windows(
        aligned, embeddings, checkpoint
    )
    print(f"Aligned raw frames: {len(aligned)}")
    print(f"Causal windows: {len(labels)}")

    pose = (
        pose - np.asarray(checkpoint["pose_mean"])[None, None, :]
    ) / np.asarray(checkpoint["pose_std"])[None, None, :]
    rgb = (
        rgb - np.asarray(checkpoint["rgb_mean"])[None, None, :]
    ) / np.asarray(checkpoint["rgb_std"])[None, None, :]
    pose = pose.astype(np.float32)
    rgb = rgb.astype(np.float32)

    outputs = predict(model, pose, rgb, quality, device)
    probabilities = outputs["fused"]
    predictions = probabilities.argmax(1)
    fused_metrics = multiclass_metrics(labels, probabilities)
    pose_branch_metrics = multiclass_metrics(labels, outputs["pose"])
    rgb_branch_metrics = multiclass_metrics(labels, outputs["rgb"])
    brier, ece = calibration_metrics(labels, probabilities)
    fused_metrics["multiclass_brier_score"] = brier
    fused_metrics["top_label_ece_10_bins"] = ece
    binary_metrics = collapsed_metrics(labels, probabilities)
    event_metrics, event_rows = run_state_machine(
        metadata,
        predictions,
        checkpoint["state_machine"],
        build_manual_event_boundaries(aligned),
    )

    prediction_rows = []
    for index, source in enumerate(metadata):
        row = dict(source)
        row.update(
            {
                "target_state_id": int(labels[index]),
                "target_state": STATE_NAMES[labels[index]],
                "prob_adl": float(probabilities[index, 0]),
                "prob_falling": float(probabilities[index, 1]),
                "prob_fallen": float(probabilities[index, 2]),
                "pose_prob_adl": float(outputs["pose"][index, 0]),
                "pose_prob_falling": float(outputs["pose"][index, 1]),
                "pose_prob_fallen": float(outputs["pose"][index, 2]),
                "rgb_prob_adl": float(outputs["rgb"][index, 0]),
                "rgb_prob_falling": float(outputs["rgb"][index, 1]),
                "rgb_prob_fallen": float(outputs["rgb"][index, 2]),
                "predicted_state_id": int(predictions[index]),
                "predicted_state": STATE_NAMES[predictions[index]],
                "pose_gate_weight": float(outputs["pose_weight"][index]),
                "branch_disagreement": float(outputs["disagreement"][index]),
            }
        )
        prediction_rows.append(row)
    write_csv_rows(OUT_DIR / "frame_predictions.csv", prediction_rows)
    write_csv_rows(OUT_DIR / "event_results.csv", event_rows)
    write_csv_rows(OUT_DIR / "urfd_resampling_summary.csv", resampling_rows)

    weights = outputs["pose_weight"]
    gate_stats = {
        "overall": {
            "count": len(weights),
            "mean_pose_weight": float(weights.mean()),
            "std_pose_weight": float(weights.std()),
            "minimum_pose_weight": float(weights.min()),
            "maximum_pose_weight": float(weights.max()),
            "correlation_with_pose_found_ratio": safe_correlation(
                [row["pose_found_ratio"] for row in metadata], weights
            ),
            "correlation_with_mean_keypoint_conf": safe_correlation(
                [row["mean_keypoint_conf"] for row in metadata], weights
            ),
            "correlation_with_branch_disagreement": safe_correlation(
                outputs["disagreement"], weights
            ),
        }
    }
    for class_id, name in enumerate(STATE_NAMES):
        values = weights[labels == class_id]
        gate_stats[name] = {
            "count": len(values),
            "mean_pose_weight": float(values.mean()),
            "std_pose_weight": float(values.std()),
            "minimum_pose_weight": float(values.min()),
            "maximum_pose_weight": float(values.max()),
        }

    summary = {
        "protocol": {
            "source_training_dataset": "CAUCAFall",
            "external_development_dataset": "URFD Camera 0 RGB",
            "model": "clean-supervised reliability-gated late-logit fusion V3",
            "fine_tuning": False,
            "target_data_used_for_model_or_policy_selection": False,
            "final_blind_external_test": False,
            "source_locked_state_machine": checkpoint["state_machine"],
            "external_resampled_fps": 20.0,
            "window_length": config["window_length"],
            "pose_features": config["pose_dim"],
            "rgb_features": config["rgb_dim"],
            "event_boundaries": "manual_event_annotations.csv",
        },
        "environment": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "three_state_frame_level": fused_metrics,
        "pose_branch_frame_level": pose_branch_metrics,
        "rgb_branch_frame_level": rgb_branch_metrics,
        "collapsed_fall_vs_adl_frame_level": binary_metrics,
        "event_level_locked_state_machine": event_metrics,
        "gate_statistics": gate_stats,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(json_ready(summary), file, ensure_ascii=False, indent=2)

    print("\nV3 external evaluation completed.")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
