#!/usr/bin/env python3
"""Run the single, label-isolated frozen Le2i inference pass.

Inputs are the already extracted label-free Pose/RGB features plus artifacts
locked before Le2i evaluation. This script never opens Le2i annotations,
fall-event tables, or the label-bearing inventory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from train_quality_gated_logit_fusion_v2 import ReliabilityGatedLogitFusion


ROOT = Path("/home/data/yoloA27")
FEATURE_DIR = ROOT / "features/le2i_frozen_features_v1"
FEATURE_MANIFEST = FEATURE_DIR / "sequence_feature_manifest.csv"
EXTRACTION_SUMMARY = FEATURE_DIR / "extraction_summary.json"
MODEL_PATH = (
    ROOT
    / "experiments/quality_gated_logit_fusion_v3_w16_fixed_seed42/"
    / "quality_gated_logit_fusion_v3_best.pt"
)
VERIFIER_PATH = (
    ROOT
    / "experiments/urfd_alarm_verifier_logistic_v1/verifier_model.json"
)
LOCKED_POLICY_PATH = (
    ROOT / "experiments/final_blind_protocol_lock_v1/locked_policy.txt"
)
OUTPUT_DIR = ROOT / "experiments/le2i_frozen_inference_v1"
PARTIAL_OUTPUT_DIR = ROOT / "experiments/le2i_frozen_inference_v1.partial"

EXPECTED_MODEL_SHA256 = (
    "9af9a610d893fa0b1a53e6cba93706d35df61b4d909297ca174c08b3a9c7b152"
)
EXPECTED_VERIFIER_SHA256 = (
    "631acee5313e955223f822557afe4e971d4023827c6bc2b828db244cbbef224e"
)
EXPECTED_POLICY_SHA256 = (
    "ee8f7caaf621549220f37c7ef3bf2b2c0a4d3534d02c740f83580581d498284c"
)

EXPECTED_SEQUENCES = 190
EXPECTED_FRAMES = 75911
EXPECTED_EMBEDDING_DIMENSION = 256
TARGET_FPS = 20.0
SAMPLE_INTERVAL_MS = 1000.0 / TARGET_FPS
BATCH_SIZE = 128
EPSILON = 1e-6

JOINT_NAMES = [
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
]

SIGNAL_NAMES = [
    "fused_fall_score",
    "prob_falling",
    "prob_fallen",
    "prob_adl",
    "pose_fall_score",
    "rgb_fall_score",
    "pose_gate_weight",
    "branch_disagreement",
    "pose_found_ratio",
    "mean_keypoint_conf",
    "rgb_roi_crop_ratio",
    "person_conf",
    "visible_keypoint_ratio",
    "torso_keypoint_conf",
    "bbox_cy",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_aspect_ratio",
    "bbox_valid",
]

STAT_NAMES = ["last", "mean", "std", "min", "max", "delta", "slope"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict, name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        result = float(value)
        return result if np.isfinite(result) else float(default)
    except (TypeError, ValueError):
        return float(default)


def integer(row: dict, name: str, default: int = 0) -> int:
    return int(round(number(row, name, default)))


def parse_locked_policy(path: Path) -> dict[str, float | int | str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    required = {
        "window_length",
        "evidence_alpha",
        "evidence_threshold",
        "evidence_consecutive_frames",
        "reset_threshold",
        "reset_consecutive_frames",
        "verifier_history_frames",
        "verifier_threshold",
        "future_frames_used",
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"Locked policy fields are missing: {missing}")
    return {
        "window_length": int(values["window_length"]),
        "alpha": float(values["evidence_alpha"]),
        "alarm_threshold": float(values["evidence_threshold"]),
        "consecutive_evidence_frames": int(
            values["evidence_consecutive_frames"]
        ),
        "reset_threshold": float(values["reset_threshold"]),
        "reset_frames": int(values["reset_consecutive_frames"]),
        "verifier_history_frames": int(values["verifier_history_frames"]),
        "verifier_threshold": float(values["verifier_threshold"]),
        "future_frames_used": int(values["future_frames_used"]),
        "verifier_short_history": values.get(
            "verifier_short_history", "left_pad_earliest"
        ),
    }


def verify_inputs() -> tuple[dict, dict, list[dict[str, str]]]:
    for path in (
        FEATURE_MANIFEST,
        EXTRACTION_SUMMARY,
        MODEL_PATH,
        VERIFIER_PATH,
        LOCKED_POLICY_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    checks = {
        MODEL_PATH: EXPECTED_MODEL_SHA256,
        VERIFIER_PATH: EXPECTED_VERIFIER_SHA256,
        LOCKED_POLICY_PATH: EXPECTED_POLICY_SHA256,
    }
    for path, expected in checks.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen artifact hash changed: {path}\n"
                f"expected={expected}\nactual={actual}"
            )

    extraction = json.loads(EXTRACTION_SUMMARY.read_text(encoding="utf-8"))
    if extraction["sequences"] != EXPECTED_SEQUENCES:
        raise RuntimeError("Feature sequence count changed")
    if extraction["frames"] != EXPECTED_FRAMES:
        raise RuntimeError("Feature frame count changed")
    protocol = extraction["protocol"]
    if protocol["labels_or_annotations_read"] is not False:
        raise RuntimeError("Feature extraction was not label-isolated")
    if protocol["prediction_or_alarm_decision_performed"] is not False:
        raise RuntimeError("Feature extraction unexpectedly made decisions")

    manifest = read_csv(FEATURE_MANIFEST)
    if len(manifest) != EXPECTED_SEQUENCES:
        raise RuntimeError("Feature manifest sequence count changed")
    if sum(integer(row, "frames") for row in manifest) != EXPECTED_FRAMES:
        raise RuntimeError("Feature manifest frame count changed")
    forbidden = {
        "sequence_type",
        "category",
        "event_label",
        "gt_fall",
        "fall_onset_frame",
        "fall_end_frame",
        "annotation_path",
    }
    if forbidden.intersection(manifest[0]):
        raise RuntimeError("Feature manifest contains forbidden label fields")
    return extraction, parse_locked_policy(LOCKED_POLICY_PATH), manifest


def validate_and_attach(
    pose_rows: list[dict[str, str]],
    rgb_rows: list[dict[str, str]],
    embeddings: np.ndarray,
) -> list[dict]:
    if len(pose_rows) != len(rgb_rows) or len(rgb_rows) != len(embeddings):
        raise ValueError(
            f"Pose/RGB alignment mismatch: "
            f"{len(pose_rows)}/{len(rgb_rows)}/{len(embeddings)}"
        )
    attached = []
    for index, (pose, rgb) in enumerate(zip(pose_rows, rgb_rows)):
        pose_key = (pose["sequence_id"], integer(pose, "frame_number"))
        rgb_key = (rgb["sequence_id"], integer(rgb, "frame_number"))
        if pose_key != rgb_key or pose["image_name"] != rgb["image_name"]:
            raise ValueError(f"Pose/RGB key mismatch: {pose_key}/{rgb_key}")
        item = dict(pose)
        item["rgb_index"] = index
        item["roi_crop_used"] = integer(rgb, "roi_crop_used")
        attached.append(item)
    return attached


def nearest_resample(rows: list[dict]) -> list[tuple[int, float, dict]]:
    rows = sorted(rows, key=lambda row: number(row, "timestamp_ms"))
    timestamps = np.asarray([number(row, "timestamp_ms") for row in rows])
    sample_times = np.arange(0.0, float(timestamps[-1]) + 0.1, SAMPLE_INTERVAL_MS)
    indices = []
    for target in sample_times:
        right = int(np.searchsorted(timestamps, target, side="left"))
        candidates = []
        if right < len(timestamps):
            candidates.append(right)
        if right > 0:
            candidates.append(right - 1)
        chosen = min(
            candidates,
            key=lambda value: (abs(timestamps[value] - target), value),
        )
        indices.append(chosen)
    return [
        (sample_index + 1, float(target), rows[index])
        for sample_index, (target, index) in enumerate(zip(sample_times, indices))
    ]


def transform_pose_sequence(
    resampled: list[tuple[int, float, dict]], expected_names: list[str]
) -> np.ndarray:
    transformed = []
    previous = None
    for _, _, row in resampled:
        bbox_w = number(row, "bbox_w")
        bbox_h = number(row, "bbox_h")
        bbox_cx = number(row, "bbox_cx")
        bbox_cy = number(row, "bbox_cy")
        values = {
            "pose_found": number(row, "pose_found"),
            "person_conf": number(row, "person_conf"),
            "visible_keypoint_ratio": number(row, "visible_keypoint_count") / 17.0,
            "mean_keypoint_conf": number(row, "mean_keypoint_conf"),
            "torso_keypoint_conf": number(row, "torso_keypoint_conf"),
            "bbox_aspect_ratio": (
                bbox_w / bbox_h if abs(bbox_h) > EPSILON else 0.0
            ),
        }
        for joint in JOINT_NAMES:
            x_name = f"{joint}_x_bbox"
            y_name = f"{joint}_y_bbox"
            confidence_name = f"{joint}_conf"
            x_value = number(row, x_name)
            y_value = number(row, y_name)
            values[x_name] = x_value
            values[y_name] = y_value
            values[confidence_name] = number(row, confidence_name)
            values[f"delta_{joint}_x_bbox"] = (
                0.0 if previous is None else x_value - number(previous, x_name)
            )
            values[f"delta_{joint}_y_bbox"] = (
                0.0 if previous is None else y_value - number(previous, y_name)
            )
        if previous is None:
            values.update(
                {
                    "delta_bbox_cx_over_w": 0.0,
                    "delta_bbox_cy_over_h": 0.0,
                    "delta_bbox_w_over_w": 0.0,
                    "delta_bbox_h_over_h": 0.0,
                }
            )
        else:
            values["delta_bbox_cx_over_w"] = np.clip(
                (bbox_cx - number(previous, "bbox_cx"))
                / max(abs(bbox_w), EPSILON),
                -2.0,
                2.0,
            )
            values["delta_bbox_cy_over_h"] = np.clip(
                (bbox_cy - number(previous, "bbox_cy"))
                / max(abs(bbox_h), EPSILON),
                -2.0,
                2.0,
            )
            values["delta_bbox_w_over_w"] = np.clip(
                (bbox_w - number(previous, "bbox_w"))
                / max(abs(bbox_w), EPSILON),
                -2.0,
                2.0,
            )
            values["delta_bbox_h_over_h"] = np.clip(
                (bbox_h - number(previous, "bbox_h"))
                / max(abs(bbox_h), EPSILON),
                -2.0,
                2.0,
            )
        missing = [name for name in expected_names if name not in values]
        if missing:
            raise ValueError(f"Cannot construct Pose features: {missing[:10]}")
        transformed.append(
            np.asarray([values[name] for name in expected_names], dtype=np.float32)
        )
        previous = row
    return np.stack(transformed)


def make_windows(
    aligned: list[dict],
    embeddings: np.ndarray,
    checkpoint: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    resampled = nearest_resample(aligned)
    pose_names = list(checkpoint["pose_feature_names"])
    quality_names = list(checkpoint["quality_feature_names"])
    quality_indices = [pose_names.index(name) for name in quality_names]
    window_length = int(checkpoint["model_config"]["window_length"])
    pose_features = transform_pose_sequence(resampled, pose_names)
    rgb_features = np.stack(
        [embeddings[int(row["rgb_index"])] for _, _, row in resampled]
    ).astype(np.float32)
    quality_features = np.clip(pose_features[:, quality_indices], 0.0, 1.0)

    pose_windows = []
    rgb_windows = []
    quality_windows = []
    metadata = []
    for end in range(window_length - 1, len(resampled)):
        start = end - window_length + 1
        sample_index, sample_timestamp, row = resampled[end]
        pose_windows.append(pose_features[start : end + 1])
        rgb_windows.append(rgb_features[start : end + 1])
        quality_windows.append(quality_features[start : end + 1])
        metadata.append(
            {
                "sequence_id": row["sequence_id"],
                "scene": row["scene"],
                "sample_index": sample_index,
                "sample_timestamp_ms": int(round(sample_timestamp)),
                "source_frame_number": integer(row, "frame_number"),
                "source_timestamp_ms": int(round(number(row, "timestamp_ms"))),
                "pose_found_ratio": float(
                    quality_features[start : end + 1, 0].mean()
                ),
                "mean_keypoint_conf": float(
                    quality_features[start : end + 1, 3].mean()
                ),
                "rgb_roi_crop_ratio": float(
                    np.mean(
                        [
                            value[2]["roi_crop_used"]
                            for value in resampled[start : end + 1]
                        ]
                    )
                ),
            }
        )
    if not pose_windows:
        raise RuntimeError(f"No causal windows for {aligned[0]['sequence_id']}")
    summary = {
        "sequence_id": aligned[0]["sequence_id"],
        "scene": aligned[0]["scene"],
        "original_frames": len(aligned),
        "resampled_frames": len(resampled),
        "windows": len(pose_windows),
    }
    return (
        np.stack(pose_windows).astype(np.float32),
        np.stack(rgb_windows).astype(np.float32),
        np.stack(quality_windows).astype(np.float32),
        metadata,
        summary,
    )


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    pose: np.ndarray,
    rgb: np.ndarray,
    quality: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    output_parts = defaultdict(list)
    model.eval()
    for start in range(0, len(pose), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(pose))
        output = model(
            torch.from_numpy(pose[start:end]).to(device),
            torch.from_numpy(rgb[start:end]).to(device),
            torch.from_numpy(quality[start:end]).to(device),
        )
        output_parts["fused"].append(
            torch.softmax(output["fused"], 1).cpu().numpy()
        )
        output_parts["pose"].append(output["pose_probabilities"].cpu().numpy())
        output_parts["rgb"].append(output["rgb_probabilities"].cpu().numpy())
        output_parts["pose_weight"].append(
            output["pose_weight"].cpu().numpy().reshape(-1)
        )
        output_parts["disagreement"].append(
            output["branch_disagreement"].cpu().numpy().reshape(-1)
        )
    return {name: np.concatenate(parts) for name, parts in output_parts.items()}


def prediction_rows(metadata: list[dict], outputs: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    fused = outputs["fused"]
    predictions = fused.argmax(1)
    states = ["ADL", "Falling", "Fallen"]
    for index, source in enumerate(metadata):
        row = dict(source)
        row.update(
            {
                "prob_adl": float(fused[index, 0]),
                "prob_falling": float(fused[index, 1]),
                "prob_fallen": float(fused[index, 2]),
                "pose_prob_adl": float(outputs["pose"][index, 0]),
                "pose_prob_falling": float(outputs["pose"][index, 1]),
                "pose_prob_fallen": float(outputs["pose"][index, 2]),
                "rgb_prob_adl": float(outputs["rgb"][index, 0]),
                "rgb_prob_falling": float(outputs["rgb"][index, 1]),
                "rgb_prob_fallen": float(outputs["rgb"][index, 2]),
                "predicted_state_id": int(predictions[index]),
                "predicted_state": states[int(predictions[index])],
                "pose_gate_weight": float(outputs["pose_weight"][index]),
                "branch_disagreement": float(outputs["disagreement"][index]),
            }
        )
        rows.append(row)
    return rows


def generate_candidates(rows: list[dict], policy: dict) -> list[dict]:
    evidence = 0.0
    above_count = 0
    below_count = 0
    latched = False
    alarms = []
    for row in rows:
        fall_score = float(row["prob_falling"]) + float(row["prob_fallen"])
        fall_score = max(0.0, min(1.0, fall_score))
        evidence = policy["alpha"] * evidence + (1.0 - policy["alpha"]) * fall_score
        if not latched:
            above_count = above_count + 1 if evidence >= policy["alarm_threshold"] else 0
            if above_count >= policy["consecutive_evidence_frames"]:
                alarms.append(
                    {
                        "sequence_id": row["sequence_id"],
                        "scene": row["scene"],
                        "sample_index": row["sample_index"],
                        "sample_timestamp_ms": row["sample_timestamp_ms"],
                        "source_frame_number": row["source_frame_number"],
                        "source_timestamp_ms": row["source_timestamp_ms"],
                        "fall_score": fall_score,
                        "accumulated_evidence": evidence,
                        "pose_gate_weight": row["pose_gate_weight"],
                        "branch_disagreement": row["branch_disagreement"],
                    }
                )
                latched = True
                above_count = 0
                below_count = 0
        else:
            below_count = below_count + 1 if evidence <= policy["reset_threshold"] else 0
            if below_count >= policy["reset_frames"]:
                latched = False
                below_count = 0
    return alarms


def calculate_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) > 1:
        x = np.arange(len(array), dtype=np.float64)
        slope = float(np.polyfit(x, array, 1)[0])
        delta = float(array[-1] - array[0])
    else:
        slope = 0.0
        delta = 0.0
    return {
        "last": float(array[-1]),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "delta": delta,
        "slope": slope,
    }


def extract_signals(prediction: dict, pose_lookup: dict) -> dict[str, float]:
    key = (prediction["sequence_id"], integer(prediction, "source_frame_number"))
    pose = pose_lookup.get(key, {})
    bbox_width = number(pose, "bbox_w")
    bbox_height = number(pose, "bbox_h")
    bbox_valid = float(
        number(pose, "pose_found") > 0.5
        and bbox_width > 0.0
        and bbox_height > 0.0
    )
    return {
        "fused_fall_score": number(prediction, "prob_falling")
        + number(prediction, "prob_fallen"),
        "prob_falling": number(prediction, "prob_falling"),
        "prob_fallen": number(prediction, "prob_fallen"),
        "prob_adl": number(prediction, "prob_adl"),
        "pose_fall_score": number(prediction, "pose_prob_falling")
        + number(prediction, "pose_prob_fallen"),
        "rgb_fall_score": number(prediction, "rgb_prob_falling")
        + number(prediction, "rgb_prob_fallen"),
        "pose_gate_weight": number(prediction, "pose_gate_weight"),
        "branch_disagreement": number(prediction, "branch_disagreement"),
        "pose_found_ratio": number(prediction, "pose_found_ratio"),
        "mean_keypoint_conf": number(prediction, "mean_keypoint_conf"),
        "rgb_roi_crop_ratio": number(prediction, "rgb_roi_crop_ratio"),
        "person_conf": number(pose, "person_conf"),
        "visible_keypoint_ratio": number(pose, "visible_keypoint_count") / 17.0,
        "torso_keypoint_conf": number(pose, "torso_keypoint_conf"),
        "bbox_cy": number(pose, "bbox_cy"),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": number(pose, "bbox_area"),
        "bbox_aspect_ratio": (
            bbox_width / bbox_height if bbox_height > 1e-8 else 0.0
        ),
        "bbox_valid": bbox_valid,
    }


def verifier_features(
    alarm: dict,
    rows: list[dict],
    pose_lookup: dict,
    history_length: int,
) -> dict:
    positions = {integer(row, "sample_index"): index for index, row in enumerate(rows)}
    sample_index = integer(alarm, "sample_index")
    if sample_index not in positions:
        raise KeyError(f"Alarm sample index is absent: {sample_index}")
    end = positions[sample_index]
    start = max(0, end - history_length + 1)
    observed = rows[start : end + 1]
    history = [observed[0]] * (history_length - len(observed)) + observed
    signals = [extract_signals(row, pose_lookup) for row in history]
    output = {
        **alarm,
        "observed_history_frames": len(observed),
        "left_padding_frames": history_length - len(observed),
        "history_frames_used": len(history),
    }
    for signal_name in SIGNAL_NAMES:
        stats = calculate_stats([row[signal_name] for row in signals])
        for stat_name in STAT_NAMES:
            output[f"{signal_name}_{stat_name}"] = stats[stat_name]
    return output


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def apply_verifier(features: dict, verifier: dict, policy: dict) -> dict:
    names = list(verifier["feature_names"])
    values = np.asarray([number(features, name) for name in names], dtype=np.float64)
    mean = np.asarray(verifier["feature_mean"], dtype=np.float64)
    scale = np.asarray(verifier["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(verifier["coefficients"], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(scale <= 0):
        raise ValueError("Invalid verifier feature or scale")
    logit = float(verifier["intercept"]) + float(
        ((values - mean) / scale) @ coefficients
    )
    probability = sigmoid(logit)
    model_threshold = float(verifier["decision_threshold"])
    if abs(model_threshold - policy["verifier_threshold"]) > 1e-12:
        raise RuntimeError("Verifier threshold differs from frozen policy")
    result = dict(features)
    result["verifier_logit"] = logit
    result["verifier_probability"] = probability
    result["verifier_threshold"] = model_threshold
    result["verifier_passed"] = int(probability >= model_threshold)
    return result


def main() -> None:
    extraction, policy, manifest = verify_inputs()
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Frozen inference output already exists; do not overwrite: {OUTPUT_DIR}"
        )
    if PARTIAL_OUTPUT_DIR.exists():
        shutil.rmtree(PARTIAL_OUTPUT_DIR)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    config = checkpoint["model_config"]
    if int(config["window_length"]) != policy["window_length"]:
        raise RuntimeError("Checkpoint and frozen-policy window lengths differ")
    if policy["window_length"] != 16 or policy["future_frames_used"] != 0:
        raise RuntimeError("Unexpected frozen W16 causal protocol")
    verifier = json.loads(VERIFIER_PATH.read_text(encoding="utf-8"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ReliabilityGatedLogitFusion(
        config["pose_dim"],
        config["rgb_dim"],
        config["quality_dim"],
        config["hidden_dim"],
        config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    PARTIAL_OUTPUT_DIR.mkdir(parents=True)
    prediction_output = PARTIAL_OUTPUT_DIR / "frame_predictions.csv"
    candidate_output = PARTIAL_OUTPUT_DIR / "candidate_alarms.csv"
    final_output = PARTIAL_OUTPUT_DIR / "final_alarms.csv"
    resampling_output = PARTIAL_OUTPUT_DIR / "resampling_summary.csv"

    prediction_handle = prediction_output.open("w", encoding="utf-8", newline="")
    candidate_handle = candidate_output.open("w", encoding="utf-8", newline="")
    prediction_writer = None
    candidate_writer = None
    final_rows = []
    resampling_rows = []
    total_windows = 0
    total_candidates = 0
    candidate_id = 0

    print(f"Python: {platform.python_version()}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}", flush=True)
    print("Le2i labels/annotations available to inference: NO", flush=True)
    print("Frozen model/policy/verifier hashes: VERIFIED", flush=True)

    try:
        for sequence_number, item in enumerate(manifest, start=1):
            sequence_id = item["sequence_id"]
            print(
                f"Sequence {sequence_number}/{len(manifest)}: {sequence_id}",
                flush=True,
            )
            pose_path = Path(item["pose_features_csv"])
            metadata_path = Path(item["rgb_metadata_csv"])
            embedding_path = Path(item["rgb_embeddings_npy"])
            for path in (pose_path, metadata_path, embedding_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            pose_rows = read_csv(pose_path)
            rgb_rows = read_csv(metadata_path)
            embeddings = np.load(embedding_path, mmap_mode="r")
            expected_frames = integer(item, "frames")
            if embeddings.shape != (
                expected_frames,
                EXPECTED_EMBEDDING_DIMENSION,
            ):
                raise ValueError(
                    f"Unexpected embedding shape for {sequence_id}: {embeddings.shape}"
                )
            aligned = validate_and_attach(pose_rows, rgb_rows, embeddings)
            pose, rgb, quality, metadata, resampling = make_windows(
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
            outputs = predict(model, pose, rgb, quality, device)
            rows = prediction_rows(metadata, outputs)
            if prediction_writer is None:
                prediction_writer = csv.DictWriter(
                    prediction_handle, fieldnames=list(rows[0])
                )
                prediction_writer.writeheader()
            prediction_writer.writerows(rows)
            prediction_handle.flush()

            candidates = generate_candidates(rows, policy)
            pose_lookup = {
                (row["sequence_id"], integer(row, "frame_number")): row
                for row in pose_rows
            }
            for alarm in candidates:
                candidate_id += 1
                alarm["candidate_id"] = candidate_id
                features = verifier_features(
                    alarm,
                    rows,
                    pose_lookup,
                    policy["verifier_history_frames"],
                )
                verified = apply_verifier(features, verifier, policy)
                if candidate_writer is None:
                    candidate_writer = csv.DictWriter(
                        candidate_handle, fieldnames=list(verified)
                    )
                    candidate_writer.writeheader()
                candidate_writer.writerow(verified)
                if verified["verifier_passed"] == 1:
                    final_rows.append(
                        {
                            "candidate_id": verified["candidate_id"],
                            "sequence_id": verified["sequence_id"],
                            "scene": verified["scene"],
                            "sample_index": verified["sample_index"],
                            "sample_timestamp_ms": verified["sample_timestamp_ms"],
                            "source_frame_number": verified["source_frame_number"],
                            "source_timestamp_ms": verified["source_timestamp_ms"],
                            "fall_score": verified["fall_score"],
                            "accumulated_evidence": verified[
                                "accumulated_evidence"
                            ],
                            "verifier_probability": verified[
                                "verifier_probability"
                            ],
                            "verifier_threshold": verified["verifier_threshold"],
                        }
                    )
            candidate_handle.flush()
            total_windows += len(rows)
            total_candidates += len(candidates)
            resampling_rows.append(resampling)
            print(
                f"  frames={expected_frames}, windows={len(rows)}, "
                f"candidates={len(candidates)}, "
                f"retained={sum(row['sequence_id'] == sequence_id for row in final_rows)}",
                flush=True,
            )
            del embeddings, pose, rgb, quality, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        prediction_handle.close()
        candidate_handle.close()

    if prediction_writer is None:
        raise RuntimeError("No frame predictions were written")
    if candidate_writer is None:
        candidate_output.write_text("candidate_id\n", encoding="utf-8")
    final_fields = [
        "candidate_id",
        "sequence_id",
        "scene",
        "sample_index",
        "sample_timestamp_ms",
        "source_frame_number",
        "source_timestamp_ms",
        "fall_score",
        "accumulated_evidence",
        "verifier_probability",
        "verifier_threshold",
    ]
    write_csv(final_output, final_rows, final_fields)
    write_csv(
        resampling_output,
        resampling_rows,
        ["sequence_id", "scene", "original_frames", "resampled_frames", "windows"],
    )

    summary = {
        "protocol": {
            "dataset": "Le2i",
            "role": "single frozen final blind inference before label reveal",
            "source_training_dataset": "CAUCAFall",
            "external_development_dataset": "URFD Camera 0 RGB",
            "labels_or_annotations_read": False,
            "label_bearing_inventory_read": False,
            "model_or_threshold_tuning_performed": False,
            "future_frames_used": 0,
            "target_fps": TARGET_FPS,
            "window_length": policy["window_length"],
            "policy": policy,
            "verifier_feature_names": verifier["feature_names"],
        },
        "locked_artifacts": {
            "fusion_model": str(MODEL_PATH),
            "fusion_model_sha256": sha256(MODEL_PATH),
            "verifier_model": str(VERIFIER_PATH),
            "verifier_model_sha256": sha256(VERIFIER_PATH),
            "policy": str(LOCKED_POLICY_PATH),
            "policy_sha256": sha256(LOCKED_POLICY_PATH),
        },
        "input_feature_extraction_summary_sha256": sha256(EXTRACTION_SUMMARY),
        "sequences": len(manifest),
        "raw_frames": sum(integer(row, "frames") for row in manifest),
        "causal_windows": total_windows,
        "evidence_candidate_alarms": total_candidates,
        "verifier_retained_alarms": len(final_rows),
        "output_files": {
            "frame_predictions": str(OUTPUT_DIR / prediction_output.name),
            "candidate_alarms": str(OUTPUT_DIR / candidate_output.name),
            "final_alarms": str(OUTPUT_DIR / final_output.name),
            "resampling_summary": str(OUTPUT_DIR / resampling_output.name),
        },
    }
    (PARTIAL_OUTPUT_DIR / "inference_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    os.replace(PARTIAL_OUTPUT_DIR, OUTPUT_DIR)
    print("\nFrozen label-isolated inference completed.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
