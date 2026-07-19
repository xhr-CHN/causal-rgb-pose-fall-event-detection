#!/usr/bin/env python3
"""Run the single label-isolated frozen inference pass on anonymous GMDCSA24.

The script consumes only features extracted from GMDCSA24_FROZEN_INPUT_V1.
It applies the locked W16 RGB/Pose fusion model, source-selected evidence
candidate rule, development-only compact verifier, and fixed causal rescue
rule. It never opens the raw GMDCSA24 tree, official CSV files, or the private
evaluation mapping.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from train_quality_gated_logit_fusion_v2 import ReliabilityGatedLogitFusion


ROOT = Path("/home/data/yoloA27")
FEATURE_DIR = ROOT / "features/gmdcsa24_frozen_features_v1"
FEATURE_MANIFEST = FEATURE_DIR / "sequence_feature_manifest.csv"
EXTRACTION_SUMMARY = FEATURE_DIR / "extraction_summary.json"
FUSION_MODEL = (
    ROOT
    / "experiments/quality_gated_logit_fusion_v3_w16_fixed_seed42/"
    "quality_gated_logit_fusion_v3_best.pt"
)
VERIFIER_MODEL = (
    ROOT
    / "experiments/multidataset_alarm_verifier_v2_seed42/"
    "alarm_verifier_v2_model.json"
)
SELECTED_POLICY = (
    ROOT / "experiments/causal_rescue_alarm_policy_v2/selected_policy.json"
)
LOCK_FILE = (
    ROOT
    / "experiments/gmdcsa24_final_blind_protocol_lock_v1/"
    "locked_artifacts_sha256.txt"
)
OUTPUT_DIR = ROOT / "experiments/gmdcsa24_frozen_inference_v1"
PARTIAL_OUTPUT_DIR = ROOT / "experiments/gmdcsa24_frozen_inference_v1.partial"

EXPECTED_FUSION_SHA256 = (
    "9af9a610d893fa0b1a53e6cba93706d35df61b4d909297ca174c08b3a9c7b152"
)
EXPECTED_VERIFIER_SHA256 = (
    "f63661d59b5731355c81eec8c3c2f3f79fa9c1f1f7f214a0a6c833398d6ae11c"
)
EXPECTED_POLICY_SHA256 = (
    "0865b34814e0481ec0ba7b7894cf6493b9b4681279910e24b1f61239e7fd4de5"
)
EXPECTED_SEQUENCES = 160
EXPECTED_EMBEDDING_DIMENSION = 256
EXPECTED_WINDOW_LENGTH = 16
EXPECTED_TARGET_FPS = 20.0
EXPECTED_PRIMARY_THRESHOLD = 0.3179642728137935
EXPECTED_RESCUE_POSE_SLOPE = 0.05
EXPECTED_RESCUE_FALLING_DELTA = 0.0
EXPECTED_CONFIRMATION_FUTURE_SAMPLES = 4
EXPECTED_MAXIMUM_ADDED_DELAY_MS = 200.0

SAMPLE_INTERVAL_MS = 1000.0 / EXPECTED_TARGET_FPS
BATCH_SIZE = 128
PRE_HISTORY_FRAMES = 20
EPSILON = 1e-6
OPAQUE_ID_PATTERN = re.compile(r"^gmdcsa24_[0-9a-f]{16}$")

MANIFEST_FIELDS = [
    "sequence_id",
    "frames",
    "fps",
    "pose_features_csv",
    "rgb_metadata_csv",
    "rgb_embeddings_npy",
    "sequence_summary_json",
]

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

PRE_SIGNAL_NAMES = [
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
CONFIRM_SIGNAL_NAMES = [
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
]
STAT_NAMES = ["last", "mean", "std", "min", "max", "delta", "slope"]


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict, name: str, default: float = 0.0) -> float:
    try:
        value = row.get(name, "")
        result = float(default if value in ("", None) else value)
        return result if math.isfinite(result) else float(default)
    except (TypeError, ValueError):
        return float(default)


def integer(row: dict, name: str, default: int = 0) -> int:
    return int(round(number(row, name, default)))


def divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def read_lock() -> dict:
    locked = {}
    for line_number, line in enumerate(
        LOCK_FILE.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        if len(line) < 67 or line[64:66] != "  ":
            fail(f"Malformed lock line {line_number}")
        digest = line[:64].lower()
        path = str(Path(line[66:]).resolve())
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            fail(f"Invalid SHA-256 in lock line {line_number}")
        locked[path] = digest
    return locked


def verify_locked_path(path: Path, locked: dict, expected: str = "") -> str:
    resolved = str(path.resolve())
    if resolved not in locked:
        fail(f"Required artifact is absent from final lock: {resolved}")
    actual = sha256(path)
    if actual != locked[resolved]:
        fail(f"Locked artifact hash changed: {resolved}")
    if expected and actual != expected:
        fail(
            f"Artifact does not match the predeclared SHA-256: {resolved}; "
            f"expected={expected}, actual={actual}"
        )
    return actual


def inside_directory(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path.resolve()), str(root.resolve())]) == str(
            root.resolve()
        )
    except ValueError:
        return False


def verify_inputs():
    required = [
        FEATURE_MANIFEST,
        EXTRACTION_SUMMARY,
        FUSION_MODEL,
        VERIFIER_MODEL,
        SELECTED_POLICY,
        LOCK_FILE,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    locked = read_lock()
    verified = {
        "fusion_model": verify_locked_path(
            FUSION_MODEL, locked, EXPECTED_FUSION_SHA256
        ),
        "verifier_model": verify_locked_path(
            VERIFIER_MODEL, locked, EXPECTED_VERIFIER_SHA256
        ),
        "selected_policy": verify_locked_path(
            SELECTED_POLICY, locked, EXPECTED_POLICY_SHA256
        ),
        "inference_script": verify_locked_path(Path(__file__).resolve(), locked),
    }

    extraction = json.loads(EXTRACTION_SUMMARY.read_text(encoding="utf-8"))
    protocol = extraction.get("protocol", {})
    if protocol.get("labels_annotations_private_mapping_or_raw_dataset_read") is not False:
        fail("Feature extraction was not label-isolated")
    if protocol.get("fall_prediction_or_alarm_decision_performed") is not False:
        fail("Feature extraction unexpectedly made fall/alarm decisions")
    if int(extraction.get("sequences", -1)) != EXPECTED_SEQUENCES:
        fail("Unexpected feature sequence count")

    manifest, fields = read_csv(FEATURE_MANIFEST)
    if fields != MANIFEST_FIELDS:
        fail(f"Feature-manifest fields changed: {fields}")
    if len(manifest) != EXPECTED_SEQUENCES:
        fail(f"Expected {EXPECTED_SEQUENCES} feature sequences, found {len(manifest)}")
    sequence_ids = [row["sequence_id"] for row in manifest]
    if len(set(sequence_ids)) != len(sequence_ids):
        fail("Duplicate sequence_id in feature manifest")
    if any(not OPAQUE_ID_PATTERN.fullmatch(value) for value in sequence_ids):
        fail("Feature manifest contains a non-opaque sequence_id")

    for row in manifest:
        sequence_root = FEATURE_DIR / "sequences" / row["sequence_id"]
        paths = [
            Path(row["pose_features_csv"]),
            Path(row["rgb_metadata_csv"]),
            Path(row["rgb_embeddings_npy"]),
            Path(row["sequence_summary_json"]),
        ]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            if not inside_directory(path, sequence_root):
                fail(f"Feature path escapes anonymous sequence directory: {path}")

    verifier = json.loads(VERIFIER_MODEL.read_text(encoding="utf-8"))
    policy = json.loads(SELECTED_POLICY.read_text(encoding="utf-8"))
    validate_policy_and_verifier(policy, verifier)
    return extraction, manifest, verifier, policy, verified


def validate_policy_and_verifier(policy: dict, verifier: dict) -> None:
    if policy.get("policy_name") != "causal_rescue_alarm_policy_v2":
        fail("Unexpected selected-policy name")
    base = policy.get("base_candidate_policy", {})
    expected_base = {
        "resampled_fps": EXPECTED_TARGET_FPS,
        "window_length": EXPECTED_WINDOW_LENGTH,
        "evidence_alpha": 0.5,
        "evidence_alarm_threshold": 0.85,
        "consecutive_evidence_frames": 2,
        "reset_threshold": 0.1,
        "reset_frames": 5,
    }
    for name, expected in expected_base.items():
        value = base.get(name)
        if isinstance(expected, float):
            if value is None or abs(float(value) - expected) > 1e-12:
                fail(f"Locked base policy changed: {name}={value}")
        elif int(value) != expected:
            fail(f"Locked base policy changed: {name}={value}")

    primary = policy.get("primary_verifier", {})
    if primary.get("model_sha256") != EXPECTED_VERIFIER_SHA256:
        fail("Policy references an unexpected verifier SHA-256")
    if primary.get("feature_configuration") != "pre_plus_confirmation_v2":
        fail("Policy references an unexpected verifier feature configuration")
    if abs(float(primary.get("threshold", -1)) - EXPECTED_PRIMARY_THRESHOLD) > 1e-12:
        fail("Policy primary threshold changed")
    if verifier.get("selected_feature_configuration") != "pre_plus_confirmation_v2":
        fail("Verifier feature configuration changed")
    if abs(float(verifier.get("decision_threshold", -1)) - EXPECTED_PRIMARY_THRESHOLD) > 1e-12:
        fail("Verifier threshold changed")
    if int(verifier.get("confirmation_future_samples", -1)) != (
        EXPECTED_CONFIRMATION_FUTURE_SAMPLES
    ):
        fail("Verifier confirmation horizon changed")

    conditions = policy.get("causal_rescue_path", {}).get(
        "all_conditions_required", {}
    )
    if abs(
        float(conditions.get("pre_pose_fall_score_slope_minimum", -1))
        - EXPECTED_RESCUE_POSE_SLOPE
    ) > 1e-12:
        fail("Rescue Pose-slope threshold changed")
    if abs(
        float(conditions.get("confirm_prob_falling_delta_minimum", -1))
        - EXPECTED_RESCUE_FALLING_DELTA
    ) > 1e-12:
        fail("Rescue Falling-delta threshold changed")
    timing = policy.get("timing", {})
    if int(timing.get("confirmation_future_samples", -1)) != (
        EXPECTED_CONFIRMATION_FUTURE_SAMPLES
    ):
        fail("Policy confirmation horizon changed")
    if abs(
        float(timing.get("maximum_added_delay_ms", -1))
        - EXPECTED_MAXIMUM_ADDED_DELAY_MS
    ) > 1e-12:
        fail("Policy maximum added delay changed")


def validate_and_attach(pose_rows: list, rgb_rows: list, embeddings: np.ndarray):
    if len(pose_rows) != len(rgb_rows) or len(rgb_rows) != len(embeddings):
        fail(
            "Pose/RGB alignment mismatch: "
            f"{len(pose_rows)}/{len(rgb_rows)}/{len(embeddings)}"
        )
    attached = []
    for index, (pose, rgb) in enumerate(zip(pose_rows, rgb_rows)):
        pose_key = (pose["sequence_id"], integer(pose, "frame_number"))
        rgb_key = (rgb["sequence_id"], integer(rgb, "frame_number"))
        if pose_key != rgb_key or pose["image_name"] != rgb["image_name"]:
            fail(f"Pose/RGB identity mismatch: {pose_key}/{rgb_key}")
        if not OPAQUE_ID_PATTERN.fullmatch(pose_key[0]):
            fail(f"Non-opaque feature sequence_id: {pose_key[0]}")
        item = dict(pose)
        item["rgb_index"] = index
        item["roi_crop_used"] = integer(rgb, "roi_crop_used")
        attached.append(item)
    return attached


def nearest_resample(rows: list):
    rows = sorted(rows, key=lambda row: number(row, "timestamp_ms"))
    timestamps = np.asarray([number(row, "timestamp_ms") for row in rows])
    sample_times = np.arange(
        0.0, float(timestamps[-1]) + 0.1, SAMPLE_INTERVAL_MS
    )
    selected = []
    for target in sample_times:
        right = int(np.searchsorted(timestamps, target, side="left"))
        choices = []
        if right < len(timestamps):
            choices.append(right)
        if right > 0:
            choices.append(right - 1)
        selected.append(
            min(choices, key=lambda value: (abs(timestamps[value] - target), value))
        )
    return [
        (sample_index + 1, float(target), rows[index])
        for sample_index, (target, index) in enumerate(zip(sample_times, selected))
    ]


def transform_pose_sequence(resampled: list, expected_names: list) -> np.ndarray:
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
            fail(f"Cannot construct Pose model features: {missing[:10]}")
        transformed.append(
            np.asarray([values[name] for name in expected_names], dtype=np.float32)
        )
        previous = row
    return np.stack(transformed)


def make_windows(aligned: list, embeddings: np.ndarray, checkpoint: dict):
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
        fail(f"No causal W16 windows for {aligned[0]['sequence_id']}")
    summary = {
        "sequence_id": aligned[0]["sequence_id"],
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
def predict(model, pose, rgb, quality, device):
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


def make_prediction_rows(metadata: list, outputs: dict) -> list:
    rows = []
    fused = outputs["fused"]
    predictions = fused.argmax(1)
    state_names = ["ADL", "Falling", "Fallen"]
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
                "predicted_state": state_names[int(predictions[index])],
                "pose_gate_weight": float(outputs["pose_weight"][index]),
                "branch_disagreement": float(outputs["disagreement"][index]),
            }
        )
        rows.append(row)
    return rows


def generate_candidates(rows: list, base_policy: dict) -> list:
    evidence = 0.0
    above_count = 0
    below_count = 0
    latched = False
    alarms = []
    alpha = float(base_policy["evidence_alpha"])
    alarm_threshold = float(base_policy["evidence_alarm_threshold"])
    consecutive = int(base_policy["consecutive_evidence_frames"])
    reset_threshold = float(base_policy["reset_threshold"])
    reset_frames = int(base_policy["reset_frames"])
    for row in rows:
        fall_score = max(
            0.0,
            min(1.0, number(row, "prob_falling") + number(row, "prob_fallen")),
        )
        evidence = alpha * evidence + (1.0 - alpha) * fall_score
        if not latched:
            above_count = above_count + 1 if evidence >= alarm_threshold else 0
            if above_count >= consecutive:
                alarms.append(
                    {
                        "sequence_id": row["sequence_id"],
                        "sample_index": integer(row, "sample_index"),
                        "sample_timestamp_ms": number(row, "sample_timestamp_ms"),
                        "source_frame_number": integer(row, "source_frame_number"),
                        "source_timestamp_ms": number(row, "source_timestamp_ms"),
                        "candidate_fall_score": fall_score,
                        "candidate_accumulated_evidence": evidence,
                    }
                )
                latched = True
                above_count = 0
                below_count = 0
        else:
            below_count = below_count + 1 if evidence <= reset_threshold else 0
            if below_count >= reset_frames:
                latched = False
                below_count = 0
    return alarms


def stats(values: list) -> dict:
    if not values:
        values = [0.0]
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


def longest_run(values: list, threshold: float) -> int:
    best = 0
    current = 0
    for value in values:
        current = current + 1 if value >= threshold else 0
        best = max(best, current)
    return best


def extract_signals(prediction: dict, pose_lookup: dict) -> dict:
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


def build_candidate_features(
    candidate: dict, rows: list, pose_lookup: dict, candidate_id: int
) -> dict:
    positions = {
        integer(row, "sample_index"): position for position, row in enumerate(rows)
    }
    sample_index = integer(candidate, "sample_index")
    if sample_index not in positions:
        fail(f"Candidate sample is absent: {candidate['sequence_id']}/{sample_index}")
    candidate_position = positions[sample_index]

    pre_start = max(0, candidate_position - PRE_HISTORY_FRAMES + 1)
    pre_observed = rows[pre_start : candidate_position + 1]
    pre_history = [pre_observed[0]] * (
        PRE_HISTORY_FRAMES - len(pre_observed)
    ) + pre_observed
    pre_signals = [extract_signals(row, pose_lookup) for row in pre_history]

    confirmation_length = EXPECTED_CONFIRMATION_FUTURE_SAMPLES + 1
    confirmation_observed = rows[
        candidate_position : candidate_position + confirmation_length
    ]
    confirmation = confirmation_observed + [confirmation_observed[-1]] * (
        confirmation_length - len(confirmation_observed)
    )
    confirm_signals = [extract_signals(row, pose_lookup) for row in confirmation]
    decision_row = confirmation_observed[-1]
    added_delay = max(
        0.0,
        number(decision_row, "sample_timestamp_ms")
        - number(candidate, "sample_timestamp_ms"),
    )
    if added_delay > EXPECTED_MAXIMUM_ADDED_DELAY_MS + 1.0:
        fail(f"Confirmation delay exceeds 200 ms: {added_delay}")

    output = dict(candidate)
    output.update(
        {
            "candidate_id": candidate_id,
            "candidate_key": (
                f"gmdcsa24::{candidate['sequence_id']}::{sample_index}::{candidate_id}"
            ),
            "decision_sample_index": integer(decision_row, "sample_index"),
            "decision_timestamp_ms": number(decision_row, "sample_timestamp_ms"),
            "confirmation_added_delay_ms": added_delay,
            "pre_history_observed_frames": len(pre_observed),
            "pre_history_left_padding_frames": PRE_HISTORY_FRAMES
            - len(pre_observed),
            "confirmation_observed_frames": len(confirmation_observed),
            "confirmation_right_padding_frames": confirmation_length
            - len(confirmation_observed),
        }
    )
    for signal_name in PRE_SIGNAL_NAMES:
        values = [row[signal_name] for row in pre_signals]
        for stat_name, value in stats(values).items():
            output[f"pre_{signal_name}_{stat_name}"] = value
    for signal_name in CONFIRM_SIGNAL_NAMES:
        values = [row[signal_name] for row in confirm_signals]
        for stat_name, value in stats(values).items():
            output[f"confirm_{signal_name}_{stat_name}"] = value

    fall_scores = [row["fused_fall_score"] for row in confirm_signals]
    pose_scores = [row["pose_fall_score"] for row in confirm_signals]
    rgb_scores = [row["rgb_fall_score"] for row in confirm_signals]
    for threshold in (0.50, 0.70, 0.85):
        tag = f"{int(round(threshold * 100)):02d}"
        output[f"confirm_fall_support_fraction_{tag}"] = divide(
            sum(value >= threshold for value in fall_scores), confirmation_length
        )
        output[f"confirm_fall_longest_run_{tag}"] = divide(
            longest_run(fall_scores, threshold), confirmation_length
        )
    output["confirm_modal_agreement_fraction_50"] = divide(
        sum(
            pose >= 0.50 and rgb >= 0.50
            for pose, rgb in zip(pose_scores, rgb_scores)
        ),
        confirmation_length,
    )
    output["confirm_modal_either_fraction_50"] = divide(
        sum(
            pose >= 0.50 or rgb >= 0.50
            for pose, rgb in zip(pose_scores, rgb_scores)
        ),
        confirmation_length,
    )
    output["confirm_recovery_drop_from_candidate"] = max(
        0.0, fall_scores[0] - min(fall_scores)
    )
    output["confirm_fallen_rise"] = (
        confirm_signals[-1]["prob_fallen"] - confirm_signals[0]["prob_fallen"]
    )
    output["confirm_falling_to_fallen_shift"] = (
        confirm_signals[-1]["prob_fallen"]
        - confirm_signals[-1]["prob_falling"]
        - confirm_signals[0]["prob_fallen"]
        + confirm_signals[0]["prob_falling"]
    )
    return output


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def apply_frozen_decision(features: dict, verifier: dict, policy: dict) -> dict:
    names = list(verifier["feature_names"])
    missing = [name for name in names if name not in features]
    if missing:
        fail(f"Candidate misses verifier features: {missing}")
    values = np.asarray([number(features, name) for name in names], dtype=np.float64)
    mean = np.asarray(verifier["feature_mean"], dtype=np.float64)
    scale = np.asarray(verifier["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(verifier["coefficients"], dtype=np.float64)
    if not (
        len(values) == len(mean) == len(scale) == len(coefficients)
        and np.isfinite(values).all()
        and np.isfinite(mean).all()
        and np.isfinite(scale).all()
        and np.isfinite(coefficients).all()
        and np.all(scale > 0.0)
    ):
        fail("Invalid verifier vector, coefficient, mean, or scale")
    logit = float(verifier["intercept"]) + float(
        ((values - mean) / scale) @ coefficients
    )
    probability = sigmoid(logit)
    threshold = float(verifier["decision_threshold"])
    primary_accepted = probability >= threshold
    conditions = policy["causal_rescue_path"]["all_conditions_required"]
    rescue_accepted = (
        not primary_accepted
        and number(features, "pre_pose_fall_score_slope")
        >= float(conditions["pre_pose_fall_score_slope_minimum"])
        and number(features, "confirm_prob_falling_delta")
        >= float(conditions["confirm_prob_falling_delta_minimum"])
    )
    output = dict(features)
    output.update(
        {
            "verifier_logit": logit,
            "verifier_probability": probability,
            "verifier_threshold": threshold,
            "primary_accepted": int(primary_accepted),
            "rescue_accepted": int(rescue_accepted),
            "final_accepted": int(primary_accepted or rescue_accepted),
            "acceptance_path": (
                "primary"
                if primary_accepted
                else "causal_rescue"
                if rescue_accepted
                else "rejected"
            ),
        }
    )
    return output


def final_alarm_row(row: dict) -> dict:
    names = [
        "candidate_id",
        "candidate_key",
        "sequence_id",
        "sample_index",
        "sample_timestamp_ms",
        "source_frame_number",
        "source_timestamp_ms",
        "decision_sample_index",
        "decision_timestamp_ms",
        "confirmation_added_delay_ms",
        "candidate_fall_score",
        "candidate_accumulated_evidence",
        "verifier_probability",
        "verifier_threshold",
        "primary_accepted",
        "rescue_accepted",
        "acceptance_path",
        "pre_pose_fall_score_slope",
        "confirm_prob_falling_delta",
    ]
    return {name: row[name] for name in names}


def main() -> None:
    extraction, manifest, verifier, policy, verified = verify_inputs()
    if OUTPUT_DIR.exists():
        fail(f"Frozen inference output already exists; refusing overwrite: {OUTPUT_DIR}")
    if PARTIAL_OUTPUT_DIR.exists():
        shutil.rmtree(PARTIAL_OUTPUT_DIR)

    checkpoint = torch.load(FUSION_MODEL, map_location="cpu")
    config = checkpoint["model_config"]
    if int(config["window_length"]) != EXPECTED_WINDOW_LENGTH:
        fail("Fusion checkpoint is not the locked W16 model")
    if int(config["rgb_dim"]) != EXPECTED_EMBEDDING_DIMENSION:
        fail("Fusion checkpoint RGB dimension changed")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ReliabilityGatedLogitFusion(
        config["pose_dim"],
        config["rgb_dim"],
        config["quality_dim"],
        config["hidden_dim"],
        config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    PARTIAL_OUTPUT_DIR.mkdir(parents=True)
    prediction_path = PARTIAL_OUTPUT_DIR / "frame_predictions.csv"
    candidate_path = PARTIAL_OUTPUT_DIR / "candidate_alarms.csv"
    final_path = PARTIAL_OUTPUT_DIR / "final_alarms.csv"
    resampling_path = PARTIAL_OUTPUT_DIR / "resampling_summary.csv"
    prediction_handle = prediction_path.open("w", encoding="utf-8", newline="")
    candidate_handle = candidate_path.open("w", encoding="utf-8", newline="")
    prediction_writer = None
    candidate_writer = None
    final_rows = []
    resampling_rows = []
    candidate_id = 0
    total_windows = 0
    total_candidates = 0
    total_primary = 0
    total_rescue = 0

    print(f"Python: {platform.python_version()}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("GMDCSA24 labels/private mapping/raw CSV available to inference: NO", flush=True)
    print("Frozen model, verifier, policy, and scripts: LOCK VERIFIED", flush=True)

    try:
        for sequence_number, item in enumerate(manifest, start=1):
            sequence_id = item["sequence_id"]
            print(
                f"Sequence {sequence_number}/{len(manifest)}: {sequence_id}",
                flush=True,
            )
            pose_rows, _ = read_csv(Path(item["pose_features_csv"]))
            rgb_rows, _ = read_csv(Path(item["rgb_metadata_csv"]))
            embeddings = np.load(Path(item["rgb_embeddings_npy"]), mmap_mode="r")
            expected_frames = integer(item, "frames")
            if embeddings.shape != (
                expected_frames,
                EXPECTED_EMBEDDING_DIMENSION,
            ):
                fail(f"Unexpected embedding shape for {sequence_id}: {embeddings.shape}")
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
            rows = make_prediction_rows(metadata, outputs)
            if prediction_writer is None:
                prediction_writer = csv.DictWriter(
                    prediction_handle, fieldnames=list(rows[0])
                )
                prediction_writer.writeheader()
            prediction_writer.writerows(rows)
            prediction_handle.flush()

            candidates = generate_candidates(rows, policy["base_candidate_policy"])
            pose_lookup = {
                (row["sequence_id"], integer(row, "frame_number")): row
                for row in pose_rows
            }
            sequence_primary = 0
            sequence_rescue = 0
            for candidate in candidates:
                candidate_id += 1
                features = build_candidate_features(
                    candidate, rows, pose_lookup, candidate_id
                )
                decision = apply_frozen_decision(features, verifier, policy)
                if candidate_writer is None:
                    candidate_writer = csv.DictWriter(
                        candidate_handle, fieldnames=list(decision)
                    )
                    candidate_writer.writeheader()
                candidate_writer.writerow(decision)
                if decision["primary_accepted"]:
                    sequence_primary += 1
                    total_primary += 1
                if decision["rescue_accepted"]:
                    sequence_rescue += 1
                    total_rescue += 1
                if decision["final_accepted"]:
                    final_rows.append(final_alarm_row(decision))
            candidate_handle.flush()
            total_windows += len(rows)
            total_candidates += len(candidates)
            resampling_rows.append(resampling)
            print(
                f"  frames={expected_frames}, windows={len(rows)}, "
                f"candidates={len(candidates)}, primary={sequence_primary}, "
                f"rescue={sequence_rescue}",
                flush=True,
            )
            del embeddings, pose, rgb, quality, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        prediction_handle.close()
        candidate_handle.close()

    if prediction_writer is None:
        fail("No frame predictions were written")
    if candidate_writer is None:
        write_csv(candidate_path, [], ["candidate_id"])

    final_fields = [
        "candidate_id",
        "candidate_key",
        "sequence_id",
        "sample_index",
        "sample_timestamp_ms",
        "source_frame_number",
        "source_timestamp_ms",
        "decision_sample_index",
        "decision_timestamp_ms",
        "confirmation_added_delay_ms",
        "candidate_fall_score",
        "candidate_accumulated_evidence",
        "verifier_probability",
        "verifier_threshold",
        "primary_accepted",
        "rescue_accepted",
        "acceptance_path",
        "pre_pose_fall_score_slope",
        "confirm_prob_falling_delta",
    ]
    write_csv(final_path, final_rows, final_fields)
    write_csv(
        resampling_path,
        resampling_rows,
        ["sequence_id", "original_frames", "resampled_frames", "windows"],
    )

    output_hashes = {
        path.name: sha256(path)
        for path in (prediction_path, candidate_path, final_path, resampling_path)
    }
    summary = {
        "protocol": {
            "dataset": "GMDCSA24 v2.1 anonymous input",
            "role": "single frozen final blind inference before label reveal",
            "source_training_dataset": "CAUCAFall",
            "external_development_datasets": ["URFD Camera 0 RGB", "Le2i revealed V1"],
            "labels_annotations_private_mapping_or_raw_dataset_read": False,
            "model_or_threshold_tuning_performed": False,
            "target_fps": EXPECTED_TARGET_FPS,
            "window_length": EXPECTED_WINDOW_LENGTH,
            "confirmation_future_samples": EXPECTED_CONFIRMATION_FUTURE_SAMPLES,
            "maximum_added_decision_delay_ms": EXPECTED_MAXIMUM_ADDED_DELAY_MS,
            "post_decision_samples_used": 0,
            "base_candidate_policy": policy["base_candidate_policy"],
            "primary_verifier": policy["primary_verifier"],
            "causal_rescue_path": policy["causal_rescue_path"],
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_artifacts_verified": verified,
        "input_feature_extraction_summary_sha256": sha256(EXTRACTION_SUMMARY),
        "sequences": len(manifest),
        "raw_frames": int(extraction["frames"]),
        "causal_windows": total_windows,
        "evidence_candidate_alarms": total_candidates,
        "primary_retained_alarms": total_primary,
        "causal_rescue_alarms": total_rescue,
        "final_retained_alarms": len(final_rows),
        "output_sha256": output_hashes,
        "inference_script_sha256": sha256(Path(__file__).resolve()),
    }
    (PARTIAL_OUTPUT_DIR / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(PARTIAL_OUTPUT_DIR, OUTPUT_DIR)
    print("\nFrozen label-isolated inference completed.", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
