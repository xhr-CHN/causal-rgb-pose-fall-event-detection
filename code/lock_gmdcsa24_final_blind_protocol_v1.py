#!/usr/bin/env python3
"""Cryptographically lock the final GMDCSA24 blind inference protocol.

Run this exactly once after uploading the frozen extractor and inference
scripts, and before any GMDCSA24 model inference. The script reads only the
anonymous blind inventory; it never opens the private evaluation mapping or
the raw dataset CSV files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/data/yoloA27")
BLIND_ROOT = ROOT / "GMDCSA24_FROZEN_INPUT_V1"
BLIND_INVENTORY = BLIND_ROOT / "blind_video_inventory.csv"
DATASET_PROTOCOL = (
    ROOT / "experiments/gmdcsa24_final_blind_lock_v1/locked_protocol.txt"
)
OUTPUT_DIR = ROOT / "experiments/gmdcsa24_final_blind_protocol_lock_v1"

POSE_MODEL = ROOT / "yolo26n-pose.pt"
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
POSE_HELPER = ROOT / "extract_caucafall_pose_features_v1.py"
FUSION_BASE_SOURCE = ROOT / "train_quality_gated_fusion_tcn.py"
FUSION_MODEL_SOURCE = ROOT / "train_quality_gated_logit_fusion_v2.py"
EXTRACTOR_SCRIPT = ROOT / "extract_gmdcsa24_frozen_features_v1.py"
INFERENCE_SCRIPT = ROOT / "run_gmdcsa24_frozen_inference_v1.py"

FEATURE_OUTPUT = ROOT / "features/gmdcsa24_frozen_features_v1"
INFERENCE_OUTPUT = ROOT / "experiments/gmdcsa24_frozen_inference_v1"
FEATURE_PARTIAL = ROOT / "features/gmdcsa24_frozen_features_v1.partial"
INFERENCE_PARTIAL = ROOT / "experiments/gmdcsa24_frozen_inference_v1.partial"

EXPECTED_SEQUENCES = 160
EXPECTED_INVENTORY_FIELDS = ["sequence_id", "video_file", "sha256"]
OPAQUE_ID_PATTERN = re.compile(r"^gmdcsa24_[0-9a-f]{16}$")
EXPECTED_SHA256 = {
    str(POSE_MODEL): (
        "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"
    ),
    str(FUSION_MODEL): (
        "9af9a610d893fa0b1a53e6cba93706d35df61b4d909297ca174c08b3a9c7b152"
    ),
    str(VERIFIER_MODEL): (
        "f63661d59b5731355c81eec8c3c2f3f79fa9c1f1f7f214a0a6c833398d6ae11c"
    ),
    str(SELECTED_POLICY): (
        "0865b34814e0481ec0ba7b7894cf6493b9b4681279910e24b1f61239e7fd4de5"
    ),
    str(DATASET_PROTOCOL): (
        "17594cce772d299a4755aaca48168b95578cfa77a0e57d6fc88277478b413f41"
    ),
}


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


def verify_blind_inventory() -> dict:
    with BLIND_INVENTORY.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if fields != EXPECTED_INVENTORY_FIELDS:
        fail(f"Blind inventory fields changed: {fields}")
    if len(rows) != EXPECTED_SEQUENCES:
        fail(f"Expected {EXPECTED_SEQUENCES} anonymous videos, found {len(rows)}")
    ids = [row["sequence_id"] for row in rows]
    if len(set(ids)) != len(ids):
        fail("Duplicate sequence_id in blind inventory")
    if any(not OPAQUE_ID_PATTERN.fullmatch(value) for value in ids):
        fail("Blind inventory contains a non-opaque sequence_id")
    for row in rows:
        relative = Path(row["video_file"])
        if relative.parent.as_posix() != "videos":
            fail(f"Unexpected blind video path: {relative}")
        video = (BLIND_ROOT / relative).resolve()
        if video.parent != (BLIND_ROOT / "videos").resolve():
            fail(f"Blind video path escapes anonymous directory: {relative}")
        if not video.is_file():
            raise FileNotFoundError(video)
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"].lower()):
            fail(f"Invalid blind video SHA-256: {row['sequence_id']}")
    return {
        "sequences": len(rows),
        "inventory_fields": fields,
        "opaque_sequence_ids": True,
        "label_or_subject_fields_present": False,
    }


def main() -> None:
    artifacts = [
        POSE_MODEL,
        FUSION_MODEL,
        VERIFIER_MODEL,
        SELECTED_POLICY,
        POSE_HELPER,
        FUSION_BASE_SOURCE,
        FUSION_MODEL_SOURCE,
        EXTRACTOR_SCRIPT,
        INFERENCE_SCRIPT,
        BLIND_INVENTORY,
        DATASET_PROTOCOL,
        Path(__file__).resolve(),
    ]
    for path in artifacts:
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT_DIR.exists():
        fail(f"Final protocol lock already exists; refusing overwrite: {OUTPUT_DIR}")
    for path in (FEATURE_OUTPUT, INFERENCE_OUTPUT, FEATURE_PARTIAL, INFERENCE_PARTIAL):
        if path.exists():
            fail(
                "A GMDCSA24 model-output path already exists before protocol lock: "
                f"{path}"
            )

    inventory_audit = verify_blind_inventory()
    hashes = {}
    for path in artifacts:
        resolved = str(path.resolve())
        actual = sha256(path)
        expected = EXPECTED_SHA256.get(resolved)
        if expected is not None and actual != expected:
            fail(
                f"Predeclared artifact SHA-256 changed: {resolved}; "
                f"expected={expected}, actual={actual}"
            )
        hashes[resolved] = actual

    verifier = json.loads(VERIFIER_MODEL.read_text(encoding="utf-8"))
    policy = json.loads(SELECTED_POLICY.read_text(encoding="utf-8"))
    if verifier.get("selected_feature_configuration") != "pre_plus_confirmation_v2":
        fail("Unexpected verifier feature configuration")
    if abs(float(verifier.get("decision_threshold", -1)) - 0.3179642728137935) > 1e-12:
        fail("Unexpected verifier decision threshold")
    if policy.get("policy_name") != "causal_rescue_alarm_policy_v2":
        fail("Unexpected selected alarm policy")
    if policy.get("gmdcsa24_read") is not False:
        fail("Development-policy file does not declare GMDCSA24 untouched")

    OUTPUT_DIR.mkdir(parents=True)
    try:
        lock_file = OUTPUT_DIR / "locked_artifacts_sha256.txt"
        lock_file.write_text(
            "".join(f"{digest}  {path}\n" for path, digest in hashes.items()),
            encoding="utf-8",
        )
        protocol = {
            "protocol_name": "gmdcsa24_final_blind_protocol_v1",
            "dataset": "GMDCSA24 v2.1 anonymous input",
            "dataset_role": "single untouched final blind test",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pre_lock_gmdcsa24_model_inference": False,
            "training_or_tuning_on_gmdcsa24": False,
            "inference_visible_labels_or_subjects": False,
            "development_datasets": ["CAUCAFall", "URFD Camera 0 RGB", "Le2i"],
            "frozen_method": {
                "pose_extractor": "YOLO26n-Pose",
                "temporal_fusion": "quality-gated RGB/Pose causal TCN",
                "resampled_fps": 20.0,
                "window_length": 16,
                "base_evidence": {
                    "alpha": 0.5,
                    "threshold": 0.85,
                    "consecutive_frames": 2,
                    "reset_threshold": 0.1,
                    "reset_frames": 5,
                },
                "primary_verifier_threshold": 0.3179642728137935,
                "confirmation_future_samples": 4,
                "maximum_added_delay_ms": 200.0,
                "causal_rescue": {
                    "pre_pose_fall_score_slope_minimum": 0.05,
                    "confirm_prob_falling_delta_minimum": 0.0,
                },
            },
            "required_execution_order": [
                "extract_gmdcsa24_frozen_features_v1.py",
                "run_gmdcsa24_frozen_inference_v1.py",
                "record output SHA-256 values",
                "only then open private mapping and evaluate",
            ],
            "post_inference_rule": (
                "No model, threshold, rescue rule, or preprocessing change after "
                "seeing GMDCSA24 results may be reported as the frozen result."
            ),
            "inventory_audit": inventory_audit,
        }
        (OUTPUT_DIR / "locked_protocol.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "status": "locked_before_first_gmdcsa24_model_inference",
            "created_utc": protocol["created_utc"],
            "locked_artifact_count": len(hashes),
            "locked_artifacts": hashes,
            "lock_file": str(lock_file),
            "lock_file_sha256": sha256(lock_file),
            "locked_protocol": str(OUTPUT_DIR / "locked_protocol.json"),
            "locked_protocol_sha256": sha256(OUTPUT_DIR / "locked_protocol.json"),
        }
        (OUTPUT_DIR / "lock_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        raise

    print("GMDCSA24 final blind protocol locked successfully.", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
