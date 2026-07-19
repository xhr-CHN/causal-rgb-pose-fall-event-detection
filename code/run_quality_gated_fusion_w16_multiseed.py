#!/usr/bin/env python3
"""Run source-only W16 fusion training for five seeds and aggregate results.

The existing verified seed-42 result is reused. Four additional seeds are
trained on CAUCAFall train/validation only. URFD, Le2i, and GMDCSA24 are never
read by this runner. The W16 inputs are the causal last 16 samples of the
already aligned W32 Pose/RGB source windows.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/data/yoloA27")
SEEDS = (7, 21, 42, 84, 2026)
WINDOW_LENGTH = 16
TRAINING_MODULE = "train_quality_gated_logit_fusion_v3_w16"
CORE_MODULE = "train_quality_gated_fusion_tcn"
OUTPUT_PATTERN = "quality_gated_logit_fusion_v3_w16_fixed_seed{seed}"
AGGREGATE_DIR = (
    ROOT / "experiments/quality_gated_logit_fusion_v3_w16_multiseed"
)
POSE_WINDOWS = (
    ROOT / "features/caucafall_pose_windows_3state_kinematic_v2"
)
RGB_WINDOWS = ROOT / "features/caucafall_rgb_roi_windows_3state_v1"

METRICS = {
    "best_selection_score": ("best_selection_score",),
    "clean_macro_f1": (
        "clean_validation_frame_metrics",
        "macro_f1",
    ),
    "clean_balanced_accuracy": (
        "clean_validation_frame_metrics",
        "balanced_accuracy",
    ),
    "clean_macro_auprc": (
        "clean_validation_frame_metrics",
        "macro_auprc_ovr",
    ),
    "clean_macro_auroc": (
        "clean_validation_frame_metrics",
        "macro_auroc_ovr",
    ),
    "falling_f1": (
        "clean_validation_frame_metrics",
        "per_class",
        "Falling",
        "f1",
    ),
    "fallen_f1": (
        "clean_validation_frame_metrics",
        "per_class",
        "Fallen",
        "f1",
    ),
    "event_recall": (
        "clean_validation_event_metrics",
        "event_recall",
    ),
    "false_alarm_count": (
        "clean_validation_event_metrics",
        "false_alarm_count",
    ),
    "mean_delay_seconds": (
        "clean_validation_event_metrics",
        "mean_delay_seconds",
    ),
    "clean_mean_pose_weight": (
        "gate_response",
        "clean_mean_pose_weight",
    ),
    "pose_missing_mean_pose_weight": (
        "gate_response",
        "pose_missing_mean_pose_weight",
    ),
    "rgb_missing_mean_pose_weight": (
        "gate_response",
        "rgb_missing_mean_pose_weight",
    ),
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested_value(document: dict, path: tuple[str, ...]) -> float:
    value = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            fail(f"Missing summary field: {'.'.join(path)}")
        value = value[key]
    result = float(value)
    if not math.isfinite(result):
        fail(f"Non-finite summary field: {'.'.join(path)}")
    return result


def output_dir(seed: int) -> Path:
    return ROOT / "experiments" / OUTPUT_PATTERN.format(seed=seed)


def validate_source_inputs() -> None:
    required = []
    for directory in (POSE_WINDOWS, RGB_WINDOWS):
        required.extend(
            [directory / "train_windows.npz", directory / "val_windows.npz"]
        )
    for path in required:
        if not path.is_file():
            fail(f"Required source window file is missing: {path}")

    for split in ("train", "val"):
        with np.load(
            POSE_WINDOWS / f"{split}_windows.npz", allow_pickle=True
        ) as pose, np.load(
            RGB_WINDOWS / f"{split}_windows.npz", allow_pickle=True
        ) as rgb:
            if pose["x"].shape[:2] != rgb["x"].shape[:2]:
                fail(f"{split}: Pose/RGB source windows are not aligned")
            if pose["x"].shape[1] < WINDOW_LENGTH:
                fail(f"{split}: fewer than {WINDOW_LENGTH} source samples")
            if not np.array_equal(pose["y"], rgb["y"]):
                fail(f"{split}: Pose/RGB labels are not aligned")


def install_w16_loader(core, trainer) -> None:
    core.POSE_DIR = POSE_WINDOWS
    core.RGB_DIR = RGB_WINDOWS
    original_load_split = core.load_split

    def load_split_w16(split):
        result = original_load_split(split)
        (
            pose,
            rgb,
            quality,
            labels,
            pose_names,
            rgb_names,
        ) = result
        return (
            pose[:, -WINDOW_LENGTH:, :].copy(),
            rgb[:, -WINDOW_LENGTH:, :].copy(),
            quality[:, -WINDOW_LENGTH:, :].copy(),
            labels,
            pose_names,
            rgb_names,
        )

    trainer.load_split = load_split_w16


def validate_completed_seed(seed: int) -> tuple[dict, Path, Path]:
    directory = output_dir(seed)
    summary_path = directory / "summary.json"
    checkpoint_path = directory / "quality_gated_logit_fusion_v3_best.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        fail(f"Seed {seed} output is incomplete: {directory}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = summary.get("protocol", {})
    if int(protocol.get("seed", -1)) != seed:
        fail(f"Seed mismatch inside {summary_path}")
    if int(protocol.get("window_length", -1)) != WINDOW_LENGTH:
        fail(f"W16 verification failed inside {summary_path}")
    return summary, summary_path, checkpoint_path


def train_missing_seeds() -> None:
    import torch

    core = importlib.import_module(CORE_MODULE)
    trainer = importlib.import_module(TRAINING_MODULE)
    install_w16_loader(core, trainer)

    for seed in SEEDS:
        directory = output_dir(seed)
        summary_path = directory / "summary.json"
        checkpoint_path = directory / "quality_gated_logit_fusion_v3_best.pt"
        if summary_path.is_file() and checkpoint_path.is_file():
            validate_completed_seed(seed)
            print(f"Seed {seed}: verified existing result", flush=True)
            continue
        if directory.exists():
            fail(
                f"Seed {seed} has a partial output directory; move it aside "
                f"before retrying: {directory}"
            )

        trainer.SEED = seed
        trainer.OUT_DIR = directory
        print(f"\n===== TRAINING SEED {seed} =====", flush=True)
        trainer.main()
        validate_completed_seed(seed)
        print(f"Seed {seed}: training completed and verified", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate() -> None:
    if AGGREGATE_DIR.exists():
        fail(
            "Aggregate output already exists; refusing overwrite: "
            f"{AGGREGATE_DIR}"
        )

    per_seed = []
    artifact_hashes = {}
    for seed in SEEDS:
        summary, summary_path, checkpoint_path = validate_completed_seed(seed)
        row = {
            "seed": seed,
            "best_epoch": int(summary["best_epoch"]),
        }
        for metric_name, metric_path in METRICS.items():
            row[metric_name] = nested_value(summary, metric_path)
        per_seed.append(row)
        artifact_hashes[str(summary_path)] = sha256(summary_path)
        artifact_hashes[str(checkpoint_path)] = sha256(checkpoint_path)

    statistics = {}
    for metric_name in ["best_epoch"] + list(METRICS):
        values = np.array(
            [float(row[metric_name]) for row in per_seed], dtype=np.float64
        )
        statistics[metric_name] = {
            "mean": float(values.mean()),
            "sample_standard_deviation": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    AGGREGATE_DIR.mkdir(parents=True)
    per_seed_path = AGGREGATE_DIR / "per_seed_metrics.csv"
    with per_seed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)

    summary = {
        "protocol": {
            "source_dataset": "CAUCAFall train/validation only",
            "external_datasets_read": False,
            "gmdcsa24_read": False,
            "window_length": WINDOW_LENGTH,
            "seeds": list(SEEDS),
            "seed_42_reused": True,
            "additional_training_runs": 4,
            "purpose": "training-stability mean and sample standard deviation",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "per_seed": per_seed,
        "statistics": statistics,
        "artifact_sha256": artifact_hashes,
        "runner_sha256": sha256(Path(__file__)),
    }
    summary_path = AGGREGATE_DIR / "multiseed_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    hashes_path = AGGREGATE_DIR / "output_sha256.txt"
    hashes_path.write_text(
        f"{sha256(per_seed_path)}  per_seed_metrics.csv\n"
        f"{sha256(summary_path)}  multiseed_summary.json\n",
        encoding="utf-8",
    )
    print("\n===== MULTI-SEED STATISTICS =====", flush=True)
    for metric_name, values in statistics.items():
        print(
            f"{metric_name}: {values['mean']:.6f} +/- "
            f"{values['sample_standard_deviation']:.6f}",
            flush=True,
        )
    print(f"\nCompleted: {AGGREGATE_DIR}", flush=True)


def main() -> None:
    validate_source_inputs()
    if not output_dir(42).is_dir():
        fail("Verified seed-42 W16 result is missing")
    train_missing_seeds()
    aggregate()


if __name__ == "__main__":
    main()
