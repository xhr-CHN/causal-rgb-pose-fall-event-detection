#!/usr/bin/env python3
"""Benchmark the frozen fall-detection model components on RTX 3090 (V2).

The benchmark uses FP32 and batch size 1. It reports wall-clock latency for
full-frame YOLO-Pose inference, person-ROI embedding inference, and the W16
quality-gated fusion model. Image decoding is excluded because images are
preloaded. Fusion inputs are real CAUCAFall validation windows. Before timing
ROI embedding, one complete pass over all representative ROI shapes is used to
remove one-time cuDNN shape-autotuning cost from the steady-state latency.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from train_quality_gated_logit_fusion_v2 import ReliabilityGatedLogitFusion


ROOT = Path("/home/data/yoloA27")
POSE_MODEL_PATH = ROOT / "yolo26n-pose.pt"
FUSION_CHECKPOINT = (
    ROOT
    / "experiments/quality_gated_logit_fusion_v3_w16_fixed_seed42/"
    "quality_gated_logit_fusion_v3_best.pt"
)
VAL_IMAGE_DIR = ROOT / "CAUCAFall_YOLO/images/val"
VAL_LABEL_DIR = ROOT / "CAUCAFall_YOLO/labels/val"
POSE_WINDOWS = (
    ROOT
    / "features/caucafall_pose_windows_3state_kinematic_v2/val_windows.npz"
)
RGB_WINDOWS = (
    ROOT
    / "features/caucafall_rgb_roi_windows_3state_v1/val_windows.npz"
)
OUTPUT_DIR = ROOT / "experiments/deployment_benchmark_rtx3090_v2"

IMAGE_SIZE = 640
DEVICE = 0
SAMPLE_IMAGES = 200
POSE_GPU_WARMUP = 20
EMBEDDING_WARMUP_PASSES = 1
GPU_REPETITIONS = 200
FUSION_GPU_WARMUP = 100
FUSION_GPU_REPETITIONS = 1000
FUSION_CPU_WARMUP = 50
FUSION_CPU_REPETITIONS = 500
CPU_THREADS = min(16, os.cpu_count() or 1)
QUALITY_NAMES = (
    "pose_found",
    "person_conf",
    "visible_keypoint_ratio",
    "mean_keypoint_conf",
    "torso_keypoint_conf",
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latency_summary(values_ms: list[float]) -> dict[str, float]:
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "repetitions": int(len(values)),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std(ddof=1)),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "minimum_ms": float(values.min()),
        "maximum_ms": float(values.max()),
        "fps_from_mean_latency": float(1000.0 / values.mean()),
    }


def evenly_spaced_paths() -> list[Path]:
    paths = sorted(
        path
        for path in VAL_IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if len(paths) < SAMPLE_IMAGES:
        fail(f"Only {len(paths)} validation images were found")
    indices = np.linspace(0, len(paths) - 1, SAMPLE_IMAGES, dtype=np.int64)
    return [paths[int(index)] for index in indices]


def crop_from_yolo_label(image: np.ndarray, image_path: Path) -> np.ndarray:
    label_path = VAL_LABEL_DIR / f"{image_path.stem}.txt"
    if not label_path.is_file():
        return image
    lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    if not lines:
        return image
    values = lines[0].split()
    if len(values) != 5:
        return image
    _, cx, cy, width, height = map(float, values)
    image_height, image_width = image.shape[:2]
    x1 = (cx - width / 2.0) * image_width
    y1 = (cy - height / 2.0) * image_height
    x2 = (cx + width / 2.0) * image_width
    y2 = (cy + height / 2.0) * image_height
    padding_x = 0.10 * max(1.0, x2 - x1)
    padding_y = 0.10 * max(1.0, y2 - y1)
    left = max(0, int(np.floor(x1 - padding_x)))
    top = max(0, int(np.floor(y1 - padding_y)))
    right = min(image_width, int(np.ceil(x2 + padding_x)))
    bottom = min(image_height, int(np.ceil(y2 + padding_y)))
    if right <= left or bottom <= top:
        return image
    return image[top:bottom, left:right].copy()


def preload_images() -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    paths = evenly_spaced_paths()
    images = []
    crops = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            fail(f"OpenCV failed to read {path}")
        images.append(image)
        crops.append(crop_from_yolo_label(image, path))
    return images, crops, [str(path) for path in paths]


def synchronize() -> None:
    torch.cuda.synchronize(DEVICE)


def benchmark_yolo_pose(model: YOLO, images: list[np.ndarray]) -> tuple[dict, float]:
    for index in range(POSE_GPU_WARMUP):
        output = model.predict(
            source=images[index % len(images)],
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            conf=0.001,
            iou=0.7,
            verbose=False,
        )
        del output
    synchronize()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    baseline = torch.cuda.memory_allocated(DEVICE)
    latencies = []
    for index in range(GPU_REPETITIONS):
        synchronize()
        start = time.perf_counter()
        output = model.predict(
            source=images[index % len(images)],
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            conf=0.001,
            iou=0.7,
            verbose=False,
        )
        synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
        del output
    peak_delta = max(0, torch.cuda.max_memory_allocated(DEVICE) - baseline)
    return latency_summary(latencies), peak_delta / (1024.0**2)


def benchmark_yolo_embedding(
    model: YOLO, crops: list[np.ndarray]
) -> tuple[dict, float]:
    for _ in range(EMBEDDING_WARMUP_PASSES):
        for crop in crops:
            output = model.embed(
                source=crop,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                batch=1,
                verbose=False,
            )
            del output
    synchronize()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    baseline = torch.cuda.memory_allocated(DEVICE)
    latencies = []
    output_dimensions = set()
    for index in range(GPU_REPETITIONS):
        synchronize()
        start = time.perf_counter()
        output = model.embed(
            source=crops[index % len(crops)],
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            batch=1,
            verbose=False,
        )
        synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
        if len(output) != 1:
            fail("Unexpected embedding output count")
        output_dimensions.add(int(output[0].numel()))
        del output
    if output_dimensions != {256}:
        fail(f"Unexpected embedding dimensions: {output_dimensions}")
    peak_delta = max(0, torch.cuda.max_memory_allocated(DEVICE) - baseline)
    return latency_summary(latencies), peak_delta / (1024.0**2)


def load_fusion_inputs(checkpoint: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(POSE_WINDOWS, allow_pickle=True) as pose_data, np.load(
        RGB_WINDOWS, allow_pickle=True
    ) as rgb_data:
        pose = pose_data["x"].astype(np.float32)[:, -16:, :]
        rgb = rgb_data["x"].astype(np.float32)[:, -16:, :]
        if pose.shape[:2] != rgb.shape[:2]:
            fail("Pose/RGB validation windows are not aligned")
        pose_names = pose_data["feature_names"].astype(str).tolist()
    quality_indices = []
    for name in QUALITY_NAMES:
        if name not in pose_names:
            fail(f"Quality feature missing: {name}")
        quality_indices.append(pose_names.index(name))
    quality = np.clip(pose[:, :, quality_indices], 0.0, 1.0).astype(np.float32)
    pose = (
        pose - np.asarray(checkpoint["pose_mean"], dtype=np.float32)[None, None, :]
    ) / np.asarray(checkpoint["pose_std"], dtype=np.float32)[None, None, :]
    rgb = (
        rgb - np.asarray(checkpoint["rgb_mean"], dtype=np.float32)[None, None, :]
    ) / np.asarray(checkpoint["rgb_std"], dtype=np.float32)[None, None, :]
    return pose.astype(np.float32), rgb.astype(np.float32), quality


def create_fusion_model(checkpoint: dict, device: torch.device):
    config = checkpoint["model_config"]
    if int(config["window_length"]) != 16:
        fail("Fusion checkpoint is not W16")
    model = ReliabilityGatedLogitFusion(
        config["pose_dim"],
        config["rgb_dim"],
        config["quality_dim"],
        config["hidden_dim"],
        config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def benchmark_fusion_gpu(
    model, pose: np.ndarray, rgb: np.ndarray, quality: np.ndarray
) -> tuple[dict, float]:
    device = torch.device("cuda:0")
    count = min(len(pose), FUSION_GPU_REPETITIONS)
    pose_tensor = torch.from_numpy(pose[:count]).to(device)
    rgb_tensor = torch.from_numpy(rgb[:count]).to(device)
    quality_tensor = torch.from_numpy(quality[:count]).to(device)
    for index in range(FUSION_GPU_WARMUP):
        output = model(
            pose_tensor[index % count : index % count + 1],
            rgb_tensor[index % count : index % count + 1],
            quality_tensor[index % count : index % count + 1],
        )
        del output
    synchronize()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    baseline = torch.cuda.memory_allocated(DEVICE)
    latencies = []
    for index in range(FUSION_GPU_REPETITIONS):
        sample = index % count
        synchronize()
        start = time.perf_counter()
        output = model(
            pose_tensor[sample : sample + 1],
            rgb_tensor[sample : sample + 1],
            quality_tensor[sample : sample + 1],
        )
        synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
        del output
    peak_delta = max(0, torch.cuda.max_memory_allocated(DEVICE) - baseline)
    return latency_summary(latencies), peak_delta / (1024.0**2)


@torch.no_grad()
def benchmark_fusion_cpu(
    checkpoint: dict, pose: np.ndarray, rgb: np.ndarray, quality: np.ndarray
) -> dict:
    torch.set_num_threads(CPU_THREADS)
    model = create_fusion_model(checkpoint, torch.device("cpu"))
    count = min(len(pose), FUSION_CPU_REPETITIONS)
    pose_tensor = torch.from_numpy(pose[:count])
    rgb_tensor = torch.from_numpy(rgb[:count])
    quality_tensor = torch.from_numpy(quality[:count])
    for index in range(FUSION_CPU_WARMUP):
        sample = index % count
        output = model(
            pose_tensor[sample : sample + 1],
            rgb_tensor[sample : sample + 1],
            quality_tensor[sample : sample + 1],
        )
        del output
    latencies = []
    for index in range(FUSION_CPU_REPETITIONS):
        sample = index % count
        start = time.perf_counter()
        output = model(
            pose_tensor[sample : sample + 1],
            rgb_tensor[sample : sample + 1],
            quality_tensor[sample : sample + 1],
        )
        latencies.append((time.perf_counter() - start) * 1000.0)
        del output
    return latency_summary(latencies)


def main() -> None:
    required = [
        POSE_MODEL_PATH,
        FUSION_CHECKPOINT,
        VAL_IMAGE_DIR,
        VAL_LABEL_DIR,
        POSE_WINDOWS,
        RGB_WINDOWS,
    ]
    for path in required:
        if not path.exists():
            fail(f"Required benchmark input missing: {path}")
    if OUTPUT_DIR.exists():
        fail(f"Output already exists; refusing overwrite: {OUTPUT_DIR}")
    if not torch.cuda.is_available():
        fail("CUDA is required for the RTX 3090 benchmark")

    torch.backends.cudnn.benchmark = True
    images, crops, selected_paths = preload_images()
    checkpoint = torch.load(FUSION_CHECKPOINT, map_location="cpu")
    pose_windows, rgb_windows, quality_windows = load_fusion_inputs(checkpoint)

    pose_model = YOLO(str(POSE_MODEL_PATH))
    pose_parameters = sum(parameter.numel() for parameter in pose_model.model.parameters())
    model_info = pose_model.info(verbose=False)
    pose_latency, pose_peak_mb = benchmark_yolo_pose(pose_model, images)
    embedding_latency, embedding_peak_mb = benchmark_yolo_embedding(
        pose_model, crops
    )

    fusion_model = create_fusion_model(checkpoint, torch.device("cuda:0"))
    fusion_parameters = sum(parameter.numel() for parameter in fusion_model.parameters())
    fusion_gpu_latency, fusion_peak_mb = benchmark_fusion_gpu(
        fusion_model, pose_windows, rgb_windows, quality_windows
    )
    fusion_cpu_latency = benchmark_fusion_cpu(
        checkpoint, pose_windows, rgb_windows, quality_windows
    )

    combined_mean_ms = (
        pose_latency["mean_ms"]
        + embedding_latency["mean_ms"]
        + fusion_gpu_latency["mean_ms"]
    )
    combined_p95_ms = (
        pose_latency["p95_ms"]
        + embedding_latency["p95_ms"]
        + fusion_gpu_latency["p95_ms"]
    )
    components = {
        "yolo_pose_gpu_fp32_batch1": {
            **pose_latency,
            "peak_incremental_allocated_memory_mb": pose_peak_mb,
        },
        "rgb_roi_embedding_gpu_fp32_batch1": {
            **embedding_latency,
            "peak_incremental_allocated_memory_mb": embedding_peak_mb,
        },
        "w16_fusion_gpu_fp32_batch1": {
            **fusion_gpu_latency,
            "peak_incremental_allocated_memory_mb": fusion_peak_mb,
        },
        "w16_fusion_cpu_fp32_batch1": fusion_cpu_latency,
        "sequential_gpu_model_core_estimate": {
            "mean_ms": combined_mean_ms,
            "p95_upper_sum_ms": combined_p95_ms,
            "fps_from_mean_latency": 1000.0 / combined_mean_ms,
        },
    }

    rows = []
    for component, metrics in components.items():
        row = {"component": component}
        row.update(metrics)
        rows.append(row)

    OUTPUT_DIR.mkdir(parents=True)
    component_path = OUTPUT_DIR / "component_latency.csv"
    write_csv(component_path, rows)
    summary = {
        "protocol": {
            "benchmark_device": "NVIDIA GeForce RTX 3090",
            "precision": "FP32",
            "batch_size": 1,
            "image_size": IMAGE_SIZE,
            "image_decode_included": False,
            "roi_crop_included": False,
            "feature_formatting_included": False,
            "sequential_core_estimate": (
                "YOLO-Pose + ROI embedding + W16 fusion latency sum"
            ),
            "source_images": "200 evenly spaced CAUCAFall validation images",
            "pose_warmup_repetitions": POSE_GPU_WARMUP,
            "embedding_warmup": (
                "one complete untimed pass over all 200 representative ROI crops"
            ),
            "embedding_warmup_passes": EMBEDDING_WARMUP_PASSES,
            "labels_used_for_model_or_policy_selection": False,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "gpu": torch.cuda.get_device_name(DEVICE),
            "cpu": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "fusion_cpu_threads": CPU_THREADS,
        },
        "complexity": {
            "yolo_pose_parameters": pose_parameters,
            "yolo_pose_weight_bytes": POSE_MODEL_PATH.stat().st_size,
            "yolo_info_return": repr(model_info),
            "fusion_parameters": fusion_parameters,
            "fusion_checkpoint_bytes": FUSION_CHECKPOINT.stat().st_size,
            "total_parameter_count_if_sequential": pose_parameters
            + fusion_parameters,
        },
        "latency": components,
        "selected_image_paths": selected_paths,
        "input_sha256": {
            str(POSE_MODEL_PATH): sha256(POSE_MODEL_PATH),
            str(FUSION_CHECKPOINT): sha256(FUSION_CHECKPOINT),
            str(POSE_WINDOWS): sha256(POSE_WINDOWS),
            str(RGB_WINDOWS): sha256(RGB_WINDOWS),
        },
        "script_sha256": sha256(Path(__file__)),
    }
    summary_path = OUTPUT_DIR / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "output_sha256.txt").write_text(
        f"{sha256(component_path)}  component_latency.csv\n"
        f"{sha256(summary_path)}  benchmark_summary.json\n",
        encoding="utf-8",
    )
    print(json.dumps(components, ensure_ascii=False, indent=2), flush=True)
    print(f"Completed: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
