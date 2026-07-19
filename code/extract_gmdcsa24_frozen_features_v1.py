#!/usr/bin/env python3
"""Extract label-isolated Pose and RGB-ROI features from anonymous GMDCSA24.

Only GMDCSA24_FROZEN_INPUT_V1/blind_video_inventory.csv and its anonymous
videos are accessible to this script. Raw paths, private mappings, category
folders, and official CSV files are never opened. Each video is committed
atomically so an interrupted extraction can resume at video boundaries.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from extract_caucafall_pose_features_v1 import missing_pose_values, pose_values


ROOT = Path("/home/data/yoloA27")
BLIND_ROOT = ROOT / "GMDCSA24_FROZEN_INPUT_V1"
BLIND_VIDEO_DIR = BLIND_ROOT / "videos"
BLIND_INVENTORY = BLIND_ROOT / "blind_video_inventory.csv"
LOCK_DIR = ROOT / "experiments/gmdcsa24_final_blind_protocol_lock_v1"
LOCK_FILE = LOCK_DIR / "locked_artifacts_sha256.txt"
MODEL_PATH = ROOT / "yolo26n-pose.pt"
POSE_HELPER_PATH = ROOT / "extract_caucafall_pose_features_v1.py"
OUTPUT_DIR = ROOT / "features/gmdcsa24_frozen_features_v1"
SEQUENCE_DIR = OUTPUT_DIR / "sequences"

EXPECTED_SEQUENCES = 160
EMBEDDING_DIMENSION = 256
IMAGE_SIZE = 640
DEVICE = 0
SOURCE_CHUNK_SIZE = 32
POSE_BATCH_SIZE = 8
EMBED_BATCH_SIZE = 16
POSE_CONFIDENCE_FLOOR = 0.01
NMS_IOU = 0.7
ROI_PADDING_RATIO = 0.10
REPORT_EVERY_FRAMES = 256
OPAQUE_ID_PATTERN = re.compile(r"^gmdcsa24_[0-9a-f]{16}$")

INVENTORY_FIELDS = ["sequence_id", "video_file", "sha256"]
POSE_METADATA_FIELDS = [
    "sequence_id",
    "frame_number",
    "timestamp_ms",
    "timestamp_source",
    "image_name",
    "video_path",
]
RGB_METADATA_FIELDS = [
    "sequence_id",
    "frame_number",
    "timestamp_ms",
    "timestamp_source",
    "image_name",
    "video_path",
    "pose_found",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "roi_crop_used",
]


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


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def read_lock() -> dict[str, str]:
    locked: dict[str, str] = {}
    for line_number, line in enumerate(
        LOCK_FILE.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        if len(line) < 67 or line[64:66] != "  ":
            fail(f"Malformed lock line {line_number}")
        locked[str(Path(line[66:]).resolve())] = line[:64].lower()
    return locked


def verify_locked_artifacts() -> dict[str, str]:
    locked = read_lock()
    required = [MODEL_PATH, POSE_HELPER_PATH, Path(__file__).resolve(), BLIND_INVENTORY]
    verified = {}
    for path in required:
        resolved = str(path.resolve())
        if resolved not in locked:
            fail(f"Required artifact is absent from final lock: {resolved}")
        actual = sha256(path)
        if actual != locked[resolved]:
            fail(f"Locked artifact hash changed: {resolved}")
        verified[resolved] = actual
    return verified


def safe_blind_video(relative: str) -> Path:
    path = (BLIND_ROOT / relative).resolve()
    video_root = BLIND_VIDEO_DIR.resolve()
    if path.parent != video_root:
        fail(f"Anonymous inventory path escapes blind video directory: {relative}")
    if path.suffix.lower() not in {".mp4", ".avi", ".mov"}:
        fail(f"Unsupported anonymous video suffix: {relative}")
    return path


def load_blind_inventory() -> list[dict[str, object]]:
    rows, fields = read_csv(BLIND_INVENTORY)
    if fields != INVENTORY_FIELDS:
        fail(f"Blind inventory fields changed: {fields}")
    if len(rows) != EXPECTED_SEQUENCES:
        fail(f"Expected {EXPECTED_SEQUENCES} anonymous videos, found {len(rows)}")
    if len({row["sequence_id"] for row in rows}) != len(rows):
        fail("Duplicate anonymous sequence_id")

    approved = []
    for row in rows:
        sequence_id = row["sequence_id"]
        if not OPAQUE_ID_PATTERN.fullmatch(sequence_id):
            fail(f"Non-opaque sequence identifier: {sequence_id}")
        path = safe_blind_video(row["video_file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = row["sha256"].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            fail(f"Invalid video SHA-256 for {sequence_id}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            fail(f"Anonymous video hash mismatch: {sequence_id}")
        approved.append(
            {
                "sequence_id": sequence_id,
                "video_file": row["video_file"],
                "video_path": str(path),
                "sha256": actual_hash,
            }
        )
    approved.sort(key=lambda row: str(row["sequence_id"]))
    return approved


def fraction_value(value: str) -> float:
    if not value or value in {"N/A", "0/0"}:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def run_ffprobe(path: Path, count_frames: bool = False) -> dict:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        command.append("-count_frames")
    command += [
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        fail(f"Expected one selected video stream: {path}")
    return streams[0]


def video_metadata(path: Path) -> dict[str, object]:
    stream = run_ffprobe(path, count_frames=False)
    frame_text = str(stream.get("nb_frames", ""))
    if not frame_text.isdigit() or int(frame_text) <= 0:
        stream = run_ffprobe(path, count_frames=True)
        frame_text = str(stream.get("nb_read_frames", ""))
    if not frame_text.isdigit() or int(frame_text) <= 0:
        fail(f"Unable to determine exact frame count: {path}")
    fps = fraction_value(str(stream.get("avg_frame_rate", "")))
    if fps <= 0.0:
        fps = fraction_value(str(stream.get("r_frame_rate", "")))
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    frames = int(frame_text)
    duration = float(stream.get("duration", 0.0) or 0.0)
    if not (1.0 <= fps <= 120.0 and width > 0 and height > 0 and frames > 0):
        fail(f"Invalid anonymous video metadata: {path}")
    return {
        "codec_name": str(stream.get("codec_name", "")),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frames,
        "duration_seconds": duration,
    }


class FFmpegVideoReader:
    def __init__(self, path: Path, width: int, height: int):
        self.path = path
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self.process: Optional[subprocess.Popen] = None

    def __enter__(self):
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(self.path),
            "-map",
            "0:v:0",
            "-an",
            "-vsync",
            "0",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_bytes * 4,
        )
        return self

    def read(self) -> Optional[np.ndarray]:
        if self.process is None or self.process.stdout is None:
            fail("FFmpeg reader is not open")
        data = bytearray()
        while len(data) < self.frame_bytes:
            block = self.process.stdout.read(self.frame_bytes - len(data))
            if not block:
                break
            data.extend(block)
        if not data:
            return None
        if len(data) != self.frame_bytes:
            fail(
                f"Incomplete raw frame from {self.path}: "
                f"{len(data)}/{self.frame_bytes} bytes"
            )
        return np.frombuffer(data, dtype=np.uint8).reshape(
            self.height, self.width, 3
        ).copy()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.process is None:
            return False
        if exc_type is not None and self.process.poll() is None:
            self.process.terminate()
        if self.process.stdout is not None:
            self.process.stdout.close()
        try:
            return_code = self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            return_code = self.process.wait()
        stderr = b""
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
            self.process.stderr.close()
        if exc_type is None and return_code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            fail(
                f"FFmpeg exited with status {return_code} for {self.path}: "
                f"{message[-2000:]}"
            )
        return False


def ffmpeg_version() -> str:
    completed = subprocess.run(
        ["ffmpeg", "-version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.splitlines()[0]


def pose_fieldnames() -> list[str]:
    return POSE_METADATA_FIELDS + list(missing_pose_values().keys())


def virtual_image_name(sequence_id: str, frame_number: int) -> str:
    return f"{sequence_id}_frame_{frame_number:06d}.jpg"


def crop_person(frame: np.ndarray, values: dict) -> tuple[np.ndarray, int]:
    height, width = frame.shape[:2]
    if int(values["pose_found"]) != 1:
        return frame, 0
    x1 = float(values["bbox_x1"])
    y1 = float(values["bbox_y1"])
    x2 = float(values["bbox_x2"])
    y2 = float(values["bbox_y2"])
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return frame, 0
    pad_x = (x2 - x1) * ROI_PADDING_RATIO
    pad_y = (y2 - y1) * ROI_PADDING_RATIO
    left = max(0, int(np.floor((x1 - pad_x) * width)))
    top = max(0, int(np.floor((y1 - pad_y) * height)))
    right = min(width, int(np.ceil((x2 + pad_x) * width)))
    bottom = min(height, int(np.ceil((y2 + pad_y) * height)))
    if right - left < 8 or bottom - top < 8:
        return frame, 0
    return frame[top:bottom, left:right].copy(), 1


def completed_sequence_summary(final_dir: Path, expected_frames: int):
    paths = {
        "summary": final_dir / "sequence_summary.json",
        "pose": final_dir / "pose_features.csv",
        "rgb": final_dir / "rgb_metadata.csv",
        "embedding": final_dir / "rgb_embeddings.npy",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or int(summary.get("frames", -1)) != expected_frames:
        return None
    embeddings = np.load(paths["embedding"], mmap_mode="r")
    if embeddings.shape != (expected_frames, EMBEDDING_DIMENSION):
        return None
    if count_csv_rows(paths["pose"]) != expected_frames:
        return None
    if count_csv_rows(paths["rgb"]) != expected_frames:
        return None
    return summary


def extract_sequence(
    row: dict[str, object], pose_model: YOLO, embed_model: YOLO
) -> dict[str, object]:
    sequence_id = str(row["sequence_id"])
    expected_frames = int(row["frame_count"])
    fps = float(row["fps"])
    width = int(row["width"])
    height = int(row["height"])
    video_path = Path(str(row["video_path"]))
    final_dir = SEQUENCE_DIR / sequence_id
    existing = completed_sequence_summary(final_dir, expected_frames)
    if existing is not None:
        print(f"{sequence_id}: already complete, skipped", flush=True)
        return existing
    if final_dir.exists():
        fail(f"Incomplete committed sequence directory requires review: {final_dir}")
    partial_dir = SEQUENCE_DIR / f"{sequence_id}.partial"
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    embeddings = np.lib.format.open_memmap(
        partial_dir / "rgb_embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(expected_frames, EMBEDDING_DIMENSION),
    )
    pose_handle = (partial_dir / "pose_features.csv").open(
        "w", encoding="utf-8", newline=""
    )
    rgb_handle = (partial_dir / "rgb_metadata.csv").open(
        "w", encoding="utf-8", newline=""
    )
    pose_writer = csv.DictWriter(pose_handle, fieldnames=pose_fieldnames())
    rgb_writer = csv.DictWriter(rgb_handle, fieldnames=RGB_METADATA_FIELDS)
    pose_writer.writeheader()
    rgb_writer.writeheader()

    frame_offset = 0
    pose_found_count = 0
    roi_count = 0
    norm_sum = 0.0
    norm_squared_sum = 0.0
    norm_minimum = float("inf")
    norm_maximum = float("-inf")
    try:
        with FFmpegVideoReader(video_path, width, height) as decoder:
            while True:
                frames = []
                for _ in range(SOURCE_CHUNK_SIZE):
                    frame = decoder.read()
                    if frame is None:
                        break
                    frames.append(frame)
                if not frames:
                    break
                if frame_offset + len(frames) > expected_frames:
                    fail(f"Decoded more than {expected_frames} frames in {sequence_id}")

                pose_results = list(
                    pose_model.predict(
                        source=frames,
                        imgsz=IMAGE_SIZE,
                        batch=POSE_BATCH_SIZE,
                        device=DEVICE,
                        conf=POSE_CONFIDENCE_FLOOR,
                        iou=NMS_IOU,
                        stream=True,
                        verbose=False,
                    )
                )
                if len(pose_results) != len(frames):
                    fail(f"Pose output mismatch in {sequence_id}")
                value_rows = [pose_values(result) for result in pose_results]
                crops_and_flags = [
                    crop_person(frame, values)
                    for frame, values in zip(frames, value_rows)
                ]
                crops = [item[0] for item in crops_and_flags]
                crop_flags = [item[1] for item in crops_and_flags]
                embed_outputs = list(
                    embed_model.embed(
                        source=crops,
                        imgsz=IMAGE_SIZE,
                        device=DEVICE,
                        batch=EMBED_BATCH_SIZE,
                        verbose=False,
                    )
                )
                if len(embed_outputs) != len(frames):
                    fail(f"Embedding output mismatch in {sequence_id}")
                embedding_array = np.stack(
                    [output.detach().float().cpu().numpy() for output in embed_outputs]
                ).astype(np.float32)
                if embedding_array.shape != (len(frames), EMBEDDING_DIMENSION):
                    fail(f"Unexpected embedding shape in {sequence_id}: {embedding_array.shape}")
                if not np.isfinite(embedding_array).all():
                    fail(f"Non-finite embedding in {sequence_id}")
                embeddings[frame_offset : frame_offset + len(frames)] = embedding_array

                norms = np.linalg.norm(embedding_array, axis=1)
                norm_sum += float(norms.sum())
                norm_squared_sum += float(np.square(norms).sum())
                norm_minimum = min(norm_minimum, float(norms.min()))
                norm_maximum = max(norm_maximum, float(norms.max()))
                for local_index, (values, crop_used) in enumerate(
                    zip(value_rows, crop_flags), start=1
                ):
                    frame_number = frame_offset + local_index
                    timestamp_ms = (frame_number - 1) * 1000.0 / fps
                    image_name = virtual_image_name(sequence_id, frame_number)
                    pose_row = {
                        "sequence_id": sequence_id,
                        "frame_number": frame_number,
                        "timestamp_ms": timestamp_ms,
                        "timestamp_source": "frame_index_and_ffprobe_average_fps",
                        "image_name": image_name,
                        "video_path": str(video_path),
                    }
                    pose_row.update(values)
                    pose_writer.writerow(pose_row)
                    rgb_writer.writerow(
                        {
                            "sequence_id": sequence_id,
                            "frame_number": frame_number,
                            "timestamp_ms": timestamp_ms,
                            "timestamp_source": "frame_index_and_ffprobe_average_fps",
                            "image_name": image_name,
                            "video_path": str(video_path),
                            "pose_found": values["pose_found"],
                            "bbox_x1": values["bbox_x1"],
                            "bbox_y1": values["bbox_y1"],
                            "bbox_x2": values["bbox_x2"],
                            "bbox_y2": values["bbox_y2"],
                            "roi_crop_used": crop_used,
                        }
                    )
                pose_found_count += sum(int(values["pose_found"]) for values in value_rows)
                roi_count += sum(crop_flags)
                frame_offset += len(frames)
                embeddings.flush()
                if frame_offset % REPORT_EVERY_FRAMES < SOURCE_CHUNK_SIZE or frame_offset == expected_frames:
                    pose_handle.flush()
                    rgb_handle.flush()
                    print(f"{sequence_id}: {frame_offset}/{expected_frames} frames", flush=True)
    finally:
        pose_handle.close()
        rgb_handle.close()
        embeddings.flush()
        del embeddings

    if frame_offset != expected_frames:
        fail(
            f"Frame count mismatch in {sequence_id}: "
            f"decoded={frame_offset}, expected={expected_frames}"
        )
    norm_mean = norm_sum / expected_frames
    norm_variance = max(0.0, norm_squared_sum / expected_frames - norm_mean**2)
    summary = {
        "status": "complete",
        "sequence_id": sequence_id,
        "video_path": str(video_path),
        "video_sha256": row["sha256"],
        "frames": expected_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": row["duration_seconds"],
        "pose_found_frames": pose_found_count,
        "pose_detection_rate": pose_found_count / expected_frames,
        "roi_crops": roi_count,
        "full_image_fallbacks": expected_frames - roi_count,
        "roi_crop_rate": roi_count / expected_frames,
        "embedding_shape": [expected_frames, EMBEDDING_DIMENSION],
        "video_decoder": "system_ffmpeg_first_video_stream_audio_disabled",
        "embedding_norm": {
            "minimum": norm_minimum,
            "maximum": norm_maximum,
            "mean": norm_mean,
            "standard_deviation": norm_variance**0.5,
        },
    }
    (partial_dir / "sequence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(partial_dir, final_dir)
    print(f"{sequence_id}: committed", flush=True)
    return summary


def main() -> None:
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"Required executable not found: {executable}")
    for path in (BLIND_INVENTORY, LOCK_FILE, MODEL_PATH, POSE_HELPER_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        fail("CUDA GPU is required for frozen feature extraction")

    locked_hashes = verify_locked_artifacts()
    inventory = load_blind_inventory()
    rows = []
    print("Inspecting anonymous video metadata with ffprobe...", flush=True)
    for index, row in enumerate(inventory, start=1):
        metadata = video_metadata(Path(str(row["video_path"])))
        merged = dict(row)
        merged.update(metadata)
        rows.append(merged)
        if index % 20 == 0 or index == len(inventory):
            print(f"Metadata: {index}/{len(inventory)}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / "label_free_video_inventory.csv",
        rows,
        [
            "sequence_id",
            "video_file",
            "video_path",
            "sha256",
            "codec_name",
            "width",
            "height",
            "fps",
            "frame_count",
            "duration_seconds",
        ],
    )

    print(f"Python: {platform.python_version()}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Ultralytics: {ultralytics.__version__}", flush=True)
    print(f"FFmpeg: {ffmpeg_version()}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}", flush=True)
    print("GMDCSA24 labels available to extractor: NO", flush=True)
    print("Resume unit: one complete anonymous video", flush=True)

    pose_model = YOLO(str(MODEL_PATH))
    embed_model = YOLO(str(MODEL_PATH))
    summaries = []
    for index, row in enumerate(rows, start=1):
        print(f"\nSequence {index}/{len(rows)}: {row['sequence_id']}", flush=True)
        summaries.append(extract_sequence(row, pose_model, embed_model))
        torch.cuda.empty_cache()

    manifest_rows = []
    for summary in summaries:
        sequence_id = str(summary["sequence_id"])
        directory = SEQUENCE_DIR / sequence_id
        manifest_rows.append(
            {
                "sequence_id": sequence_id,
                "frames": summary["frames"],
                "fps": summary["fps"],
                "pose_features_csv": str(directory / "pose_features.csv"),
                "rgb_metadata_csv": str(directory / "rgb_metadata.csv"),
                "rgb_embeddings_npy": str(directory / "rgb_embeddings.npy"),
                "sequence_summary_json": str(directory / "sequence_summary.json"),
            }
        )
    write_csv(
        OUTPUT_DIR / "sequence_feature_manifest.csv",
        manifest_rows,
        [
            "sequence_id",
            "frames",
            "fps",
            "pose_features_csv",
            "rgb_metadata_csv",
            "rgb_embeddings_npy",
            "sequence_summary_json",
        ],
    )

    total_frames = sum(int(item["frames"]) for item in summaries)
    total_pose = sum(int(item["pose_found_frames"]) for item in summaries)
    total_roi = sum(int(item["roi_crops"]) for item in summaries)
    summary = {
        "protocol": {
            "dataset": "GMDCSA24 v2.1 anonymous input",
            "role": "frozen final blind feature extraction",
            "model_inference_performed": True,
            "fall_prediction_or_alarm_decision_performed": False,
            "labels_annotations_private_mapping_or_raw_dataset_read": False,
            "inventory_fields_read": INVENTORY_FIELDS,
            "pose_person_selection": "highest detection confidence",
            "pose_confidence_floor": POSE_CONFIDENCE_FLOOR,
            "nms_iou": NMS_IOU,
            "image_size": IMAGE_SIZE,
            "roi_padding_ratio": ROI_PADDING_RATIO,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "resume_unit": "complete anonymous sequence committed atomically",
            "video_decoder": "system ffmpeg rawvideo pipe; audio disabled",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "ffmpeg": ffmpeg_version(),
            "gpu": torch.cuda.get_device_name(DEVICE),
        },
        "locked_artifacts_verified": locked_hashes,
        "sequences": len(summaries),
        "frames": total_frames,
        "duration_hours": sum(float(item["duration_seconds"]) for item in summaries) / 3600.0,
        "pose_found_frames": total_pose,
        "pose_detection_rate": divide(total_pose, total_frames),
        "roi_crops": total_roi,
        "full_image_fallbacks": total_frames - total_roi,
        "roi_crop_rate": divide(total_roi, total_frames),
        "embedding_shape_across_sequences": [total_frames, EMBEDDING_DIMENSION],
        "extractor_script_sha256": sha256(Path(__file__).resolve()),
    }
    (OUTPUT_DIR / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\nFeature extraction completed.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Outputs: {OUTPUT_DIR}", flush=True)


def divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
