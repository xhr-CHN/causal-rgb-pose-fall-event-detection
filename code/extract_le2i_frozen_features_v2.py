"""Extract label-isolated Le2i Pose and RGB-ROI features.

This script reads only label-free columns from the frozen Le2i inventory. It
does not open annotation files, event tables, or use any label in inference.
Each sequence is committed atomically so interrupted runs can resume safely.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from extract_caucafall_pose_features_v1 import missing_pose_values, pose_values


ROOT = Path("/home/data/yoloA27")
INVENTORY_DIR = ROOT / "experiments/le2i_inventory_v2"
INVENTORY_CSV = INVENTORY_DIR / "sequence_inventory.csv"
INVENTORY_SUMMARY = INVENTORY_DIR / "summary.json"
LOCK_FILE = (
    ROOT / "experiments/final_blind_protocol_lock_v1/locked_artifacts_sha256.txt"
)
MODEL_PATH = ROOT / "yolo26n-pose.pt"
POSE_HELPER_PATH = ROOT / "extract_caucafall_pose_features_v1.py"
OUTPUT_DIR = ROOT / "features/le2i_frozen_features_v1"
SEQUENCE_DIR = OUTPUT_DIR / "sequences"

EXPECTED_SEQUENCES = 190
EXPECTED_FRAMES = 75911
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

ALLOWED_INVENTORY_FIELDS = (
    "sequence_id",
    "scene",
    "video_path",
    "frame_count",
    "fps",
    "width",
    "height",
)

POSE_METADATA_FIELDS = [
    "sequence_id",
    "scene",
    "frame_number",
    "timestamp_ms",
    "timestamp_source",
    "image_name",
    "video_path",
]

RGB_METADATA_FIELDS = [
    "sequence_id",
    "scene",
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


class FFmpegVideoReader:
    """Decode only the first video stream and never initialize audio decoding."""

    def __init__(self, path: Path, width: int, height: int):
        self.path = path
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        self.process = None

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

    def read(self):
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("FFmpeg reader is not open")
        data = bytearray()
        while len(data) < self.frame_bytes:
            block = self.process.stdout.read(self.frame_bytes - len(data))
            if not block:
                break
            data.extend(block)
        if not data:
            return None
        if len(data) != self.frame_bytes:
            raise RuntimeError(
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
            raise RuntimeError(
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


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def load_label_free_inventory() -> list[dict[str, object]]:
    summary = json.loads(INVENTORY_SUMMARY.read_text(encoding="utf-8"))
    if summary["video_count"] != EXPECTED_SEQUENCES:
        raise RuntimeError("Frozen inventory sequence count changed")
    if summary["total_frames"] != EXPECTED_FRAMES:
        raise RuntimeError("Frozen inventory frame count changed")
    if summary["error_count"] != 0:
        raise RuntimeError("Frozen inventory contains errors")
    if summary["protocol"]["model_inference_performed"]:
        raise RuntimeError("Inventory unexpectedly reports prior inference")

    source_rows = read_csv(INVENTORY_CSV)
    rows = []
    for source in source_rows:
        # Copy only the explicitly approved, label-free fields. Event labels,
        # sequence type, onset, and end fields are intentionally not retained.
        missing = [name for name in ALLOWED_INVENTORY_FIELDS if name not in source]
        if missing:
            raise ValueError(f"Missing inventory fields: {missing}")
        row = {name: source[name] for name in ALLOWED_INVENTORY_FIELDS}
        row["frame_count"] = int(row["frame_count"])
        row["fps"] = float(row["fps"])
        row["width"] = int(row["width"])
        row["height"] = int(row["height"])
        row["video_path"] = str(row["video_path"])
        rows.append(row)

    rows.sort(key=lambda row: str(row["sequence_id"]))
    if len(rows) != EXPECTED_SEQUENCES:
        raise RuntimeError(f"Expected {EXPECTED_SEQUENCES} sequences, got {len(rows)}")
    if sum(int(row["frame_count"]) for row in rows) != EXPECTED_FRAMES:
        raise RuntimeError("Label-free inventory frame total mismatch")
    if len({str(row["sequence_id"]) for row in rows}) != len(rows):
        raise RuntimeError("Duplicate sequence identifier")
    for row in rows:
        path = Path(str(row["video_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if row["fps"] <= 0 or row["frame_count"] <= 0:
            raise ValueError(f"Invalid video metadata: {row}")
    return rows


def verify_locked_model() -> str:
    locked = {}
    for line in LOCK_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected_hash, filename = line.split(maxsplit=1)
        locked[str(Path(filename.strip()))] = expected_hash
    model_key = str(MODEL_PATH)
    if model_key not in locked:
        raise RuntimeError("Pose model is absent from the final artifact lock")
    actual_hash = sha256(MODEL_PATH)
    if actual_hash != locked[model_key]:
        raise RuntimeError("Locked pose model hash changed")
    return actual_hash


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
    summary_path = final_dir / "sequence_summary.json"
    pose_path = final_dir / "pose_features.csv"
    metadata_path = final_dir / "rgb_metadata.csv"
    embedding_path = final_dir / "rgb_embeddings.npy"
    if not all(path.is_file() for path in (summary_path, pose_path, metadata_path, embedding_path)):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        return None
    if int(summary.get("frames", -1)) != expected_frames:
        return None
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.shape != (expected_frames, EMBEDDING_DIMENSION):
        return None
    if count_csv_rows(pose_path) != expected_frames:
        return None
    if count_csv_rows(metadata_path) != expected_frames:
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
        raise RuntimeError(
            f"Incomplete non-atomic sequence directory requires review: {final_dir}"
        )
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
    pose_path = partial_dir / "pose_features.csv"
    rgb_metadata_path = partial_dir / "rgb_metadata.csv"
    pose_handle = pose_path.open("w", encoding="utf-8", newline="")
    rgb_handle = rgb_metadata_path.open("w", encoding="utf-8", newline="")
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
                    raise RuntimeError(
                        f"Decoded more than {expected_frames} frames in {sequence_id}"
                    )

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
                    raise RuntimeError(
                        f"Pose output mismatch in {sequence_id}: "
                        f"{len(pose_results)}/{len(frames)}"
                    )

                value_rows = [pose_values(result) for result in pose_results]
                crop_rows = [
                    crop_person(frame, values)
                    for frame, values in zip(frames, value_rows)
                ]
                crops = [item[0] for item in crop_rows]
                crop_flags = [item[1] for item in crop_rows]
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
                    raise RuntimeError(
                        f"Embedding output mismatch in {sequence_id}: "
                        f"{len(embed_outputs)}/{len(frames)}"
                    )
                embedding_array = np.stack(
                    [output.detach().float().cpu().numpy() for output in embed_outputs]
                ).astype(np.float32)
                if embedding_array.shape != (len(frames), EMBEDDING_DIMENSION):
                    raise RuntimeError(
                        f"Unexpected embedding shape in {sequence_id}: "
                        f"{embedding_array.shape}"
                    )
                if not np.isfinite(embedding_array).all():
                    raise RuntimeError(f"Non-finite embedding in {sequence_id}")

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
                        "scene": row["scene"],
                        "frame_number": frame_number,
                        "timestamp_ms": timestamp_ms,
                        "timestamp_source": "frame_index_and_video_fps",
                        "image_name": image_name,
                        "video_path": str(video_path),
                    }
                    pose_row.update(values)
                    pose_writer.writerow(pose_row)
                    rgb_writer.writerow({
                        "sequence_id": sequence_id,
                        "scene": row["scene"],
                        "frame_number": frame_number,
                        "timestamp_ms": timestamp_ms,
                        "timestamp_source": "frame_index_and_video_fps",
                        "image_name": image_name,
                        "video_path": str(video_path),
                        "pose_found": values["pose_found"],
                        "bbox_x1": values["bbox_x1"],
                        "bbox_y1": values["bbox_y1"],
                        "bbox_x2": values["bbox_x2"],
                        "bbox_y2": values["bbox_y2"],
                        "roi_crop_used": crop_used,
                    })

                pose_found_count += sum(
                    int(values["pose_found"]) for values in value_rows
                )
                roi_count += sum(crop_flags)
                frame_offset += len(frames)
                embeddings.flush()
                if (
                    frame_offset % REPORT_EVERY_FRAMES < SOURCE_CHUNK_SIZE
                    or frame_offset == expected_frames
                ):
                    pose_handle.flush()
                    rgb_handle.flush()
                    print(
                        f"{sequence_id}: {frame_offset}/{expected_frames} frames",
                        flush=True,
                    )
    finally:
        pose_handle.close()
        rgb_handle.close()
        embeddings.flush()
        del embeddings

    if frame_offset != expected_frames:
        raise RuntimeError(
            f"Frame count mismatch in {sequence_id}: "
            f"decoded={frame_offset}, expected={expected_frames}"
        )
    norm_mean = norm_sum / expected_frames
    norm_variance = max(0.0, norm_squared_sum / expected_frames - norm_mean**2)
    summary = {
        "status": "complete",
        "sequence_id": sequence_id,
        "scene": row["scene"],
        "video_path": str(video_path),
        "frames": expected_frames,
        "fps": fps,
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
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    os.replace(partial_dir, final_dir)
    print(f"{sequence_id}: committed", flush=True)
    return summary


def main() -> None:
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("System ffmpeg executable was not found in PATH")
    for path in (
        INVENTORY_CSV,
        INVENTORY_SUMMARY,
        LOCK_FILE,
        MODEL_PATH,
        POSE_HELPER_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    model_hash = verify_locked_model()
    rows = load_label_free_inventory()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Python: {platform.python_version()}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Ultralytics: {ultralytics.__version__}", flush=True)
    print(f"FFmpeg: {ffmpeg_version()}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}", flush=True)
    print(f"Locked model SHA256: {model_hash}", flush=True)
    print("Le2i labels available to extractor: NO", flush=True)
    print("Resume unit: one complete video", flush=True)

    pose_model = YOLO(str(MODEL_PATH))
    embed_model = YOLO(str(MODEL_PATH))
    summaries = []
    for index, row in enumerate(rows, start=1):
        print(
            f"\nSequence {index}/{len(rows)}: {row['sequence_id']}", flush=True
        )
        summaries.append(extract_sequence(row, pose_model, embed_model))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_rows = []
    for summary in summaries:
        sequence_id = str(summary["sequence_id"])
        directory = SEQUENCE_DIR / sequence_id
        manifest_rows.append({
            "sequence_id": sequence_id,
            "scene": summary["scene"],
            "frames": summary["frames"],
            "fps": summary["fps"],
            "pose_features_csv": str(directory / "pose_features.csv"),
            "rgb_metadata_csv": str(directory / "rgb_metadata.csv"),
            "rgb_embeddings_npy": str(directory / "rgb_embeddings.npy"),
            "sequence_summary_json": str(directory / "sequence_summary.json"),
        })
    write_csv(
        OUTPUT_DIR / "sequence_feature_manifest.csv",
        manifest_rows,
        list(manifest_rows[0]),
    )

    total_frames = sum(int(item["frames"]) for item in summaries)
    total_pose = sum(int(item["pose_found_frames"]) for item in summaries)
    total_roi = sum(int(item["roi_crops"]) for item in summaries)
    if total_frames != EXPECTED_FRAMES:
        raise RuntimeError(f"Final frame total mismatch: {total_frames}")
    summary = {
        "protocol": {
            "dataset": "Le2i",
            "role": "frozen final blind test feature extraction",
            "model_inference_performed": True,
            "prediction_or_alarm_decision_performed": False,
            "labels_or_annotations_read": False,
            "inventory_fields_read": list(ALLOWED_INVENTORY_FIELDS),
            "pose_person_selection": "highest detection confidence",
            "pose_confidence_floor": POSE_CONFIDENCE_FLOOR,
            "nms_iou": NMS_IOU,
            "image_size": IMAGE_SIZE,
            "roi_padding_ratio": ROI_PADDING_RATIO,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "resume_unit": "complete sequence committed atomically",
            "video_decoder": "system ffmpeg rawvideo pipe",
            "audio_decoding": "disabled (-an)",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "ffmpeg": ffmpeg_version(),
            "gpu": torch.cuda.get_device_name(DEVICE),
        },
        "locked_model": str(MODEL_PATH),
        "locked_model_sha256": model_hash,
        "pose_feature_helper": str(POSE_HELPER_PATH),
        "pose_feature_helper_sha256": sha256(POSE_HELPER_PATH),
        "extractor_script_sha256": sha256(Path(__file__).resolve()),
        "sequences": len(summaries),
        "frames": total_frames,
        "pose_found_frames": total_pose,
        "pose_detection_rate": total_pose / total_frames,
        "roi_crops": total_roi,
        "full_image_fallbacks": total_frames - total_roi,
        "roi_crop_rate": total_roi / total_frames,
        "embedding_shape_across_sequences": [total_frames, EMBEDDING_DIMENSION],
    }
    (OUTPUT_DIR / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nFeature extraction completed.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
