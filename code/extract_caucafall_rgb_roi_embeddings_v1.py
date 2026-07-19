from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


POSE_FEATURE_DIR = Path("/home/data/yoloA27/features/caucafall_pose_v1")
MODEL_PATH = Path("/home/data/yoloA27/yolo26n-pose.pt")
OUTPUT_DIR = Path(
    "/home/data/yoloA27/features/caucafall_rgb_roi_embeddings_v1"
)
SPLITS = ("train", "val")
EXPECTED_ROWS = {"train": 11818, "val": 4070}
EXPECTED_SEQUENCES = {"train": 59, "val": 20}
EMBEDDING_DIMENSION = 256
IMAGE_SIZE = 640
DEVICE = 0
BATCH_SIZE = 16
CHUNK_SIZE = 64
ROI_PADDING_RATIO = 0.10

METADATA_FIELDS = [
    "split",
    "subject_id",
    "activity_name",
    "category",
    "sequence_id",
    "frame_index",
    "raw_frame_code",
    "image_name",
    "image_path",
    "frame_label",
    "pose_found",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "roi_crop_used",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot save empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rows(split: str, rows: list[dict[str, str]]) -> None:
    if len(rows) != EXPECTED_ROWS[split]:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS[split]} {split} rows, found {len(rows)}"
        )
    sequences: dict[str, list[int]] = {}
    image_names = []
    for row in rows:
        if row["split"] != split:
            raise ValueError(f"Unexpected row split: {row['split']}")
        path = Path(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")
        if path.name != row["image_name"]:
            raise ValueError(f"Image name/path mismatch: {path}")
        sequences.setdefault(row["sequence_id"], []).append(
            int(row["frame_index"])
        )
        image_names.append(row["image_name"])
    if len(sequences) != EXPECTED_SEQUENCES[split]:
        raise RuntimeError(
            f"Expected {EXPECTED_SEQUENCES[split]} {split} sequences, "
            f"found {len(sequences)}"
        )
    if len(image_names) != len(set(image_names)):
        raise RuntimeError(f"Duplicate image names in {split}")
    for sequence_id, indices in sequences.items():
        if indices != list(range(1, len(indices) + 1)):
            raise RuntimeError(f"Non-contiguous sequence: {sequence_id}")


def crop_person(row: dict[str, str]) -> tuple[np.ndarray, int]:
    image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {row['image_path']}")
    height, width = image.shape[:2]
    if int(row["pose_found"]) != 1:
        return image, 0

    x1 = float(row["bbox_x1"])
    y1 = float(row["bbox_y1"])
    x2 = float(row["bbox_x2"])
    y2 = float(row["bbox_y2"])
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return image, 0
    pad_x = (x2 - x1) * ROI_PADDING_RATIO
    pad_y = (y2 - y1) * ROI_PADDING_RATIO
    left = max(0, int(np.floor((x1 - pad_x) * width)))
    top = max(0, int(np.floor((y1 - pad_y) * height)))
    right = min(width, int(np.ceil((x2 + pad_x) * width)))
    bottom = min(height, int(np.ceil((y2 + pad_y) * height)))
    if right - left < 8 or bottom - top < 8:
        return image, 0
    return image[top:bottom, left:right].copy(), 1


def extract_split(split: str, model: YOLO) -> dict:
    rows = read_csv(POSE_FEATURE_DIR / f"pose_features_{split}.csv")
    validate_rows(split, rows)
    embeddings = np.lib.format.open_memmap(
        OUTPUT_DIR / f"embeddings_{split}.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), EMBEDDING_DIMENSION),
    )
    metadata = []
    roi_count = 0
    for start in range(0, len(rows), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(rows))
        crops = []
        flags = []
        for row in rows[start:end]:
            crop, crop_used = crop_person(row)
            crops.append(crop)
            flags.append(crop_used)

        outputs = model.embed(
            source=crops,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            batch=BATCH_SIZE,
            verbose=False,
        )
        if len(outputs) != len(crops):
            raise RuntimeError(
                f"Embedding count mismatch in {split}: {len(outputs)}/{len(crops)}"
            )
        array = np.stack(
            [output.detach().float().cpu().numpy() for output in outputs]
        ).astype(np.float32)
        if array.shape != (len(crops), EMBEDDING_DIMENSION):
            raise RuntimeError(f"Unexpected embedding shape: {array.shape}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"Non-finite embedding in {split} chunk {start}:{end}")
        embeddings[start:end] = array
        embeddings.flush()

        for row, crop_used in zip(rows[start:end], flags):
            item = {field: row.get(field, "") for field in METADATA_FIELDS[:-1]}
            item["roi_crop_used"] = crop_used
            metadata.append(item)
        roi_count += sum(flags)
        print(f"{split}: embedded {end}/{len(rows)} frames", flush=True)

    save_csv(OUTPUT_DIR / f"metadata_{split}.csv", metadata)
    norms = np.linalg.norm(np.asarray(embeddings), axis=1)
    result = {
        "split": split,
        "frames": len(rows),
        "sequences": len({row["sequence_id"] for row in rows}),
        "embedding_shape": [len(rows), EMBEDDING_DIMENSION],
        "roi_crops": roi_count,
        "full_image_fallbacks": len(rows) - roi_count,
        "roi_crop_rate": roi_count / len(rows),
        "embedding_norm": {
            "minimum": float(norms.min()),
            "maximum": float(norms.max()),
            "mean": float(norms.mean()),
            "standard_deviation": float(norms.std()),
        },
    }
    del embeddings
    return result


def main() -> None:
    required = [MODEL_PATH]
    required.extend(
        POSE_FEATURE_DIR / f"pose_features_{split}.csv" for split in SPLITS
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    print(f"Python: {platform.python_version()}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Ultralytics: {ultralytics.__version__}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}", flush=True)
    print(f"Embedding model: {MODEL_PATH}", flush=True)
    print("Protocol: CAUCAFall train/val only; person ROI with 10% padding", flush=True)

    model = YOLO(str(MODEL_PATH))
    summaries = {split: extract_split(split, model) for split in SPLITS}
    summary = {
        "protocol": {
            "task": "person-ROI RGB embedding extraction",
            "splits": list(SPLITS),
            "test_used": False,
            "external_data_used": False,
            "model": str(MODEL_PATH),
            "model_sha256": sha256(MODEL_PATH),
            "image_size": IMAGE_SIZE,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "roi_padding_ratio": ROI_PADDING_RATIO,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(DEVICE),
        },
        "splits": summaries,
    }
    (OUTPUT_DIR / "embedding_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Completed. Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
