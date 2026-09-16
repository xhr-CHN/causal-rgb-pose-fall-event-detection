#!/usr/bin/env python3
"""Extract frozen Pose/RGB-ROI features for the prepared OOPS test subset."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import torch
from ultralytics import YOLO

import extract_gmdcsa24_frozen_features_v1 as core


ROOT = Path("/home/data/yoloA27")
OOPS_ROOT = Path("/home/data/omnifall_benchmark_2026_09/raw/oops")
VIDEO_DIR = OOPS_ROOT / "oops_test_videos"
TEST_CSV = OOPS_ROOT / "test.csv"
OUTPUT_DIR = ROOT / "features/oops_test_frozen_features_v1"
SEQUENCE_DIR = OUTPUT_DIR / "sequences"
MODEL_PATH = ROOT / "yolo26n-pose.pt"
EXPECTED_VIDEOS = 572


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", Path(value).stem.lower())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_test_paths() -> dict[str, str]:
    with TEST_CSV.open(encoding="utf-8", newline="") as f:
        paths = [row["path"] for row in csv.DictReader(f)]
    mapping = {norm(path): path for path in paths}
    if len(mapping) != EXPECTED_VIDEOS:
        raise RuntimeError(f"Expected {EXPECTED_VIDEOS} unique test paths, found {len(mapping)}")
    return mapping


def main() -> None:
    if not VIDEO_DIR.is_dir() or not TEST_CSV.is_file():
        raise FileNotFoundError("OOPS test videos or test.csv is missing")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OOPS feature extraction")
    if OUTPUT_DIR.exists() and (OUTPUT_DIR / "sequence_feature_manifest.csv").exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {OUTPUT_DIR}")

    wanted = read_test_paths()
    videos = {}
    for path in VIDEO_DIR.glob("*.mp4"):
        key = norm(path.name)
        if key in wanted:
            if key in videos:
                raise RuntimeError(f"Duplicate normalized video name: {path.name}")
            videos[key] = path
    if set(videos) != set(wanted):
        raise RuntimeError(
            f"Video/test mismatch: videos={len(videos)}, test_paths={len(wanted)}, "
            f"missing={len(set(wanted)-set(videos))}"
        )

    core.ROOT = ROOT
    core.MODEL_PATH = MODEL_PATH
    core.OUTPUT_DIR = OUTPUT_DIR
    core.SEQUENCE_DIR = SEQUENCE_DIR
    core.EXPECTED_SEQUENCES = EXPECTED_VIDEOS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"Preparing metadata for {EXPECTED_VIDEOS} OOPS videos", flush=True)
    for index, key in enumerate(sorted(videos), 1):
        path = videos[key]
        metadata = core.video_metadata(path)
        rows.append(
            {
                "sequence_id": wanted[key],
                "video_file": path.name,
                "video_path": str(path),
                "sha256": sha256(path),
                **metadata,
            }
        )
        if index % 50 == 0 or index == EXPECTED_VIDEOS:
            print(f"Metadata: {index}/{EXPECTED_VIDEOS}", flush=True)

    pose_model = YOLO(str(MODEL_PATH))
    embed_model = YOLO(str(MODEL_PATH))
    summaries = []
    for index, row in enumerate(rows, 1):
        print(f"Sequence {index}/{EXPECTED_VIDEOS}: {row['sequence_id']}", flush=True)
        summaries.append(core.extract_sequence(row, pose_model, embed_model))
        torch.cuda.empty_cache()

    manifest = []
    for summary in summaries:
        sequence_id = str(summary["sequence_id"])
        directory = SEQUENCE_DIR / sequence_id
        manifest.append(
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

    fields = [
        "sequence_id", "frames", "fps", "pose_features_csv",
        "rgb_metadata_csv", "rgb_embeddings_npy", "sequence_summary_json",
    ]
    with (OUTPUT_DIR / "sequence_feature_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "dataset": "OOPS test subset",
        "videos": len(manifest),
        "frames": sum(int(item["frames"]) for item in manifest),
        "model": str(MODEL_PATH),
        "model_task": "pose",
        "labels_read_by_extractor": False,
        "output": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
