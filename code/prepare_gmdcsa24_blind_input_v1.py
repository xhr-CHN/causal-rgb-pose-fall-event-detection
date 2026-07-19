#!/usr/bin/env python3
"""Verify and freeze GMDCSA24 as an anonymized, label-isolated blind input.

This script performs no model inference and does not parse the dataset CSV files.
It verifies every raw file against the Windows SHA-256 manifest, creates opaque
video names using hard links (copy fallback), and stores the private mapping
outside the blind-input directory for evaluation only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path("/home/data/yoloA27")
RAW_ROOT = ROOT / "GMDCSA24"
RAW_MANIFEST = ROOT / "gmdcsa24_raw_sha256.txt"
BLIND_ROOT = ROOT / "GMDCSA24_FROZEN_INPUT_V1"
BLIND_VIDEO_DIR = BLIND_ROOT / "videos"
LOCK_DIR = ROOT / "experiments/gmdcsa24_final_blind_lock_v1"

EXPECTED_MANIFEST_SHA256 = (
    "2a78dd9bf063f37a06cf46990a92598c9ef248fd422e0d542c1be1203de6bc1d"
)
EXPECTED_MANIFEST_ENTRIES = 170
EXPECTED_VIDEO_COUNT = 160
EXPECTED_CSV_COUNT = 8
EXPECTED_CATEGORY_COUNTS = {"ADL": 81, "Fall": 79}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_manifest(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\r")
        if not line.strip():
            continue
        if len(line) < 67 or line[64:66] != "  ":
            fail(f"Manifest line {line_number} is not '<sha256>  <relative path>'")

        expected_hash = line[:64].lower()
        relative = line[66:].replace("\\", "/")
        if any(character not in "0123456789abcdef" for character in expected_hash):
            fail(f"Manifest line {line_number} has an invalid SHA-256 value")

        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            fail(f"Manifest line {line_number} has an unsafe path: {relative}")
        normalized = pure.as_posix()
        if normalized in seen:
            fail(f"Duplicate manifest path: {normalized}")
        seen.add(normalized)
        rows.append((expected_hash, normalized))

    return rows


def parse_video_identity(relative: str) -> tuple[str, str]:
    parts = PurePosixPath(relative).parts
    if len(parts) != 3:
        fail(f"Unexpected video path layout: {relative}")
    subject, category, _ = parts
    if not subject.startswith("Subject ") or category not in {"ADL", "Fall"}:
        fail(f"Unexpected video identity: {relative}")
    return subject, category


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for required in (RAW_ROOT, RAW_MANIFEST):
        if not required.exists():
            fail(f"Required path does not exist: {required}")
    if BLIND_ROOT.exists():
        fail(f"Blind-input directory already exists; refusing to overwrite: {BLIND_ROOT}")
    if LOCK_DIR.exists():
        fail(f"Lock directory already exists; refusing to overwrite: {LOCK_DIR}")

    manifest_hash = sha256_file(RAW_MANIFEST)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        fail(
            "Raw manifest SHA-256 mismatch: "
            f"expected {EXPECTED_MANIFEST_SHA256}, got {manifest_hash}"
        )

    manifest_rows = read_manifest(RAW_MANIFEST)
    if len(manifest_rows) != EXPECTED_MANIFEST_ENTRIES:
        fail(
            f"Expected {EXPECTED_MANIFEST_ENTRIES} manifest entries, "
            f"found {len(manifest_rows)}"
        )

    manifest_paths = {relative for _, relative in manifest_rows}
    actual_paths = {
        path.relative_to(RAW_ROOT).as_posix()
        for path in RAW_ROOT.rglob("*")
        if path.is_file()
    }
    if manifest_paths != actual_paths:
        missing = sorted(manifest_paths - actual_paths)
        extra = sorted(actual_paths - manifest_paths)
        fail(f"Dataset/manifest file-set mismatch; missing={missing[:5]}, extra={extra[:5]}")

    verified: list[tuple[str, str]] = []
    for index, (expected_hash, relative) in enumerate(manifest_rows, start=1):
        actual_hash = sha256_file(RAW_ROOT / Path(relative))
        if actual_hash != expected_hash:
            fail(f"File hash mismatch: {relative}")
        verified.append((expected_hash, relative))
        if index % 20 == 0 or index == len(manifest_rows):
            print(f"Verified raw files: {index}/{len(manifest_rows)}", flush=True)

    video_rows = [
        (digest, relative)
        for digest, relative in verified
        if Path(relative).suffix.lower() in VIDEO_SUFFIXES
    ]
    csv_count = sum(Path(relative).suffix.lower() == ".csv" for _, relative in verified)
    if len(video_rows) != EXPECTED_VIDEO_COUNT:
        fail(f"Expected {EXPECTED_VIDEO_COUNT} videos, found {len(video_rows)}")
    if csv_count != EXPECTED_CSV_COUNT:
        fail(f"Expected {EXPECTED_CSV_COUNT} CSV files, found {csv_count}")

    category_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    opaque_ids: set[str] = set()
    private_rows: list[dict[str, object]] = []
    blind_rows: list[dict[str, object]] = []
    link_counts: Counter[str] = Counter()

    BLIND_VIDEO_DIR.mkdir(parents=True)
    LOCK_DIR.mkdir(parents=True)

    try:
        for digest, relative in sorted(video_rows, key=lambda item: item[0]):
            source = RAW_ROOT / Path(relative)
            subject, category = parse_video_identity(relative)
            category_counts[category] += 1
            subject_counts[subject] += 1

            opaque_id = f"gmdcsa24_{digest[:16]}"
            if opaque_id in opaque_ids:
                fail(f"Opaque ID collision: {opaque_id}")
            opaque_ids.add(opaque_id)

            suffix = source.suffix.lower()
            destination = BLIND_VIDEO_DIR / f"{opaque_id}{suffix}"
            try:
                os.link(source, destination)
                link_method = "hardlink"
            except OSError:
                shutil.copy2(source, destination)
                link_method = "copy"
            link_counts[link_method] += 1

            if sha256_file(destination) != digest:
                fail(f"Blind-input hash mismatch after linking/copying: {opaque_id}")

            blind_rows.append(
                {
                    "sequence_id": opaque_id,
                    "video_file": f"videos/{destination.name}",
                    "sha256": digest,
                }
            )
            private_rows.append(
                {
                    "sequence_id": opaque_id,
                    "source_relative_path": relative,
                    "subject": subject,
                    "category": category,
                    "source_sha256": digest,
                    "blind_video_file": f"videos/{destination.name}",
                    "link_method": link_method,
                }
            )

        if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
            fail(
                f"Category counts differ: expected {EXPECTED_CATEGORY_COUNTS}, "
                f"found {dict(category_counts)}"
            )
        if len(subject_counts) != 4:
            fail(f"Expected 4 subjects, found {dict(subject_counts)}")

        write_csv(
            BLIND_ROOT / "blind_video_inventory.csv",
            ["sequence_id", "video_file", "sha256"],
            blind_rows,
        )
        write_csv(
            LOCK_DIR / "PRIVATE_EVALUATION_MAPPING_DO_NOT_USE_FOR_INFERENCE.csv",
            [
                "sequence_id",
                "source_relative_path",
                "subject",
                "category",
                "source_sha256",
                "blind_video_file",
                "link_method",
            ],
            private_rows,
        )

        policy = """GMDCSA24 FINAL BLIND PROTOCOL V1

Dataset role: frozen final blind test for the next method version.
Raw dataset integrity: verified against the Windows SHA-256 manifest.
Inference input: GMDCSA24_FROZEN_INPUT_V1 only.
Inference-visible labels: none.
Private mapping and official CSV files: evaluation only; do not read before frozen inference completes.
Model inference performed during this preparation step: NO.
Training, calibration, threshold selection, or verifier fitting on GMDCSA24: PROHIBITED.
Model/checkpoint/policy lock: must be completed before the first GMDCSA24 inference.
Final evaluation: run once after predictions are cryptographically locked.
"""
        (LOCK_DIR / "locked_protocol.txt").write_text(policy, encoding="utf-8")

        script_hash = sha256_file(Path(__file__).resolve())
        summary = {
            "protocol": {
                "dataset": "GMDCSA24 v2.1",
                "role": "frozen final blind test for next method version",
                "model_inference_performed": False,
                "labels_used_for_training_or_threshold_selection": False,
                "inference_input": str(BLIND_ROOT),
                "private_mapping": str(
                    LOCK_DIR / "PRIVATE_EVALUATION_MAPPING_DO_NOT_USE_FOR_INFERENCE.csv"
                ),
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "raw_manifest": str(RAW_MANIFEST),
            "raw_manifest_sha256": manifest_hash,
            "raw_manifest_entries": len(manifest_rows),
            "verified_raw_files": len(verified),
            "video_count": len(video_rows),
            "csv_count": csv_count,
            "subject_count": len(subject_counts),
            "private_category_counts": dict(sorted(category_counts.items())),
            "private_subject_counts": dict(sorted(subject_counts.items())),
            "blind_input_contains_category_or_subject_columns": False,
            "link_method_counts": dict(sorted(link_counts.items())),
            "preparation_script_sha256": script_hash,
            "locked_protocol_sha256": sha256_file(LOCK_DIR / "locked_protocol.txt"),
        }
        (LOCK_DIR / "dataset_lock_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    except Exception:
        shutil.rmtree(BLIND_ROOT, ignore_errors=True)
        shutil.rmtree(LOCK_DIR, ignore_errors=True)
        raise

    print("\nGMDCSA24 blind input prepared successfully.", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Blind input: {BLIND_ROOT}", flush=True)
    print(f"Private lock: {LOCK_DIR}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
