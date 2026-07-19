#!/usr/bin/env python3
"""Build post-hoc review clips for the frozen GMDCSA24 blind evaluation.

This script is descriptive only. It does not run a model, change any policy,
or tune a threshold. It extracts:
  * every fall event missed by the final_causal_rescue stage; and
  * every false alarm emitted by the final_causal_rescue stage.

Expected project root: /home/data/yoloA27
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/data/yoloA27")
RAW_VIDEO_ROOT = ROOT / "GMDCSA24"
EVALUATION_DIR = ROOT / "experiments/gmdcsa24_final_blind_evaluation_v1"
OUTPUT_DIR = ROOT / "experiments/gmdcsa24_failure_review_v1"

EVENT_RESULTS = EVALUATION_DIR / "event_results.csv"
ALARM_CLASSIFICATION = EVALUATION_DIR / "alarm_classification.csv"
ANNOTATION_AUDIT = EVALUATION_DIR / "annotation_audit.csv"
EVALUATION_LOCK = EVALUATION_DIR / "evaluation_complete.lock"

FINAL_STAGE = "final_causal_rescue"
SECONDS_BEFORE_FOCUS = 2.0
SECONDS_AFTER_FOCUS = 3.0


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        fail(f"Refusing to write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def as_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_token(value: str) -> str:
    cleaned = []
    for character in value.strip().lower():
        if character.isalnum():
            cleaned.append(character)
        elif not cleaned or cleaned[-1] != "_":
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "unknown"


def select_encoder() -> tuple[str, list[str]]:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if "libx264" in result.stdout:
        return "libx264", ["-preset", "veryfast", "-crf", "23"]
    return "mpeg4", ["-q:v", "3"]


def extract_clip(
    source: Path,
    destination: Path,
    start_seconds: float,
    duration_seconds: float,
    encoder: str,
    encoder_arguments: list[str],
) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration_seconds:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "scale='min(1280,iw)':-2:force_original_aspect_ratio=decrease,format=yuv420p",
        "-c:v",
        encoder,
        *encoder_arguments,
        "-movflags",
        "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)


def make_case(
    *,
    case_number: int,
    case_type: str,
    sequence_id: str,
    focus_timestamp_ms: float,
    audit: dict[str, str],
    miss_stage: str = "",
    alarm_classification: str = "",
    candidate_key: str = "",
) -> dict[str, object]:
    source_relative_path = audit["source_relative_path"]
    source = RAW_VIDEO_ROOT / source_relative_path
    if not source.is_file():
        fail(f"Source video not found: {source}")

    duration_seconds = as_float(audit.get("actual_duration_seconds"))
    focus_seconds = max(0.0, focus_timestamp_ms / 1000.0)
    clip_start_seconds = max(0.0, focus_seconds - SECONDS_BEFORE_FOCUS)
    clip_end_seconds = min(duration_seconds, focus_seconds + SECONDS_AFTER_FOCUS)
    clip_duration_seconds = clip_end_seconds - clip_start_seconds
    if clip_duration_seconds <= 0:
        fail(f"Invalid clip interval for {sequence_id}")

    subject_token = safe_token(audit.get("subject", ""))
    source_stem = safe_token(Path(source_relative_path).stem)
    type_token = safe_token(miss_stage or alarm_classification or case_type)
    filename = (
        f"{case_type.upper()}_{case_number:03d}_"
        f"{subject_token}_{source_stem}_{type_token}.mp4"
    )

    return {
        "case_id": f"{case_type.upper()}_{case_number:03d}",
        "case_type": case_type,
        "miss_stage": miss_stage,
        "alarm_classification": alarm_classification,
        "candidate_key": candidate_key,
        "sequence_id": sequence_id,
        "subject": audit.get("subject", ""),
        "category": audit.get("category", ""),
        "fall_type": audit.get("fall_type", ""),
        "source_relative_path": source_relative_path,
        "clip_file": filename,
        "clip_start_ms": round(clip_start_seconds * 1000.0, 3),
        "clip_end_ms": round(clip_end_seconds * 1000.0, 3),
        "focus_timestamp_ms": round(focus_timestamp_ms, 3),
        "fall_onset_ms": round(as_float(audit.get("fall_onset_seconds")) * 1000.0, 3),
        "time_of_recording": audit.get("time_of_recording", ""),
        "description": audit.get("description", ""),
        "source_path": str(source),
        "clip_path": str(OUTPUT_DIR / case_type / filename),
    }


def main() -> None:
    required = [
        RAW_VIDEO_ROOT,
        EVENT_RESULTS,
        ALARM_CLASSIFICATION,
        ANNOTATION_AUDIT,
        EVALUATION_LOCK,
    ]
    for path in required:
        if not path.exists():
            fail(f"Required input missing: {path}")
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg is not available in PATH")
    if OUTPUT_DIR.exists():
        fail(f"Output already exists; refusing overwrite: {OUTPUT_DIR}")

    audit_rows = read_csv(ANNOTATION_AUDIT)
    event_rows = read_csv(EVENT_RESULTS)
    alarm_rows = read_csv(ALARM_CLASSIFICATION)
    audit_by_sequence = {row["sequence_id"]: row for row in audit_rows}

    base_detection = {
        row["sequence_id"]: as_int(row.get("detected"))
        for row in event_rows
        if row.get("stage") == "base_candidate"
    }

    final_misses = sorted(
        (
            row
            for row in event_rows
            if row.get("stage") == FINAL_STAGE
            and row.get("category") == "Fall"
            and as_int(row.get("detected")) == 0
        ),
        key=lambda row: (row.get("subject", ""), as_float(row.get("fall_onset_ms"))),
    )
    final_false_alarms = sorted(
        (
            row
            for row in alarm_rows
            if row.get("stage") == FINAL_STAGE
            and row.get("classification")
            in {"false_alarm_adl", "false_alarm_pre_onset"}
        ),
        key=lambda row: (row.get("subject", ""), row.get("sequence_id", "")),
    )

    if len(final_misses) != 19:
        fail(f"Expected 19 final misses, found {len(final_misses)}")
    if len(final_false_alarms) != 14:
        fail(f"Expected 14 final false alarms, found {len(final_false_alarms)}")

    cases: list[dict[str, object]] = []
    for number, row in enumerate(final_misses, start=1):
        sequence_id = row["sequence_id"]
        if sequence_id not in audit_by_sequence:
            fail(f"No annotation audit row for {sequence_id}")
        miss_stage = (
            "base_no_candidate"
            if base_detection.get(sequence_id, 0) == 0
            else "verifier_rejected_not_rescued"
        )
        cases.append(
            make_case(
                case_number=number,
                case_type="miss",
                sequence_id=sequence_id,
                focus_timestamp_ms=as_float(row.get("fall_onset_ms")),
                audit=audit_by_sequence[sequence_id],
                miss_stage=miss_stage,
            )
        )

    for number, row in enumerate(final_false_alarms, start=1):
        sequence_id = row["sequence_id"]
        if sequence_id not in audit_by_sequence:
            fail(f"No annotation audit row for {sequence_id}")
        cases.append(
            make_case(
                case_number=number,
                case_type="false_alarm",
                sequence_id=sequence_id,
                focus_timestamp_ms=as_float(row.get("alarm_timestamp_ms")),
                audit=audit_by_sequence[sequence_id],
                alarm_classification=row.get("classification", ""),
                candidate_key=row.get("candidate_key", ""),
            )
        )

    (OUTPUT_DIR / "miss").mkdir(parents=True)
    (OUTPUT_DIR / "false_alarm").mkdir(parents=True)
    encoder, encoder_arguments = select_encoder()

    for index, case in enumerate(cases, start=1):
        print(
            f"[{index:02d}/{len(cases)}] {case['case_id']} "
            f"{case['source_relative_path']}",
            flush=True,
        )
        extract_clip(
            source=Path(str(case["source_path"])),
            destination=Path(str(case["clip_path"])),
            start_seconds=float(case["clip_start_ms"]) / 1000.0,
            duration_seconds=(
                float(case["clip_end_ms"]) - float(case["clip_start_ms"])
            )
            / 1000.0,
            encoder=encoder,
            encoder_arguments=encoder_arguments,
        )

    public_rows = [
        {key: value for key, value in case.items() if key not in {"source_path", "clip_path"}}
        for case in cases
    ]
    write_csv(OUTPUT_DIR / "failure_review_manifest.csv", public_rows)

    miss_stage_counts = Counter(str(case["miss_stage"]) for case in cases if case["case_type"] == "miss")
    false_alarm_counts = Counter(
        str(case["alarm_classification"])
        for case in cases
        if case["case_type"] == "false_alarm"
    )
    summary = {
        "status": "post_hoc_descriptive_failure_review",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_inference_performed": False,
        "threshold_or_policy_changed": False,
        "final_stage": FINAL_STAGE,
        "miss_clip_count": len(final_misses),
        "false_alarm_clip_count": len(final_false_alarms),
        "total_clip_count": len(cases),
        "miss_stage_counts": dict(miss_stage_counts),
        "false_alarm_classification_counts": dict(false_alarm_counts),
        "clip_window_seconds": {
            "before_focus": SECONDS_BEFORE_FOCUS,
            "after_focus": SECONDS_AFTER_FOCUS,
        },
        "video_encoder": encoder,
        "evaluation_complete_lock_sha256": sha256(EVALUATION_LOCK),
        "event_results_sha256": sha256(EVENT_RESULTS),
        "alarm_classification_sha256": sha256(ALARM_CLASSIFICATION),
        "annotation_audit_sha256": sha256(ANNOTATION_AUDIT),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    output_files = sorted(path for path in OUTPUT_DIR.rglob("*") if path.is_file())
    with (OUTPUT_DIR / "output_sha256.txt").open("w", encoding="utf-8") as handle:
        for path in output_files:
            handle.write(f"{sha256(path)}  {path.relative_to(OUTPUT_DIR)}\n")
    (OUTPUT_DIR / "review_complete.lock").write_text(
        "Post-hoc GMDCSA24 failure review clips completed. "
        "Do not use this blind test to tune and rerun the frozen protocol.\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Completed: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
