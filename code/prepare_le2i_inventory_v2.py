"""Prepare the frozen Le2i blind-test V2 inventory without model inference.

The official files contain two valid layouts: the two single-integer event
lines can be at the start or embedded among six-column frame rows. A 0,0 event
pair denotes an annotated ADL sequence. Repeated frame rows are normalized by
frame number; official event lines, not box rows, define fall events.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil

import cv2


ROOT = Path("/home/data/yoloA27")
LE2I_ROOT = ROOT / "Le2i"
MANIFEST = ROOT / "le2i_file_manifest_sha256.txt"
LOCK_DIR = ROOT / "experiments/final_blind_protocol_lock_v1"
INTEGRITY_LOG = LOCK_DIR / "le2i_integrity_check.txt"
ARTIFACT_LOCK = LOCK_DIR / "locked_artifacts_sha256.txt"
POLICY_LOCK = LOCK_DIR / "locked_policy.txt"
OUTPUT_DIR = ROOT / "experiments/le2i_inventory_v2"

EXPECTED_VIDEO_COUNT = 190
EXPECTED_ANNOTATION_COUNT = 130
EXPECTED_FALL_COUNT = 99
EXPECTED_ANNOTATED_ADL_COUNT = 31
EXPECTED_UNANNOTATED_ADL_COUNT = 60
EXPECTED_SCENES = {
    "Coffee_room_01",
    "Coffee_room_02",
    "Home_01",
    "Home_02",
    "Lecture_room",
    "Office",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalized_key(path):
    relative = path.relative_to(LE2I_ROOT)
    scene = relative.parts[0]
    return scene.casefold(), path.stem.casefold()


def sequence_id(path):
    scene = path.relative_to(LE2I_ROOT).parts[0]
    raw = f"{scene}_{path.stem}"
    return re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()


def parse_annotation(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if line.strip()
    ]
    event_values = []
    event_line_numbers = []
    detail_rows = []
    for line_number, line in enumerate(lines, start=1):
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 1:
            try:
                event_values.append(int(fields[0]))
            except ValueError as error:
                raise ValueError(
                    f"line {line_number}: invalid single-integer event value"
                ) from error
            event_line_numbers.append(line_number)
            continue
        if len(fields) != 6:
            raise ValueError(
                f"line {line_number}: expected 1 or 6 comma-separated integers, "
                f"got {len(fields)}"
            )
        try:
            values = [int(item) for item in fields]
        except ValueError as error:
            raise ValueError(f"line {line_number}: non-integer value") from error
        detail_rows.append(values)
    if len(event_values) != 2:
        raise ValueError(
            f"expected exactly two single-integer event lines, got {len(event_values)}"
        )
    return (
        event_values[0],
        event_values[1],
        detail_rows,
        event_line_numbers,
    )


def read_video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("OpenCV could not open video")
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    if frame_count <= 0:
        raise ValueError(f"invalid frame count: {frame_count}")
    if fps <= 0:
        raise ValueError(f"invalid FPS: {fps}")
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame size: {width}x{height}")
    return frame_count, fps, width, height


def verify_lock_files():
    required = [MANIFEST, INTEGRITY_LOG, ARTIFACT_LOCK, POLICY_LOCK]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest_lines = [
        line for line in MANIFEST.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    integrity_lines = INTEGRITY_LOG.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    ok_count = sum(line.rstrip().endswith(": OK") for line in integrity_lines)
    failed_count = sum("FAILED" in line for line in integrity_lines)
    if failed_count or ok_count != len(manifest_lines):
        raise RuntimeError(
            "Le2i integrity lock is not valid: "
            f"manifest={len(manifest_lines)}, OK={ok_count}, FAILED={failed_count}"
        )

    locked_rows = []
    for line_number, line in enumerate(
        ARTIFACT_LOCK.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid artifact lock line {line_number}")
        expected_hash, filename = parts
        path = Path(filename.strip())
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"locked artifact changed: {path}")
        locked_rows.append({
            "path": str(path),
            "sha256": actual_hash,
        })
    if len(locked_rows) != 5:
        raise RuntimeError(f"expected 5 locked artifacts, found {len(locked_rows)}")

    return {
        "manifest_entries": len(manifest_lines),
        "integrity_ok_entries": ok_count,
        "integrity_failed_entries": failed_count,
        "locked_artifacts": locked_rows,
        "locked_policy_sha256": sha256(POLICY_LOCK),
    }


def add_issue(issues, severity, code, sequence, path, details):
    issues.append({
        "severity": severity,
        "code": code,
        "sequence_id": sequence,
        "path": str(path),
        "details": details,
    })


def main():
    if not LE2I_ROOT.is_dir():
        raise FileNotFoundError(LE2I_ROOT)
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")

    lock_summary = verify_lock_files()
    videos = sorted(
        path for path in LE2I_ROOT.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".avi"
    )
    annotations = sorted(
        path for path in LE2I_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.casefold() == ".txt"
        and path.name.casefold() != "readme.txt"
    )

    annotation_by_key = {}
    duplicate_annotation_keys = []
    for annotation in annotations:
        key = normalized_key(annotation)
        if key in annotation_by_key:
            duplicate_annotation_keys.append((key, annotation_by_key[key], annotation))
        annotation_by_key[key] = annotation

    issues = []
    for key, first, second in duplicate_annotation_keys:
        add_issue(
            issues, "error", "duplicate_annotation_key", "", second,
            f"key={key}; first={first}",
        )

    inventory_rows = []
    event_rows = []
    used_annotation_keys = set()

    for index, video in enumerate(videos, start=1):
        sid = sequence_id(video)
        relative = video.relative_to(LE2I_ROOT)
        scene = relative.parts[0]
        key = normalized_key(video)
        annotation = annotation_by_key.get(key)
        if annotation is not None:
            used_annotation_keys.add(key)

        frame_count = 0
        fps = 0.0
        width = 0
        height = 0
        try:
            frame_count, fps, width, height = read_video_metadata(video)
        except Exception as error:
            add_issue(issues, "error", "video_metadata_error", sid, video, str(error))

        official_start = None
        official_end = None
        has_fall = False
        detail_rows = []
        unique_detail_rows = []
        event_line_numbers = []
        duplicate_detail_rows = 0
        conflicting_duplicate_rows = 0
        state_codes = []
        detail_first_frame = None
        detail_last_frame = None
        detail_contiguous = None
        if annotation is not None:
            try:
                (
                    official_start,
                    official_end,
                    detail_rows,
                    event_line_numbers,
                ) = parse_annotation(annotation)

                detail_by_frame = {}
                for detail_row in detail_rows:
                    frame_number = detail_row[0]
                    if frame_number in detail_by_frame:
                        duplicate_detail_rows += 1
                        if detail_by_frame[frame_number] != detail_row:
                            conflicting_duplicate_rows += 1
                    detail_by_frame[frame_number] = detail_row
                unique_detail_rows = [
                    detail_by_frame[frame_number]
                    for frame_number in sorted(detail_by_frame)
                ]
                detail_frames = [row[0] for row in unique_detail_rows]
                state_codes = sorted(set(row[1] for row in unique_detail_rows))
                if detail_frames:
                    detail_first_frame = detail_frames[0]
                    detail_last_frame = detail_frames[-1]
                    detail_contiguous = detail_frames == list(
                        range(detail_frames[0], detail_frames[-1] + 1)
                    )
                    if not detail_contiguous:
                        add_issue(
                            issues, "error", "non_contiguous_annotation_frames",
                            sid, annotation,
                            "unique detail frame numbers are not contiguous",
                        )
                if official_start == 0 and official_end == 0:
                    has_fall = False
                elif (
                    official_start >= 1
                    and official_end >= official_start
                    and (not frame_count or official_end <= frame_count)
                ):
                    has_fall = True
                else:
                    add_issue(
                        issues, "error", "invalid_fall_interval", sid, annotation,
                        f"start={official_start}, end={official_end}, "
                        f"video_frames={frame_count}",
                    )
                if frame_count and len(unique_detail_rows) != frame_count:
                    add_issue(
                        issues, "error", "annotation_video_length_mismatch",
                        sid, annotation,
                        f"unique_annotation_frames={len(unique_detail_rows)}, "
                        f"video_frames={frame_count}",
                    )
                if conflicting_duplicate_rows:
                    add_issue(
                        issues, "warning", "conflicting_duplicate_box_rows",
                        sid, annotation,
                        f"conflicting_duplicate_rows={conflicting_duplicate_rows}; "
                        "official event interval remains usable",
                    )
            except Exception as error:
                add_issue(issues, "error", "annotation_parse_error", sid, annotation, str(error))

        negative_frames = (
            max(0, official_start - 1)
            if has_fall and official_start is not None
            else frame_count
        )
        if has_fall:
            sequence_type = "fall"
        elif annotation is not None:
            sequence_type = "adl_annotated"
        else:
            sequence_type = "adl_unannotated"
        inventory_rows.append({
            "sequence_index": index,
            "sequence_id": sid,
            "scene": scene,
            "sequence_type": sequence_type,
            "video_path": str(video),
            "annotation_path": str(annotation) if annotation else "",
            "annotation_present": int(annotation is not None),
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_seconds": frame_count / fps if fps else 0.0,
            "official_event_start_raw": (
                official_start if official_start is not None else ""
            ),
            "official_event_end_raw": (
                official_end if official_end is not None else ""
            ),
            "fall_onset_frame": official_start if has_fall else "",
            "fall_end_frame": official_end if has_fall else "",
            "negative_exposure_frames": negative_frames,
            "negative_exposure_seconds": negative_frames / fps if fps else 0.0,
            "annotation_event_line_numbers": "|".join(map(str, event_line_numbers)),
            "annotation_event_lines_at_start": int(event_line_numbers == [1, 2]),
            "annotation_detail_rows_raw": len(detail_rows),
            "annotation_unique_frames": len(unique_detail_rows),
            "annotation_duplicate_rows": duplicate_detail_rows,
            "annotation_conflicting_duplicate_rows": conflicting_duplicate_rows,
            "annotation_first_frame": detail_first_frame if detail_first_frame is not None else "",
            "annotation_last_frame": detail_last_frame if detail_last_frame is not None else "",
            "annotation_frames_contiguous": (
                int(detail_contiguous) if detail_contiguous is not None else ""
            ),
            "annotation_state_codes": "|".join(map(str, state_codes)),
        })
        if has_fall:
            event_rows.append({
                "sequence_id": sid,
                "scene": scene,
                "event_id": f"{sid}_fall_1",
                "onset_frame": official_start,
                "end_frame": official_end,
                "onset_ms": (official_start - 1) * 1000.0 / fps if fps else "",
                "end_ms": (official_end - 1) * 1000.0 / fps if fps else "",
                "evaluation_rule": "detect once at or after onset; post-onset duplicate alarms suppressed",
            })

    for key, annotation in annotation_by_key.items():
        if key not in used_annotation_keys:
            add_issue(
                issues, "error", "annotation_without_video", "", annotation,
                f"unmatched key={key}",
            )

    observed_scenes = {row["scene"] for row in inventory_rows}
    if len(videos) != EXPECTED_VIDEO_COUNT:
        add_issue(
            issues, "error", "unexpected_video_count", "", LE2I_ROOT,
            f"expected={EXPECTED_VIDEO_COUNT}, observed={len(videos)}",
        )
    if len(annotations) != EXPECTED_ANNOTATION_COUNT:
        add_issue(
            issues, "error", "unexpected_annotation_count", "", LE2I_ROOT,
            f"expected={EXPECTED_ANNOTATION_COUNT}, observed={len(annotations)}",
        )
    annotated_adl_count = sum(
        row["sequence_type"] == "adl_annotated" for row in inventory_rows
    )
    unannotated_adl_count = sum(
        row["sequence_type"] == "adl_unannotated" for row in inventory_rows
    )
    if len(event_rows) != EXPECTED_FALL_COUNT:
        add_issue(
            issues, "error", "unexpected_fall_count", "", LE2I_ROOT,
            f"expected={EXPECTED_FALL_COUNT}, observed={len(event_rows)}",
        )
    if annotated_adl_count != EXPECTED_ANNOTATED_ADL_COUNT:
        add_issue(
            issues, "error", "unexpected_annotated_adl_count", "", LE2I_ROOT,
            f"expected={EXPECTED_ANNOTATED_ADL_COUNT}, observed={annotated_adl_count}",
        )
    if unannotated_adl_count != EXPECTED_UNANNOTATED_ADL_COUNT:
        add_issue(
            issues, "error", "unexpected_unannotated_adl_count", "", LE2I_ROOT,
            f"expected={EXPECTED_UNANNOTATED_ADL_COUNT}, "
            f"observed={unannotated_adl_count}",
        )
    if observed_scenes != EXPECTED_SCENES:
        add_issue(
            issues, "error", "unexpected_scene_set", "", LE2I_ROOT,
            f"expected={sorted(EXPECTED_SCENES)}, observed={sorted(observed_scenes)}",
        )

    OUTPUT_DIR.mkdir(parents=True)
    write_csv(OUTPUT_DIR / "sequence_inventory.csv", inventory_rows, list(inventory_rows[0]))
    write_csv(OUTPUT_DIR / "fall_events.csv", event_rows, list(event_rows[0]))
    issue_fields = ["severity", "code", "sequence_id", "path", "details"]
    write_csv(OUTPUT_DIR / "issues.csv", issues, issue_fields)
    shutil.copy2(POLICY_LOCK, OUTPUT_DIR / "locked_policy.txt")
    shutil.copy2(ARTIFACT_LOCK, OUTPUT_DIR / "locked_artifacts_sha256.txt")

    scene_summary = defaultdict(
        lambda: {
            "videos": 0,
            "falls": 0,
            "adl_annotated": 0,
            "adl_unannotated": 0,
        }
    )
    for row in inventory_rows:
        values = scene_summary[row["scene"]]
        values["videos"] += 1
        if row["sequence_type"] == "fall":
            values["falls"] += 1
        else:
            values[row["sequence_type"]] += 1

    severity_counts = Counter(issue["severity"] for issue in issues)
    code_counts = Counter(issue["code"] for issue in issues)
    summary = {
        "protocol": {
            "dataset": "Le2i",
            "role": "frozen final blind test",
            "model_inference_performed": False,
            "labels_used_for_training_or_threshold_selection": False,
            "evaluation_negative_exposure": (
                "all ADL-only video frames plus pre-onset frames of fall videos; "
                "post-onset frames excluded from false-alarm exposure"
            ),
            "event_detection_rule": (
                "one detection at or after official onset; later alarms in the same "
                "fall sequence are duplicate alarms"
            ),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "video_count": len(videos),
        "fall_video_count": len(event_rows),
        "adl_video_count": annotated_adl_count + unannotated_adl_count,
        "annotated_adl_video_count": annotated_adl_count,
        "unannotated_adl_video_count": unannotated_adl_count,
        "annotation_file_count": len(annotations),
        "scene_count": len(observed_scenes),
        "scene_summary": dict(sorted(scene_summary.items())),
        "total_frames": sum(row["frame_count"] for row in inventory_rows),
        "total_duration_hours": sum(row["duration_seconds"] for row in inventory_rows) / 3600.0,
        "negative_exposure_hours": sum(
            row["negative_exposure_seconds"] for row in inventory_rows
        ) / 3600.0,
        "fps_distribution": dict(sorted(Counter(
            f"{row['fps']:.6f}" for row in inventory_rows
        ).items())),
        "frame_size_distribution": dict(sorted(Counter(
            f"{row['width']}x{row['height']}" for row in inventory_rows
        ).items())),
        "annotation_layout": {
            "event_lines_at_start": sum(
                row["annotation_present"]
                and row["annotation_event_lines_at_start"]
                for row in inventory_rows
            ),
            "event_lines_embedded": sum(
                row["annotation_present"]
                and not row["annotation_event_lines_at_start"]
                for row in inventory_rows
            ),
            "annotations_with_duplicate_rows": sum(
                row["annotation_duplicate_rows"] > 0 for row in inventory_rows
            ),
            "duplicate_rows_total": sum(
                row["annotation_duplicate_rows"] for row in inventory_rows
            ),
            "conflicting_duplicate_rows_total": sum(
                row["annotation_conflicting_duplicate_rows"]
                for row in inventory_rows
            ),
        },
        "issue_count": len(issues),
        "error_count": severity_counts.get("error", 0),
        "warning_count": severity_counts.get("warning", 0),
        "issue_code_counts": dict(sorted(code_counts.items())),
        "lock_verification": lock_summary,
        "inventory_script_sha256": sha256(Path(__file__).resolve()),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"\nCompleted: {OUTPUT_DIR}")
    if summary["error_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
