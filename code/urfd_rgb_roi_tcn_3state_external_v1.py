from __future__ import annotations

import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import urfd_pose_tcn_external_v1 as urfd
from train_pose_mlp_ablation_v1 import binary_metrics, ranking_metrics
from train_pose_tcn_baseline_v1 import PoseTCN
from urfd_pose_tcn_3state_external_v1 import (
    STATE_NAMES,
    STATE_TO_ID,
    event_evaluation,
    multiclass_brier,
    multiclass_metrics,
    top_label_ece,
)


ROOT = Path("/home/data/yoloA27/URFD")
FRAME_LABELS = ROOT / "metadata/rgb_frame_labels.csv"
MANUAL_EVENTS = ROOT / "metadata/manual_event_annotations.csv"
EMBEDDING_DIR = Path("/home/data/yoloA27/features/urfd_rgb_roi_embeddings_v1")
EMBEDDING_PATH = EMBEDDING_DIR / "embeddings.npy"
EMBEDDING_METADATA = EMBEDDING_DIR / "metadata.csv"
CHECKPOINT = Path(
    "/home/data/yoloA27/experiments/rgb_roi_tcn_3state_v1_seed42/"
    "rgb_roi_tcn_3state_best.pt"
)
OUTPUT_DIR = Path(
    "/home/data/yoloA27/experiments/urfd_rgb_roi_tcn_3state_external_v1"
)

TARGET_FPS = 20.0
SAMPLE_INTERVAL_MS = 1000.0 / TARGET_FPS
WINDOW_LENGTH = 32
EMBEDDING_DIMENSION = 256
TCN_BATCH_SIZE = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resample_embeddings(
    embeddings: np.ndarray, rows: list[dict[str, str]]
) -> tuple[np.ndarray, list[dict], list[dict]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["sequence_id"]].append(index)
    if len(groups) != 70:
        raise RuntimeError(f"Expected 70 sequences, found {len(groups)}")

    sampled_arrays = []
    sampled_metadata = []
    sequence_summary = []
    for sequence_id in sorted(groups):
        indices = sorted(groups[sequence_id], key=lambda i: int(rows[i]["timestamp_ms"]))
        timestamps = np.asarray([int(rows[i]["timestamp_ms"]) for i in indices])
        targets = np.arange(
            float(timestamps[0]),
            float(timestamps[-1]) + 1e-6,
            SAMPLE_INTERVAL_MS,
        )
        selected_local = urfd.nearest_indices(timestamps, targets)
        if len(set(selected_local.tolist())) != len(selected_local):
            raise RuntimeError(f"Resampling duplicated source frames: {sequence_id}")
        selected_global = [indices[int(item)] for item in selected_local]
        sampled_arrays.append(np.asarray(embeddings[selected_global], dtype=np.float32))

        for sample_index, (target_ms, global_index) in enumerate(
            zip(targets, selected_global), start=1
        ):
            source = rows[global_index]
            sampled_metadata.append(
                {
                    "sequence_id": sequence_id,
                    "category": source["category"],
                    "sample_index": sample_index,
                    "sample_timestamp_ms": int(round(float(target_ms))),
                    "source_frame_number": int(source["frame_number"]),
                    "source_timestamp_ms": int(source["timestamp_ms"]),
                    "event_label": source["event_label"],
                    "gt_fall": int(source["event_label"] in {"Falling", "Fallen"}),
                    "roi_crop_used": int(source["roi_crop_used"]),
                }
            )
        sequence_summary.append(
            {
                "sequence_id": sequence_id,
                "category": rows[indices[0]]["category"],
                "source_frames": len(indices),
                "resampled_frames": len(targets),
                "start_timestamp_ms": int(timestamps[0]),
                "end_timestamp_ms": int(timestamps[-1]),
            }
        )

    sampled = np.concatenate(sampled_arrays, axis=0)
    if len(sampled) != len(sampled_metadata):
        raise RuntimeError("Sampled embeddings and metadata differ in length")
    if not np.isfinite(sampled).all():
        raise RuntimeError("Non-finite resampled RGB embeddings")
    return sampled, sampled_metadata, sequence_summary


def build_windows(
    sampled: np.ndarray, metadata: list[dict]
) -> tuple[np.ndarray, list[dict]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[row["sequence_id"]].append(index)
    total_windows = sum(max(0, len(indices) - WINDOW_LENGTH + 1) for indices in groups.values())
    x = np.empty(
        (total_windows, WINDOW_LENGTH, EMBEDDING_DIMENSION), dtype=np.float32
    )
    output_metadata = []
    window_index = 0
    for sequence_id in sorted(groups):
        indices = sorted(groups[sequence_id], key=lambda i: int(metadata[i]["sample_index"]))
        sequence_embeddings = sampled[indices]
        sequence_rows = [metadata[i] for i in indices]
        for target_index in range(WINDOW_LENGTH - 1, len(indices)):
            start = target_index - WINDOW_LENGTH + 1
            x[window_index] = sequence_embeddings[start : target_index + 1]
            target = sequence_rows[target_index]
            output_metadata.append(
                {
                    "sequence_id": sequence_id,
                    "category": target["category"],
                    "sample_index": int(target["sample_index"]),
                    "sample_timestamp_ms": int(target["sample_timestamp_ms"]),
                    "source_frame_number": int(target["source_frame_number"]),
                    "source_timestamp_ms": int(target["source_timestamp_ms"]),
                    "event_label": target["event_label"],
                    "gt_fall": int(target["gt_fall"]),
                    "rgb_roi_crop_ratio": float(
                        np.mean(
                            [
                                int(row["roi_crop_used"])
                                for row in sequence_rows[start : target_index + 1]
                            ]
                        )
                    ),
                }
            )
            window_index += 1
    if window_index != total_windows or len(output_metadata) != total_windows:
        raise RuntimeError("RGB external window count mismatch")
    if x.shape[1:] != (WINDOW_LENGTH, EMBEDDING_DIMENSION):
        raise RuntimeError(f"Unexpected external RGB window shape: {x.shape}")
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite external RGB windows")
    return x, output_metadata


def make_model() -> PoseTCN:
    model = PoseTCN(input_features=EMBEDDING_DIMENSION)
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    return model


@torch.no_grad()
def predict(x: np.ndarray, checkpoint: dict) -> np.ndarray:
    mean = checkpoint["input_mean"].numpy().reshape(1, 1, -1)
    std = checkpoint["input_std"].numpy().reshape(1, 1, -1)
    standardized = (x - mean) / std
    if not np.isfinite(standardized).all():
        raise RuntimeError("Non-finite standardized external RGB windows")
    model = make_model().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(standardized.astype(np.float32))),
        batch_size=TCN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    outputs = []
    for batch_index, (batch,) in enumerate(loader, start=1):
        logits = model(batch.to(DEVICE, non_blocking=True))
        outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        if batch_index % 10 == 0 or batch_index == len(loader):
            print(f"RGB-TCN inference: {batch_index}/{len(loader)} batches", flush=True)
    return np.concatenate(outputs)


def main() -> None:
    for path in (
        FRAME_LABELS,
        MANUAL_EVENTS,
        EMBEDDING_PATH,
        EMBEDDING_METADATA,
        CHECKPOINT,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    frame_rows = urfd.read_csv(FRAME_LABELS)
    manual_rows = urfd.read_csv(MANUAL_EVENTS)
    embedding_rows = urfd.read_csv(EMBEDDING_METADATA)
    embeddings = np.load(EMBEDDING_PATH, mmap_mode="r")
    if len(frame_rows) != 11936 or len(embedding_rows) != 11936:
        raise RuntimeError("Expected 11936 URFD frame and embedding rows")
    if embeddings.shape != (11936, EMBEDDING_DIMENSION):
        raise RuntimeError(f"Unexpected embeddings: {embeddings.shape}")
    if len(manual_rows) != 30:
        raise RuntimeError(f"Expected 30 fall annotations, found {len(manual_rows)}")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    if checkpoint.get("modality") != "person-ROI RGB embedding":
        raise ValueError("Unexpected checkpoint modality")
    expected_names = [f"rgb_embedding_{index:03d}" for index in range(256)]
    if list(checkpoint["feature_names"]) != expected_names:
        raise ValueError("RGB checkpoint feature names do not match")
    if tuple(checkpoint["state_names"]) != STATE_NAMES:
        raise ValueError("RGB checkpoint state names do not match")
    policy = dict(checkpoint["config"]["state_machine"])

    print(f"Python: {platform.python_version()}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("Protocol: frozen RGB-TCN zero-shot evaluation", flush=True)
    print(f"Locked state machine: {json.dumps(policy)}", flush=True)

    sampled, sampled_metadata, resampling_summary = resample_embeddings(
        embeddings, embedding_rows
    )
    urfd.save_csv(OUTPUT_DIR / "urfd_resampling_summary.csv", resampling_summary)
    print(
        f"Resampling: {len(embeddings)} source frames -> {len(sampled)} at 20 FPS",
        flush=True,
    )
    x, metadata = build_windows(sampled, sampled_metadata)
    print(f"External RGB windows: {x.shape}", flush=True)
    probabilities = predict(x, checkpoint)
    if len(probabilities) != len(metadata):
        raise RuntimeError("RGB predictions and metadata differ in length")

    prediction_rows = []
    for item, probability in zip(metadata, probabilities):
        event_label = str(item["event_label"])
        if event_label not in STATE_TO_ID:
            raise ValueError(f"Unknown URFD state: {event_label}")
        predicted_id = int(np.argmax(probability))
        row = dict(item)
        row["target_state_id"] = STATE_TO_ID[event_label]
        row["target_state"] = event_label
        row["prob_adl"] = float(probability[0])
        row["prob_falling"] = float(probability[1])
        row["prob_fallen"] = float(probability[2])
        row["predicted_state_id"] = predicted_id
        row["predicted_state"] = STATE_NAMES[predicted_id]
        prediction_rows.append(row)

    y_true = np.asarray([row["target_state_id"] for row in prediction_rows], dtype=int)
    y_pred = np.asarray([row["predicted_state_id"] for row in prediction_rows], dtype=int)
    metrics = multiclass_metrics(y_true, probabilities)
    metrics["multiclass_brier_score"] = multiclass_brier(y_true, probabilities)
    metrics["top_label_ece_10_bins"] = top_label_ece(y_true, probabilities)
    binary_true = (y_true != 0).astype(int)
    binary_pred = (y_pred != 0).astype(int)
    fall_score = 1.0 - probabilities[:, 0]
    collapsed = binary_metrics(binary_true, binary_pred)
    auroc, auprc = ranking_metrics(binary_true, fall_score)
    collapsed["auroc"] = auroc
    collapsed["auprc"] = auprc
    event_rows, event_summary = event_evaluation(
        prediction_rows, manual_rows, frame_rows, policy
    )

    urfd.save_csv(OUTPUT_DIR / "frame_predictions.csv", prediction_rows)
    urfd.save_csv(OUTPUT_DIR / "event_results.csv", event_rows)
    summary = {
        "protocol": {
            "source_training_dataset": "CAUCAFall",
            "external_test_dataset": "URFD Camera 0 RGB",
            "model": "person-ROI RGB-embedding causal TCN",
            "fine_tuning": False,
            "target_data_used_for_model_or_policy_selection": False,
            "source_locked_state_machine": policy,
            "external_resampled_fps": TARGET_FPS,
            "window_length": WINDOW_LENGTH,
            "input_features": EMBEDDING_DIMENSION,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "three_state_frame_level": metrics,
        "collapsed_fall_vs_adl_frame_level": collapsed,
        "event_level_locked_state_machine": event_summary,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Completed. Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
