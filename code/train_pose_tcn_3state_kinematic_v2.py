from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import train_pose_tcn_3state_v1 as base
from train_pose_tcn_baseline_v1 import PoseTCN


DATA_DIR = Path(
    "/home/data/yoloA27/features/"
    "caucafall_pose_windows_3state_kinematic_v2"
)
OUTPUT_DIR = Path(
    "/home/data/yoloA27/experiments/"
    "pose_tcn_3state_kinematic_v2_seed42"
)
INPUT_FEATURES = 95


def load_data():
    train_npz = np.load(DATA_DIR / "train_windows.npz")
    val_npz = np.load(DATA_DIR / "val_windows.npz")
    x_train = train_npz["x"].astype(np.float32)
    y_train = train_npz["y"].astype(np.int64)
    x_val = val_npz["x"].astype(np.float32)
    y_val = val_npz["y"].astype(np.int64)
    train_names = train_npz["feature_names"].astype(str)
    val_names = val_npz["feature_names"].astype(str)
    state_names = train_npz["state_names"].astype(str)

    if x_train.shape != (2513, 32, INPUT_FEATURES):
        raise ValueError(f"Unexpected train shape: {x_train.shape}")
    if x_val.shape != (3450, 32, INPUT_FEATURES):
        raise ValueError(f"Unexpected validation shape: {x_val.shape}")
    if not np.array_equal(train_names, val_names):
        raise ValueError("Train/validation feature names differ")
    if tuple(state_names.tolist()) != base.STATE_NAMES:
        raise ValueError(f"Unexpected states: {state_names.tolist()}")
    if set(np.unique(y_train)) != {0, 1, 2}:
        raise ValueError("Training split does not contain all states")
    if set(np.unique(y_val)) != {0, 1, 2}:
        raise ValueError("Validation split does not contain all states")

    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
        raise ValueError("Standardized input contains NaN or infinity")
    return x_train, y_train, x_val, y_val, train_names, mean, std


def make_model() -> PoseTCN:
    model = PoseTCN(input_features=INPUT_FEATURES)
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    return model.to(base.DEVICE)


def main() -> None:
    base.set_seed(base.SEED)
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    base.OUTPUT_DIR = OUTPUT_DIR

    x_train, y_train, x_val, y_val, names, mean, std = load_data()
    train_loader, val_loader = base.make_loaders(
        x_train, y_train, x_val, y_val
    )
    model = make_model()

    class_counts = np.bincount(y_train, minlength=3)
    class_weights = len(y_train) / (3.0 * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights, dtype=torch.float32, device=base.DEVICE
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base.LEARNING_RATE,
        weight_decay=base.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=base.MAX_EPOCHS
    )

    print(f"Device: {base.DEVICE}", flush=True)
    print(f"Input features: {INPUT_FEATURES}", flush=True)
    print(f"Class counts: {class_counts.tolist()}", flush=True)
    print(f"Class weights: {class_weights.tolist()}", flush=True)
    print(
        f"Parameters: {sum(parameter.numel() for parameter in model.parameters())}",
        flush=True,
    )

    history = []
    best_auprc = -math.inf
    best_epoch = 0
    stale = 0
    state_path = OUTPUT_DIR / "best_model_state.pt"
    for epoch in range(1, base.MAX_EPOCHS + 1):
        train_loss = base.train_epoch(
            model, train_loader, criterion, optimizer
        )
        val_metrics, _, _ = base.evaluate(model, val_loader, criterion)
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_auprc": val_metrics["macro_auprc_ovr"],
            "val_macro_auroc": val_metrics["macro_auroc_ovr"],
            "val_falling_f1": val_metrics["per_class"]["Falling"]["f1"],
            "val_fallen_f1": val_metrics["per_class"]["Fallen"]["f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} loss={train_loss:.4f}/{val_metrics['loss']:.4f} "
            f"macroF1={val_metrics['macro_f1']:.4f} "
            f"FallingF1={row['val_falling_f1']:.4f} "
            f"AUPRC={val_metrics['macro_auprc_ovr']:.4f}",
            flush=True,
        )
        if val_metrics["macro_auprc_ovr"] > best_auprc + 1e-6:
            best_auprc = val_metrics["macro_auprc_ovr"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), state_path)
        else:
            stale += 1
            if stale >= base.PATIENCE:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    base.save_csv(OUTPUT_DIR / "history.csv", history)
    base.plot_history(history)
    model.load_state_dict(torch.load(state_path, map_location=base.DEVICE))
    validation_metrics, y_true, probabilities = base.evaluate(
        model, val_loader, criterion
    )
    metadata = base.read_csv(DATA_DIR / "val_window_metadata.csv")
    expected_labels = np.asarray(
        [int(row["target_state_id"]) for row in metadata]
    )
    if len(metadata) != len(probabilities):
        raise RuntimeError("Validation metadata and predictions differ in length")
    if not np.array_equal(y_true, expected_labels):
        raise RuntimeError("Validation labels do not match metadata")

    event_rows, event_summary, predictions = base.event_evaluation(
        metadata, probabilities
    )
    base.save_csv(OUTPUT_DIR / "val_predictions.csv", predictions)
    base.save_csv(OUTPUT_DIR / "val_event_results.csv", event_rows)
    base.plot_confusion(validation_metrics["confusion_matrix"])

    state_machine = {
        "falling_required": base.FALLING_REQUIRED,
        "falling_lookback_frames": base.FALLING_LOOKBACK_FRAMES,
        "arm_memory_frames": base.ARM_MEMORY_FRAMES,
        "fallen_consecutive_frames": base.FALLEN_CONSECUTIVE_FRAMES,
        "adl_reset_frames": base.ADL_RESET_FRAMES,
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_mean": torch.from_numpy(mean.squeeze()).float(),
        "input_std": torch.from_numpy(std.squeeze()).float(),
        "feature_names": names.tolist(),
        "state_names": list(base.STATE_NAMES),
        "feature_transform": "bbox-scale-normalized-global-kinematics-v2",
        "config": {
            "window_length": 32,
            "input_features": INPUT_FEATURES,
            "state_machine": state_machine,
        },
    }
    torch.save(
        checkpoint,
        OUTPUT_DIR / "pose_tcn_3state_kinematic_v2_best.pt",
    )

    summary = {
        "protocol": {
            "model": "kinematic-v2 three-state causal Pose-TCN",
            "states": list(base.STATE_NAMES),
            "seed": base.SEED,
            "train_subjects": [1, 3, 6, 7, 8, 9],
            "val_subjects": [4, 5],
            "test_used": False,
            "external_data_used": False,
            "early_stopping_metric": "validation macro one-vs-rest AUPRC",
            "standardization": "source-train statistics only",
            "window_length": 32,
            "input_features": INPUT_FEATURES,
            "state_machine": state_machine,
        },
        "feature_design": {
            "name": "bbox-scale-normalized-global-kinematics-v2",
            "description": (
                "Camera/framing-dependent bbox channels replaced by aspect "
                "ratio and bbox-scale-normalized translation/scale changes"
            ),
            "feature_names": names.tolist(),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(base.DEVICE),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "model_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "class_counts": class_counts.tolist(),
        "class_weights": class_weights.tolist(),
        "best_epoch": best_epoch,
        "best_validation_macro_auprc": best_auprc,
        "validation_frame_metrics": validation_metrics,
        "validation_event_metrics": event_summary,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Completed. Outputs: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
