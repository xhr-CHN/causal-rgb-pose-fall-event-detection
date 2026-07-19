#!/usr/bin/env python3
"""Train source-only reliability-gated late-logit fusion V3.

V3 retains explicit missing-modality supervision and adds a soft clean-sample
gate target based on which branch assigns lower loss to the true source label.
No CAUCAFall test data or external dataset is read.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_quality_gated_fusion_tcn import (
    FusionDataset,
    QUALITY_NAMES,
    STATE_MACHINE,
    STATE_NAMES,
    compute_stats,
    json_ready,
    load_split,
    read_csv_rows,
    run_state_machine,
    save_tabular_outputs,
    set_seed,
    write_csv_rows,
)
from train_quality_gated_logit_fusion_v2 import (
    ReliabilityGatedLogitFusion,
    evaluate_condition,
    gate_response_checks,
    masked_mean,
)


SEED = 42
ROOT = Path("/home/data/yoloA27")
VAL_METADATA = (
    ROOT
    / "experiments/pose_tcn_3state_kinematic_v2_seed42/val_predictions.csv"
)
OUT_DIR = ROOT / "experiments/quality_gated_logit_fusion_v3_seed42"

BATCH_SIZE = 128
MAX_EPOCHS = 120
PATIENCE = 18
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
HIDDEN_DIM = 64
NUM_WORKERS = 4

# V3 uses more clean examples than V2.
POSE_CORRUPTION_PROBABILITY = 0.20
RGB_CORRUPTION_PROBABILITY = 0.20
CLEAN_PROBABILITY = 0.60

AUXILIARY_BRANCH_LOSS_WEIGHT = 0.25
GATE_SUPERVISION_LOSS_WEIGHT = 0.35

# Soft target: sigmoid((RGB loss - Pose loss) / temperature).
CLEAN_GATE_TEMPERATURE = 0.50
CLEAN_GATE_TARGET_MINIMUM = 0.10
CLEAN_GATE_TARGET_MAXIMUM = 0.90
CORRUPTED_POSE_GATE_TARGET = 0.05
CORRUPTED_RGB_GATE_TARGET = 0.95

# Clean performance is the main source-domain model-selection criterion.
CLEAN_SELECTION_WEIGHT = 0.80
POSE_MISSING_SELECTION_WEIGHT = 0.10
RGB_MISSING_SELECTION_WEIGHT = 0.10


def corrupt_modalities_v3(pose, rgb, quality):
    batch_size = pose.shape[0]
    random_values = torch.rand(batch_size, device=pose.device)
    pose_bad = random_values < POSE_CORRUPTION_PROBABILITY
    rgb_bad = (random_values >= POSE_CORRUPTION_PROBABILITY) & (
        random_values
        < POSE_CORRUPTION_PROBABILITY + RGB_CORRUPTION_PROBABILITY
    )

    pose_out = pose.clone()
    rgb_out = rgb.clone()
    quality_out = quality.clone()

    pose_indices = torch.where(pose_bad)[0]
    if len(pose_indices):
        complete_loss = torch.rand(len(pose_indices), device=pose.device) < 0.5
        complete_indices = pose_indices[complete_loss]
        partial_indices = pose_indices[~complete_loss]
        if len(complete_indices):
            pose_out[complete_indices] = 0.0
            quality_out[complete_indices] = 0.0
        if len(partial_indices):
            channel_keep = (
                torch.rand(
                    len(partial_indices), 1, pose.shape[2], device=pose.device
                )
                > 0.65
            ).float()
            noise = 0.20 * torch.randn_like(pose_out[partial_indices])
            pose_out[partial_indices] = (
                pose_out[partial_indices] * channel_keep + noise * channel_keep
            )
            quality_scale = 0.05 + 0.20 * torch.rand(
                len(partial_indices), 1, 1, device=pose.device
            )
            quality_out[partial_indices] *= quality_scale

    rgb_indices = torch.where(rgb_bad)[0]
    if len(rgb_indices):
        complete_loss = torch.rand(len(rgb_indices), device=rgb.device) < 0.5
        complete_indices = rgb_indices[complete_loss]
        partial_indices = rgb_indices[~complete_loss]
        if len(complete_indices):
            rgb_out[complete_indices] = 0.0
        if len(partial_indices):
            keep_scale = 0.05 + 0.20 * torch.rand(
                len(partial_indices), 1, 1, device=rgb.device
            )
            noise = 0.20 * torch.randn_like(rgb_out[partial_indices])
            rgb_out[partial_indices] = (
                rgb_out[partial_indices] * keep_scale + noise
            )

    clean = ~(pose_bad | rgb_bad)
    return pose_out, rgb_out, quality_out, pose_bad, rgb_bad, clean


def build_gate_targets(output, labels, pose_bad, rgb_bad, clean):
    pose_branch_losses = F.cross_entropy(
        output["pose"].float(), labels, reduction="none"
    )
    rgb_branch_losses = F.cross_entropy(
        output["rgb"].float(), labels, reduction="none"
    )
    clean_targets = torch.sigmoid(
        (rgb_branch_losses.detach() - pose_branch_losses.detach())
        / CLEAN_GATE_TEMPERATURE
    ).clamp(CLEAN_GATE_TARGET_MINIMUM, CLEAN_GATE_TARGET_MAXIMUM)

    targets = clean_targets.clone()
    targets[pose_bad] = CORRUPTED_POSE_GATE_TARGET
    targets[rgb_bad] = CORRUPTED_RGB_GATE_TARGET
    return targets, pose_branch_losses, rgb_branch_losses


def summarize_clean_gate_targets(targets, clean_mask):
    if clean_mask.any():
        values = targets[clean_mask]
        return float(values.mean()), float(values.std())
    return float("nan"), float("nan")


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_pose, train_rgb, train_quality, train_y, pose_names, rgb_names = load_split(
        "train"
    )
    val_pose, val_rgb, val_quality, val_y, _, _ = load_split("val")

    pose_mean, pose_std = compute_stats(train_pose)
    rgb_mean, rgb_std = compute_stats(train_rgb)
    train_pose = ((train_pose - pose_mean) / pose_std).astype(np.float32)
    val_pose = ((val_pose - pose_mean) / pose_std).astype(np.float32)
    train_rgb = ((train_rgb - rgb_mean) / rgb_std).astype(np.float32)
    val_rgb = ((val_rgb - rgb_mean) / rgb_std).astype(np.float32)

    train_loader = DataLoader(
        FusionDataset(train_pose, train_rgb, train_quality, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        FusionDataset(val_pose, val_rgb, val_quality, val_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    counts = np.bincount(train_y, minlength=3)
    class_weights_np = len(train_y) / (3.0 * counts)
    class_weights = torch.tensor(
        class_weights_np, dtype=torch.float32, device=device
    )

    model = ReliabilityGatedLogitFusion(
        pose_dim=train_pose.shape[2],
        rgb_dim=train_rgb.shape[2],
        quality_dim=train_quality.shape[2],
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_score = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train windows: {len(train_y)} | Val windows: {len(val_y)}")
    print(f"Class counts: {counts.tolist()}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print("External data used: NO", flush=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_gate_loss = 0.0
        target_sum = 0.0
        target_square_sum = 0.0
        target_count = 0
        processed = 0

        for pose, rgb, quality, labels in train_loader:
            pose = pose.to(device, non_blocking=True)
            rgb = rgb.to(device, non_blocking=True)
            quality = quality.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            (
                pose_input,
                rgb_input,
                quality_input,
                pose_bad,
                rgb_bad,
                clean_mask,
            ) = corrupt_modalities_v3(pose, rgb, quality)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(pose_input, rgb_input, quality_input)
                gate_targets, pose_losses, rgb_losses = build_gate_targets(
                    output, labels, pose_bad, rgb_bad, clean_mask
                )
                fused_loss = F.cross_entropy(
                    output["fused"], labels, weight=class_weights
                )
                # Class weighting is used for branch training; the unweighted
                # losses above are retained only for the relative gate target.
                pose_auxiliary_losses = F.cross_entropy(
                    output["pose"], labels, weight=class_weights, reduction="none"
                )
                rgb_auxiliary_losses = F.cross_entropy(
                    output["rgb"], labels, weight=class_weights, reduction="none"
                )
                pose_auxiliary_loss = masked_mean(
                    pose_auxiliary_losses, ~pose_bad
                )
                rgb_auxiliary_loss = masked_mean(rgb_auxiliary_losses, ~rgb_bad)
                gate_loss = F.binary_cross_entropy_with_logits(
                    output["gate_logit"], gate_targets
                )
                loss = (
                    fused_loss
                    + AUXILIARY_BRANCH_LOSS_WEIGHT
                    * (pose_auxiliary_loss + rgb_auxiliary_loss)
                    + GATE_SUPERVISION_LOSS_WEIGHT * gate_loss
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(labels)
            running_gate_loss += gate_loss.item() * len(labels)
            if clean_mask.any():
                clean_targets = gate_targets[clean_mask].detach()
                target_sum += clean_targets.sum().item()
                target_square_sum += (clean_targets**2).sum().item()
                target_count += len(clean_targets)
            processed += len(labels)

        clean = evaluate_condition(model, val_loader, class_weights, device, "clean")
        pose_missing = evaluate_condition(
            model, val_loader, class_weights, device, "pose_missing"
        )
        rgb_missing = evaluate_condition(
            model, val_loader, class_weights, device, "rgb_missing"
        )
        selection_score = (
            CLEAN_SELECTION_WEIGHT
            * clean["metrics"]["macro_auprc_ovr"]
            + POSE_MISSING_SELECTION_WEIGHT
            * pose_missing["metrics"]["macro_auprc_ovr"]
            + RGB_MISSING_SELECTION_WEIGHT
            * rgb_missing["metrics"]["macro_auprc_ovr"]
        )
        gate_checks = gate_response_checks(clean, pose_missing, rgb_missing)
        clean_target_mean = target_sum / target_count if target_count else float("nan")
        clean_target_variance = (
            target_square_sum / target_count - clean_target_mean**2
            if target_count
            else float("nan")
        )
        clean_target_std = float(np.sqrt(max(0.0, clean_target_variance)))
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / processed,
                "train_gate_loss": running_gate_loss / processed,
                "selection_score": selection_score,
                "clean_macro_auprc": clean["metrics"]["macro_auprc_ovr"],
                "clean_macro_f1": clean["metrics"]["macro_f1"],
                "pose_missing_macro_auprc": pose_missing["metrics"]["macro_auprc_ovr"],
                "rgb_missing_macro_auprc": rgb_missing["metrics"]["macro_auprc_ovr"],
                "clean_pose_weight": gate_checks["clean_mean_pose_weight"],
                "pose_missing_weight": gate_checks["pose_missing_mean_pose_weight"],
                "rgb_missing_weight": gate_checks["rgb_missing_mean_pose_weight"],
                "clean_gate_target_mean": clean_target_mean,
                "clean_gate_target_std": clean_target_std,
            }
        )
        print(
            f"Epoch {epoch:03d} | loss={running_loss / processed:.5f} "
            f"score={selection_score:.5f} "
            f"clean_AP={clean['metrics']['macro_auprc_ovr']:.5f} "
            f"poseMissing_AP={pose_missing['metrics']['macro_auprc_ovr']:.5f} "
            f"rgbMissing_AP={rgb_missing['metrics']['macro_auprc_ovr']:.5f} "
            f"gate={gate_checks['pose_missing_mean_pose_weight']:.3f}/"
            f"{gate_checks['clean_mean_pose_weight']:.3f}/"
            f"{gate_checks['rgb_missing_mean_pose_weight']:.3f} "
            f"cleanTarget={clean_target_mean:.3f}",
            flush=True,
        )

        if selection_score > best_score + 1e-6:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "pose_mean": pose_mean.squeeze().astype(np.float32),
                    "pose_std": pose_std.squeeze().astype(np.float32),
                    "rgb_mean": rgb_mean.squeeze().astype(np.float32),
                    "rgb_std": rgb_std.squeeze().astype(np.float32),
                    "pose_feature_names": pose_names,
                    "rgb_feature_names": rgb_names,
                    "quality_feature_names": QUALITY_NAMES,
                    "quality_feature_indices": list(range(5)),
                    "state_names": STATE_NAMES,
                    "state_machine": STATE_MACHINE,
                    "model_config": {
                        "pose_dim": train_pose.shape[2],
                        "rgb_dim": train_rgb.shape[2],
                        "quality_dim": train_quality.shape[2],
                        "hidden_dim": HIDDEN_DIM,
                        "dropout": DROPOUT,
                        "window_length": train_pose.shape[1],
                    },
                    "corruption_protocol": {
                        "clean_probability": CLEAN_PROBABILITY,
                        "pose_corruption_probability": POSE_CORRUPTION_PROBABILITY,
                        "rgb_corruption_probability": RGB_CORRUPTION_PROBABILITY,
                        "pose_bad_target": CORRUPTED_POSE_GATE_TARGET,
                        "rgb_bad_target": CORRUPTED_RGB_GATE_TARGET,
                    },
                    "clean_gate_supervision": {
                        "formula": "sigmoid((rgb_loss-pose_loss)/temperature)",
                        "temperature": CLEAN_GATE_TEMPERATURE,
                        "minimum": CLEAN_GATE_TARGET_MINIMUM,
                        "maximum": CLEAN_GATE_TARGET_MAXIMUM,
                    },
                    "selection_protocol": {
                        "clean_macro_auprc_weight": CLEAN_SELECTION_WEIGHT,
                        "pose_missing_macro_auprc_weight": POSE_MISSING_SELECTION_WEIGHT,
                        "rgb_missing_macro_auprc_weight": RGB_MISSING_SELECTION_WEIGHT,
                    },
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_selection_score": best_score,
                },
                OUT_DIR / "quality_gated_logit_fusion_v3_best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(
        OUT_DIR / "quality_gated_logit_fusion_v3_best.pt", map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    clean = evaluate_condition(model, val_loader, class_weights, device, "clean")
    pose_missing = evaluate_condition(
        model, val_loader, class_weights, device, "pose_missing"
    )
    rgb_missing = evaluate_condition(
        model, val_loader, class_weights, device, "rgb_missing"
    )
    gate_checks = gate_response_checks(clean, pose_missing, rgb_missing)

    metadata = read_csv_rows(VAL_METADATA)
    labels = clean["labels"]
    if len(metadata) != len(labels):
        raise ValueError("Validation metadata row count does not match predictions")
    metadata_labels = np.asarray(
        [int(float(row["target_state_id"])) for row in metadata], dtype=np.int64
    )
    if not np.array_equal(metadata_labels, labels):
        raise ValueError("Validation metadata labels do not match NPZ labels")

    probabilities = clean["probabilities"]
    predictions = probabilities.argmax(1)
    output_rows = []
    for index, source in enumerate(metadata):
        row = dict(source)
        row.update(
            {
                "prob_adl_fusion_v3": float(probabilities[index, 0]),
                "prob_falling_fusion_v3": float(probabilities[index, 1]),
                "prob_fallen_fusion_v3": float(probabilities[index, 2]),
                "pose_prob_adl": float(clean["pose_probabilities"][index, 0]),
                "pose_prob_falling": float(clean["pose_probabilities"][index, 1]),
                "pose_prob_fallen": float(clean["pose_probabilities"][index, 2]),
                "rgb_prob_adl": float(clean["rgb_probabilities"][index, 0]),
                "rgb_prob_falling": float(clean["rgb_probabilities"][index, 1]),
                "rgb_prob_fallen": float(clean["rgb_probabilities"][index, 2]),
                "predicted_state_id_fusion_v3": int(predictions[index]),
                "predicted_state_fusion_v3": STATE_NAMES[predictions[index]],
                "pose_gate_weight": float(clean["pose_weights"][index]),
                "branch_disagreement": float(clean["disagreement"][index]),
            }
        )
        output_rows.append(row)
    write_csv_rows(OUT_DIR / "val_predictions.csv", output_rows)

    event_metrics, event_rows = run_state_machine(metadata, predictions)
    write_csv_rows(OUT_DIR / "val_event_results.csv", event_rows)
    save_tabular_outputs(history, labels, probabilities)

    summary = {
        "protocol": {
            "model": "clean-supervised reliability-gated late-logit fusion V3",
            "source_dataset": "CAUCAFall",
            "test_used": False,
            "external_data_used": False,
            "seed": SEED,
            "window_length": 32,
            "pose_features": 95,
            "rgb_features": 256,
            "quality_features": QUALITY_NAMES,
            "state_machine": STATE_MACHINE,
        },
        "training_corruption": checkpoint["corruption_protocol"],
        "clean_gate_supervision": checkpoint["clean_gate_supervision"],
        "model_selection": checkpoint["selection_protocol"],
        "environment": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "class_counts": counts.tolist(),
        "class_weights": class_weights_np.tolist(),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "clean_validation_frame_metrics": clean["metrics"],
        "pose_missing_validation_frame_metrics": pose_missing["metrics"],
        "rgb_missing_validation_frame_metrics": rgb_missing["metrics"],
        "gate_response": gate_checks,
        "clean_validation_event_metrics": event_metrics,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(json_ready(summary), file, ensure_ascii=False, indent=2)

    print("\nV3 training completed.")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
