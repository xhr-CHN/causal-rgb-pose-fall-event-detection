#!/usr/bin/env python3
"""Train quality-gated late-logit RGB/Pose fusion V2 on CAUCAFall.

V2 addresses the nearly constant V1 gate with source-only controlled modality
corruption and explicit gate supervision. No source test data or URFD is read.
Requires only PyTorch, NumPy, and the offline V1 helper script.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_quality_gated_fusion_tcn import (
    CausalEncoder,
    FusionDataset,
    QUALITY_NAMES,
    STATE_MACHINE,
    STATE_NAMES,
    compute_stats,
    json_ready,
    load_split,
    multiclass_metrics,
    read_csv_rows,
    run_state_machine,
    save_tabular_outputs,
    set_seed,
    write_csv_rows,
)


SEED = 42
ROOT = Path("/home/data/yoloA27")
VAL_METADATA = (
    ROOT
    / "experiments/pose_tcn_3state_kinematic_v2_seed42/val_predictions.csv"
)
OUT_DIR = ROOT / "experiments/quality_gated_logit_fusion_v2_seed42"

BATCH_SIZE = 128
MAX_EPOCHS = 120
PATIENCE = 18
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
HIDDEN_DIM = 64
NUM_WORKERS = 4

# Predefined source-only corruption protocol. Exactly one of Pose or RGB is
# strongly corrupted in half of the training samples. Clean samples remain 50%.
POSE_CORRUPTION_PROBABILITY = 0.25
RGB_CORRUPTION_PROBABILITY = 0.25
CLEAN_PROBABILITY = 0.50

AUXILIARY_BRANCH_LOSS_WEIGHT = 0.20
GATE_SUPERVISION_LOSS_WEIGHT = 0.50

# Model selection rewards clean accuracy and the ability to fall back to the
# surviving modality. These weights are fixed before any external evaluation.
CLEAN_SELECTION_WEIGHT = 0.50
POSE_MISSING_SELECTION_WEIGHT = 0.25
RGB_MISSING_SELECTION_WEIGHT = 0.25


class ReliabilityGatedLogitFusion(nn.Module):
    def __init__(self, pose_dim, rgb_dim, quality_dim, hidden_dim, dropout):
        super().__init__()
        self.pose_encoder = CausalEncoder(pose_dim, hidden_dim, dropout)
        self.rgb_encoder = CausalEncoder(rgb_dim, hidden_dim, dropout)
        self.pose_classifier = nn.Linear(hidden_dim, 3)
        self.rgb_classifier = nn.Linear(hidden_dim, 3)

        # 5 quality means + 5 quality minima + three reliability statistics
        # per branch + one branch-disagreement statistic = 17 inputs.
        gate_input_dim = quality_dim * 2 + 3 + 3 + 1
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    @staticmethod
    def reliability(probabilities):
        confidence = probabilities.max(dim=1).values
        entropy = -(
            probabilities * torch.log(probabilities.clamp_min(1e-8))
        ).sum(dim=1) / np.log(3.0)
        top_two = probabilities.topk(2, dim=1).values
        margin = top_two[:, 0] - top_two[:, 1]
        return torch.stack([confidence, entropy, margin], dim=1)

    def forward(self, pose, rgb, quality):
        pose_features = self.pose_encoder(pose)[:, :, -1]
        rgb_features = self.rgb_encoder(rgb)[:, :, -1]
        pose_logits = self.pose_classifier(pose_features)
        rgb_logits = self.rgb_classifier(rgb_features)

        pose_probabilities = torch.softmax(pose_logits, dim=1)
        rgb_probabilities = torch.softmax(rgb_logits, dim=1)
        quality_mean = quality.mean(dim=1)
        quality_minimum = quality.min(dim=1).values

        # Reliability statistics are detached so branch classifiers cannot
        # manipulate their probabilities merely to influence the gate.
        pose_reliability = self.reliability(pose_probabilities.detach())
        rgb_reliability = self.reliability(rgb_probabilities.detach())
        disagreement = torch.abs(
            pose_probabilities.detach() - rgb_probabilities.detach()
        ).mean(dim=1, keepdim=True)
        gate_inputs = torch.cat(
            [
                quality_mean,
                quality_minimum,
                pose_reliability,
                rgb_reliability,
                disagreement,
            ],
            dim=1,
        )
        gate_logit = self.gate(gate_inputs)
        pose_weight = torch.sigmoid(gate_logit)
        fused_logits = pose_weight * pose_logits + (1.0 - pose_weight) * rgb_logits
        return {
            "fused": fused_logits,
            "pose": pose_logits,
            "rgb": rgb_logits,
            "gate_logit": gate_logit.squeeze(1),
            "pose_weight": pose_weight.squeeze(1),
            "pose_probabilities": pose_probabilities,
            "rgb_probabilities": rgb_probabilities,
            "branch_disagreement": disagreement.squeeze(1),
        }


def corrupt_modalities(pose, rgb, quality):
    """Apply strong, labeled corruption to one modality per selected sample."""
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
        count = len(pose_indices)
        # Half use complete loss. The remainder imitate heavy joint occlusion:
        # most standardized feature channels disappear and the rest are noisy.
        complete_loss = torch.rand(count, device=pose.device) < 0.5
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
        count = len(rgb_indices)
        complete_loss = torch.rand(count, device=rgb.device) < 0.5
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

    # Pose weight targets: 0 when Pose is bad, 1 when RGB is bad. Clean samples
    # do not receive gate supervision and are determined by classification.
    gate_targets = torch.zeros(batch_size, device=pose.device)
    gate_targets[rgb_bad] = 1.0
    gate_supervision_mask = pose_bad | rgb_bad
    return (
        pose_out,
        rgb_out,
        quality_out,
        pose_bad,
        rgb_bad,
        gate_targets,
        gate_supervision_mask,
    )


def masked_mean(values, mask):
    if mask.any():
        return values[mask].mean()
    return values.new_tensor(0.0)


@torch.no_grad()
def evaluate_condition(model, loader, class_weights, device, condition):
    model.eval()
    all_probabilities = []
    all_pose_probabilities = []
    all_rgb_probabilities = []
    all_labels = []
    all_pose_weights = []
    all_disagreement = []
    total_loss = 0.0

    for pose, rgb, quality, labels in loader:
        pose = pose.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True)
        quality = quality.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if condition == "pose_missing":
            pose = torch.zeros_like(pose)
            quality = torch.zeros_like(quality)
        elif condition == "rgb_missing":
            rgb = torch.zeros_like(rgb)
        elif condition != "clean":
            raise ValueError(f"Unknown validation condition: {condition}")

        output = model(pose, rgb, quality)
        loss = F.cross_entropy(output["fused"], labels, weight=class_weights)
        total_loss += loss.item() * len(labels)
        all_probabilities.append(torch.softmax(output["fused"], 1).cpu().numpy())
        all_pose_probabilities.append(output["pose_probabilities"].cpu().numpy())
        all_rgb_probabilities.append(output["rgb_probabilities"].cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_pose_weights.append(output["pose_weight"].cpu().numpy())
        all_disagreement.append(output["branch_disagreement"].cpu().numpy())

    probabilities = np.concatenate(all_probabilities)
    pose_probabilities = np.concatenate(all_pose_probabilities)
    rgb_probabilities = np.concatenate(all_rgb_probabilities)
    labels = np.concatenate(all_labels)
    pose_weights = np.concatenate(all_pose_weights)
    disagreement = np.concatenate(all_disagreement)
    metrics = multiclass_metrics(labels, probabilities)
    metrics["loss"] = total_loss / len(labels)
    metrics["mean_pose_weight"] = float(pose_weights.mean())
    metrics["std_pose_weight"] = float(pose_weights.std())
    metrics["minimum_pose_weight"] = float(pose_weights.min())
    metrics["maximum_pose_weight"] = float(pose_weights.max())
    metrics["mean_branch_disagreement"] = float(disagreement.mean())
    return {
        "metrics": metrics,
        "probabilities": probabilities,
        "pose_probabilities": pose_probabilities,
        "rgb_probabilities": rgb_probabilities,
        "labels": labels,
        "pose_weights": pose_weights,
        "disagreement": disagreement,
    }


def gate_response_checks(clean, pose_missing, rgb_missing):
    clean_weight = clean["metrics"]["mean_pose_weight"]
    pose_missing_weight = pose_missing["metrics"]["mean_pose_weight"]
    rgb_missing_weight = rgb_missing["metrics"]["mean_pose_weight"]
    return {
        "clean_mean_pose_weight": clean_weight,
        "pose_missing_mean_pose_weight": pose_missing_weight,
        "rgb_missing_mean_pose_weight": rgb_missing_weight,
        "pose_missing_weight_drop": clean_weight - pose_missing_weight,
        "rgb_missing_weight_increase": rgb_missing_weight - clean_weight,
        "ordered_response": bool(
            pose_missing_weight < clean_weight < rgb_missing_weight
        ),
    }


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
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("External data used: NO", flush=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_gate_loss = 0.0
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
                gate_targets,
                gate_mask,
            ) = corrupt_modalities(pose, rgb, quality)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(pose_input, rgb_input, quality_input)
                fused_loss = F.cross_entropy(
                    output["fused"], labels, weight=class_weights
                )
                pose_losses = F.cross_entropy(
                    output["pose"], labels, weight=class_weights, reduction="none"
                )
                rgb_losses = F.cross_entropy(
                    output["rgb"], labels, weight=class_weights, reduction="none"
                )
                pose_auxiliary_loss = masked_mean(pose_losses, ~pose_bad)
                rgb_auxiliary_loss = masked_mean(rgb_losses, ~rgb_bad)
                if gate_mask.any():
                    gate_loss = F.binary_cross_entropy_with_logits(
                        output["gate_logit"][gate_mask], gate_targets[gate_mask]
                    )
                else:
                    gate_loss = output["fused"].new_tensor(0.0)
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
            f"{gate_checks['rgb_missing_mean_pose_weight']:.3f}",
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
                        "gate_supervision": "Pose bad -> weight 0; RGB bad -> weight 1",
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
                OUT_DIR / "quality_gated_logit_fusion_v2_best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(
        OUT_DIR / "quality_gated_logit_fusion_v2_best.pt", map_location=device
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
                "prob_adl_fusion_v2": float(probabilities[index, 0]),
                "prob_falling_fusion_v2": float(probabilities[index, 1]),
                "prob_fallen_fusion_v2": float(probabilities[index, 2]),
                "pose_prob_adl": float(clean["pose_probabilities"][index, 0]),
                "pose_prob_falling": float(clean["pose_probabilities"][index, 1]),
                "pose_prob_fallen": float(clean["pose_probabilities"][index, 2]),
                "rgb_prob_adl": float(clean["rgb_probabilities"][index, 0]),
                "rgb_prob_falling": float(clean["rgb_probabilities"][index, 1]),
                "rgb_prob_fallen": float(clean["rgb_probabilities"][index, 2]),
                "predicted_state_id_fusion_v2": int(predictions[index]),
                "predicted_state_fusion_v2": STATE_NAMES[predictions[index]],
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
            "model": "reliability-gated late-logit RGB/Pose causal TCN V2",
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

    print("\nV2 training completed.")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
