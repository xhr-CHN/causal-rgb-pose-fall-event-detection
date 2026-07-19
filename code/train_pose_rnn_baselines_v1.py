from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_pose_mlp_ablation_v1 import (
    ALARM_CONSECUTIVE_FRAMES,
    BATCH_SIZE,
    CAUCAFALL_FPS,
    DATA_DIR,
    DEVICE,
    DROPOUT,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    SEED,
    WEIGHT_DECAY,
    best_f1_threshold,
    evaluate,
    event_evaluation,
    load_base_data,
    make_loaders,
    plot_confusion,
    plot_history,
    ranking_metrics,
    read_csv,
    save_csv,
    set_seed,
    train_epoch,
)


OUTPUT_ROOT = Path("/home/data/yoloA27/experiments/pose_rnn_baselines_v1_seed42")
HIDDEN_SIZE = 128
NUM_LAYERS = 2

EXPERIMENTS = {
    "pose_gru32": "GRU",
    "pose_lstm32": "LSTM",
}


class PoseRNN(nn.Module):
    def __init__(self, cell_type: str, input_dim: int = 103) -> None:
        super().__init__()
        rnn_class = nn.GRU if cell_type == "GRU" else nn.LSTM
        self.recurrent = rnn_class(
            input_size=input_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(x)
        return self.classifier(output[:, -1, :])


def prepare_sequence_input(x_train: np.ndarray, x_val: np.ndarray):
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return (x_train - mean) / std, (x_val - mean) / std, mean, std


def run_experiment(name, cell_type, base_data, metadata):
    set_seed(SEED)
    output_dir = OUTPUT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=False)

    x_train_raw, y_train, x_val_raw, y_val, feature_names = base_data
    x_train, x_val, mean, std = prepare_sequence_input(x_train_raw, x_val_raw)
    train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val)

    model = PoseRNN(cell_type=cell_type, input_dim=x_train.shape[-1]).to(DEVICE)
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2.0 * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"\n=== {name} ===", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Train windows: {len(y_train)}, val windows: {len(y_val)}", flush=True)
    print(f"Parameters: {parameters}", flush=True)

    history = []
    best_auprc = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    state_path = output_dir / "best_model_state.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_metrics, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_precision_at_0.5": val_metrics["precision"],
            "val_recall_at_0.5": val_metrics["recall_sensitivity"],
            "val_f1_at_0.5": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} "
            f"loss={train_loss:.4f}/{val_metrics['loss']:.4f} "
            f"F1={val_metrics['f1']:.4f} "
            f"AUPRC={val_metrics['auprc']:.4f} "
            f"AUROC={val_metrics['auroc']:.4f}",
            flush=True,
        )

        if val_metrics["auprc"] > best_auprc + 1e-6:
            best_auprc = val_metrics["auprc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), state_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    save_csv(output_dir / "history.csv", history)
    plot_history(history, output_dir)

    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    metrics_at_05, y_true, probabilities = evaluate(model, val_loader, criterion)
    threshold, selected_metrics = best_f1_threshold(y_true, probabilities)
    auroc, auprc = ranking_metrics(y_true, probabilities)
    selected_metrics.update({"auroc": auroc, "auprc": auprc})

    event_rows, event_summary, enriched = event_evaluation(
        metadata, probabilities, threshold
    )
    save_csv(output_dir / "val_predictions.csv", enriched)
    save_csv(output_dir / "val_event_results.csv", event_rows)
    plot_confusion(selected_metrics, output_dir)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_mean": torch.from_numpy(mean.squeeze()).float(),
        "input_std": torch.from_numpy(std.squeeze()).float(),
        "feature_names": feature_names.tolist(),
        "threshold": threshold,
        "config": {
            "model": name,
            "cell_type": cell_type,
            "window_length": 32,
            "input_features": 103,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "bidirectional": False,
            "alarm_consecutive_frames": ALARM_CONSECUTIVE_FRAMES,
        },
    }
    torch.save(checkpoint, output_dir / f"{name}_best.pt")

    summary = {
        "protocol": {
            "model": name,
            "cell_type": cell_type,
            "seed": SEED,
            "train_subjects": [1, 3, 6, 7, 8, 9],
            "val_subjects": [4, 5],
            "test_used": False,
            "window_length": 32,
            "early_stopping_metric": "validation AUPRC",
            "threshold_selection": "maximum validation frame-level F1",
            "standardization": "train statistics only across windows and time",
            "alarm_consecutive_frames": ALARM_CONSECUTIVE_FRAMES,
            "caucafall_fps": CAUCAFALL_FPS,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "model_parameters": parameters,
        "best_epoch": best_epoch,
        "best_validation_auprc_during_training": best_auprc,
        "selected_threshold": threshold,
        "validation_metrics_at_0.5": metrics_at_05,
        "validation_metrics_at_selected_threshold": selected_metrics,
        "validation_event_metrics": event_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)

    base_data = load_base_data()
    metadata = read_csv(DATA_DIR / "val_window_metadata.csv")
    if len(metadata) != len(base_data[3]):
        raise ValueError("Validation metadata and labels have different lengths")

    summaries = {}
    for name, cell_type in EXPERIMENTS.items():
        summaries[name] = run_experiment(name, cell_type, base_data, metadata)

    (OUTPUT_ROOT / "comparison_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nCompleted GRU and LSTM baselines. Outputs: {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
