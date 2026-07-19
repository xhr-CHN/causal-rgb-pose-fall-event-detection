# Causal RGB-Pose Fall-Event Detection

Reproducibility materials for the manuscript **Cross-Dataset Fall-Event
Detection via Causal RGB-Pose Temporal Modeling and Alarm Verification**.

This release contains the executed training, evaluation, ablation,
alarm-policy, statistical-analysis, and latency-benchmark scripts; derived
annotations and split manifests; locked policies and protocol hashes; compact
result summaries; and selected final checkpoints.

## Scientific scope

The work evaluates a causal RGB-pose temporal framework at both frame and
event level. The external fusion ablation is included in full: quality-gated
fusion produced only marginal, dataset-dependent differences from fixed
late-logit fusion and is not claimed to be uniformly superior.

Key frozen evaluations are:

- CAUCAFall held-out source binary test: 3,958 frames; positive-class F1
  0.965, AUROC 0.998, and positive-class AUPRC 0.996 at the
  source-validation-locked threshold 0.623.
- URFD/Le2i external fusion ablation: pose-only, RGB-only, fixed 0.5, and
  quality-gated variants evaluated under the same event protocol.
- One-time GMDCSA24 blind test: 60/79 fall events detected, 14 false alarms
  (53.32/h of negative exposure), and 1.25 s median detection delay for the
  final causal-rescue policy.

These results do not establish clinical effectiveness or unattended-home
deployment readiness.

## Repository contents

- `code/`: exact executed Python scripts. They retain the historical
  `/home/data/yoloA27` project-root constant for provenance; configure that
  constant for another installation without changing frozen roles, thresholds,
  window endpoints, or policies.
- `configs/`: original detector configuration plus a portable dataset template.
  `${PROJECT_ROOT}` marks metadata paths that must be resolved locally.
- `annotations/`: derived temporal annotations and the blind annotation audit.
- `splits/`: relative-path CAUCAFall source split manifest.
- `policies/`: selected policies, locked protocols, and SHA-256 lock records.
- `results/`: compact JSON/CSV summaries supporting the article and supplement.
- `checkpoints/`: selected compact final checkpoints.
- `environment/`: Python, conda, package, CUDA, and GPU records.
- `DATASETS.md`: official dataset sources, roles, and usage restrictions.
- `VERIFIER_FEATURES.md`: complete definitions of the 18 verifier inputs.
- `REPRODUCTION.md`: ordered reproduction workflow.
- `THIRD_PARTY_NOTICES.md`: software and checkpoint licensing notes.
- `FILE_MANIFEST.csv` and `SHA256SUMS.txt`: inventory and checksums.

Raw dataset images/videos, cached pose arrays, ROI embeddings, temporal-window
NPZ files, and per-frame prediction tables are not redistributed. Obtain each
dataset from its official provider and comply with its terms; see
`DATASETS.md`.

## Quick verification

From the repository root:

```bash
sha256sum -c SHA256SUMS.txt
```

Create the recorded environment with:

```bash
conda env create -f environment/conda_environment.yml
conda activate yolo
```

The exact package snapshot is also retained in `environment/pip_freeze.txt`.

## Reproduction entry points

1. Acquire the four datasets and reconstruct the relative CAUCAFall split
   using `splits/prepared_dataset_manifest.csv`.
2. Update only project/data path constants in the executed scripts.
3. Follow `REPRODUCTION.md` in order to build features, train models, select
   the source policy, and run frozen external evaluation.
4. Compare regenerated metrics with `results/` and verify frozen artifacts
   against `policies/` and `SHA256SUMS.txt`.

## License

The repository code and included checkpoints are released under the GNU Affero
General Public License v3.0 only (`AGPL-3.0-only`); see `LICENSE`. This choice
preserves compatibility with the Ultralytics AGPL-3.0 components used by the
pipeline. Dataset licenses remain with their original providers, and users
must independently comply with all attribution and non-commercial conditions.
See `DATASETS.md` and `THIRD_PARTY_NOTICES.md`.

