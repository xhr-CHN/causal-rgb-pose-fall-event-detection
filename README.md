# Causal RGB-Pose Fall-Event Detection

> **October 2026 revision supplement:** see [revision materials](revisions/2026-10-19/README.md) for corrected Le2i labels, verifier reassessment, statistical analyses and exploratory source-only calibrated fusion. The historical [v1.3.0 release](https://github.com/xhr-CHN/causal-rgb-pose-fall-event-detection/releases/tag/v1.3.0) remains unchanged; the manuscript's existing URL remains valid. The supplement is not a replacement blind test or a newly validated deployment model.

Reproducibility materials for the manuscript **Cross-Dataset Fall-Event
Detection via Causal RGB-Pose Temporal Modeling and Alarm Verification**.

This release contains the executed training, evaluation, ablation,
alarm-policy, statistical-analysis, and latency-benchmark scripts; derived
annotations and split manifests; locked policies and protocol hashes; compact
result summaries; and selected final checkpoints.

Release v1.3.0 additionally includes the standalone Le2i baseline runner and
evaluator, plus the frozen OOPS feature-extraction, inference, and event-level
evaluation scripts used by the supplementary reproducibility materials.

## Scientific scope

The work evaluates a causal RGB-pose temporal framework at both frame and
event level. The external fusion ablation is included in full: quality-gated
fusion produced only marginal, dataset-dependent differences from fixed
late-logit fusion and is not claimed to be uniformly superior.

Historical baseline evaluations are listed below; apply the correction notes and the revision supplement for current label and exposure definitions:

- CAUCAFall held-out source binary test: 3,958 frames; positive-class F1
  0.965, AUROC 0.998, and positive-class AUPRC 0.996 at the
  source-validation-locked threshold 0.623.
- URFD/Le2i external fusion ablation: pose-only, RGB-only, fixed 0.5, and
  quality-gated variants evaluated under the same event protocol.
- URFD independent baseline reference: CAUCAFall-trained RGB-ROI TCN and
  Pose-TCN results are provided in `results/urfd_independent_baselines_v1/`;
  the old v1.3.0 reference used a mixed-frame 0.135 h denominator. The revision corrects common nominal full-sequence exposure to 0.094106389 h. This remains distinct from the historical fusion-ablation endpoint exposure of 0.080444444 h; cross-convention rate comparisons are not valid.
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
- `VERIFIER_FEATURES.md`: definitions of the historical 18 verifier inputs; the revision supplement separately documents the selected eight-feature pre-candidate verifier.
- `REPRODUCTION.md`: ordered reproduction workflow.
- `THIRD_PARTY_NOTICES.md`: software and checkpoint licensing notes.
- `FILE_MANIFEST.csv` and `SHA256SUMS.txt`: inventory and checksums; the two root inventory files exclude themselves to avoid self-referential checksums.

Raw dataset images/videos and full source feature tensors are not redistributed. The original baseline archive omits cached arrays and per-frame predictions; the separate revision supplement adds saved derived predictions and compact head-fitting records for metric reconstruction. Obtain original datasets from their providers and comply with their terms; see `DATASETS.md`.

## Quick verification

From the repository root:

```bash
sha256sum -c SHA256SUMS.txt
```

Checksums describe exact repository/archive bytes. On Windows, verify a GitHub source archive or a checkout with automatic line-ending conversion disabled. The revision folder has scoped Git attributes to retain its submitted source bytes; no historical tag is rewritten.

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

The standalone external workflows are:

- `code/run_le2i_standalone_baselines_v1.py` followed by
  `code/evaluate_le2i_standalone_baselines_v1.py` for the independent Le2i
  reference models.
- `code/extract_oops_frozen_features_v1.py`,
  `code/run_oops_frozen_inference_v1.py`, and
  `code/evaluate_oops_frozen_v1.py` for the prepared OOPS test subset.

## License

Source code authored for this repository is released under the GNU Affero
General Public License v3.0 only (`AGPL-3.0-only`); see `LICENSE`.
Dataset-derived annotations, result summaries, checkpoints, and other derived
artifacts are provided for scientific reproducibility and remain subject to
any applicable terms of the original datasets described in `DATASETS.md`.

This licensing choice preserves compatibility with the Ultralytics AGPL-3.0
components used by the pipeline. Dataset licenses remain with their original
providers, and users must independently comply with all applicable attribution,
non-commercial, share-alike, privacy, and other provider-specific conditions.
See `DATASETS.md` and `THIRD_PARTY_NOTICES.md`.
