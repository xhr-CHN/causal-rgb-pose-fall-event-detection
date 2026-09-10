# Reproduction workflow

## 1. Verify and create the environment

Run `sha256sum -c SHA256SUMS.txt`, then create the recorded Python 3.9
environment from `environment/conda_environment.yml`. The exact pip snapshot
and captured GPU/CUDA record are retained for audit. Equivalent newer hardware
may change latency but should not change deterministic metric calculations.

## 2. Acquire datasets and preserve roles

Obtain CAUCAFall, URFD, Le2i, and GMDCSA24 from the official sources in
`DATASETS.md`. Do not redistribute their media with this repository.

The frozen roles are:

- CAUCAFall: source training/validation; Subjects 2 and 10 are the held-out
  binary-detector test.
- URFD: external development and diagnostic evaluation.
- Le2i: staged external evaluation and verifier-development evidence.
- GMDCSA24: one-time final blind test; never use it for model, threshold, or
  policy selection.

Use `splits/prepared_dataset_manifest.csv` to reconstruct the CAUCAFall
train/validation/test placement. Its paths are relative and contain no media.

## 3. Configure paths without changing the protocol

The files under `code/` are the exact executed scripts. They retain the
historical project root `/home/data/yoloA27`. Change that root to the local
installation path before execution. In configuration and result metadata,
`${PROJECT_ROOT}` denotes the same root.

Path edits are portability edits only. Do not change subject partitions,
dataset roles, causal window endpoints, validation-selected thresholds, the
selected W16 evidence policy, or the locked GMDCSA24 protocol/hashes.

## 4. Source detector and temporal features

Run these stages in order:

1. `train_baseline.py`
2. `extract_caucafall_pose_features_v1.py`
3. `extract_caucafall_rgb_roi_embeddings_v1.py`
4. `build_caucafall_pose_windows_v1.py`
5. `build_caucafall_pose_windows_3state_v1.py`
6. `build_caucafall_pose_windows_3state_kinematic_v2.py`
7. `build_caucafall_rgb_roi_windows_3state_v1.py`

Raw/cached feature arrays are excluded because they can be regenerated and may
inherit dataset redistribution restrictions.

## 5. Source models and ablations

Use the corresponding scripts for the reported comparisons:

- `train_pose_mlp_ablation_v1.py`
- `train_pose_rnn_baselines_v1.py`
- `train_pose_tcn_baseline_v1.py`
- `train_pose_tcn_feature_ablation_v1.py`
- `train_pose_tcn_3state_v1.py`
- `train_pose_tcn_3state_kinematic_v2.py`
- `train_rgb_roi_tcn_3state_v1.py`
- `train_quality_gated_logit_fusion_v3_w16.py`
- `run_quality_gated_fusion_w16_multiseed.py`

Compare regenerated summaries with matching directories under `results/`.

## 6. Source policy and external evaluation

Select and audit the source-derived policy with:

1. `search_w16_evidence_policy.py`
2. `evaluate_urfd_w16_evidence_policy.py`
3. `build_urfd_alarm_verifier_features.py`
4. `prepare_le2i_inventory_v2.py`
5. `extract_le2i_frozen_features_v2.py`
6. `run_le2i_frozen_inference_v1.py`
7. `build_multidataset_alarm_confirmation_features_v2.py`
8. `train_multidataset_alarm_verifier_v2.py`
9. `select_event_preserving_alarm_policy_v2.py`
10. `finalize_causal_rescue_alarm_policy_v2.py`

The complete 18-feature definition is in `VERIFIER_FEATURES.md`. GMDCSA24 must
not be accessed during these stages.

Run the fusion comparison with `external_fusion_ablation_urfd_le2i_v1.py`.
The reported conclusion is the observed marginal, dataset-dependent difference
between fixed and gated fusion, not uniform superiority of the learned gate.

## 7. Locked final blind evaluation

Use `lock_gmdcsa24_final_blind_protocol_v1.py` and verify the hashes under
`policies/` before final-dataset evaluation. Then run the GMDCSA24 feature,
frozen-inference, evaluation, failure-review, and post-hoc-statistics scripts.
The blind dataset must not feed back into training or policy selection.

The canonical final evaluator for the reported GMDCSA24 metrics is
`code/evaluate_gmdcsa24_final_blind_v1_fixed.py`. The unsuffixed
`code/evaluate_gmdcsa24_final_blind_v1.py` is retained for provenance only and
must not be used to reproduce the reported final metrics.

## 8. Held-out source test and latency audit

- `evaluate_caucafall_heldout_binary_v1.py` reproduces the 3,958-frame held-out
  binary test at the validation-locked threshold 0.623.
- `benchmark_deployment_rtx3090_v2.py` reproduces the model-core latency audit.
  Hardware, I/O, decoding, multi-person tracking, and alert-transport overhead
  must be reported separately for any deployment.

## 9. Acceptance checks

A reproduction is protocol-consistent when CAUCAFall roles match the manifest,
external datasets do not tune source models, GMDCSA24 hashes match before
evaluation, summaries agree within ordinary floating-point tolerance, and no
frames later than the declared 200 ms confirmation horizon are used.
