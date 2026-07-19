# Alarm-verifier feature definitions

The selected verifier is a standardized logistic model using 18 causal
candidate-level features. The frozen feature order, means, scales,
coefficients, and decision threshold are stored in
`results/multidataset_alarm_verifier_v2_seed42/alarm_verifier_v2_model.json`.

## Timing and notation

- A candidate alarm occurs at sample `t`.
- The pre-candidate history contains 20 samples ending at `t`; if fewer are
  available, the earliest observed sample is repeated on the left.
- The confirmation interval contains samples `t` through `t+4` at the 20 Hz
  evaluation rate, adding at most 200 ms. If a sequence ends early, its last
  observed sample is repeated on the right.
- `delta` is last minus first value in the relevant interval.
- `slope` is the ordinary least-squares slope against sample index.
- Bounding-box coordinates and sizes are normalized in the upstream pose
  feature extraction.
- `fall score` is `P(Falling) + P(Fallen)`.
- `branch disagreement` is the frozen pose/RGB branch disagreement signal
  emitted by the fusion inference scripts.

## Eight pre-candidate features

1. `pre_person_conf_max`: maximum person-detection confidence in the 20-sample
   pre-history.
2. `pre_bbox_aspect_ratio_std`: population standard deviation of normalized
   bounding-box width divided by height in the pre-history; zero when height is
   invalid.
3. `pre_branch_disagreement_mean`: mean pose/RGB branch-disagreement signal in
   the pre-history.
4. `pre_bbox_cy_delta`: last minus first normalized bounding-box center-y value
   in the pre-history.
5. `pre_bbox_aspect_ratio_slope`: least-squares slope of bounding-box aspect
   ratio across the 20 pre-history samples.
6. `pre_rgb_fall_score_delta`: last minus first RGB-branch fall score in the
   pre-history.
7. `pre_torso_keypoint_conf_max`: maximum aggregate torso-keypoint confidence
   in the pre-history.
8. `pre_bbox_height_delta`: last minus first normalized bounding-box height in
   the pre-history.

## Ten confirmation features

9. `confirm_fused_fall_score_mean`: mean fused fall score over samples `t` to
   `t+4`.
10. `confirm_fused_fall_score_min`: minimum fused fall score over the same
    interval.
11. `confirm_fused_fall_score_delta`: fused fall score at `t+4` minus its value
    at `t`.
12. `confirm_fall_support_fraction_70`: fraction of the five confirmation
    samples whose fused fall score is at least 0.70.
13. `confirm_fall_longest_run_70`: longest consecutive run with fused fall
    score at least 0.70, divided by five.
14. `confirm_modal_agreement_fraction_50`: fraction of confirmation samples
    where both pose and RGB branch fall scores are at least 0.50.
15. `confirm_recovery_drop_from_candidate`: non-negative difference between
    the fused fall score at `t` and its minimum over the confirmation interval.
16. `confirm_fallen_rise`: `P(Fallen)` at the final confirmation sample minus
    `P(Fallen)` at the candidate sample.
17. `confirm_branch_disagreement_mean`: mean pose/RGB branch disagreement over
    the confirmation interval.
18. `confirm_pose_found_ratio_mean`: mean pose-found ratio over the
    confirmation interval.

## Selection boundary

The 18-feature configuration was selected from URFD and revealed Le2i
development evidence using deterministic five-fold group cross-validation,
with complete sequences kept within folds. GMDCSA24 was not read during
feature, model, or threshold selection.

Authoritative implementations are
`code/build_urfd_alarm_verifier_features.py`,
`code/build_multidataset_alarm_confirmation_features_v2.py`, and
`code/train_multidataset_alarm_verifier_v2.py`.

