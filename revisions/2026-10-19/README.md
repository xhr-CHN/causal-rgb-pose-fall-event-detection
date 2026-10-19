# October 2026 revision materials

This is an additive reproducibility supplement for **Cross-Dataset Fall-Event Detection via Causal RGB-Pose Temporal Modeling and Alarm Verification**. The manuscript's original [v1.3.0 URL](https://github.com/xhr-CHN/causal-rgb-pose-fall-event-detection/releases/tag/v1.3.0), tag and historical records remain unchanged.

## Source and data

- [Complete code snapshot](Revision_Analysis_Code_20261019.zip): 124 Python sources and 18 compact result summaries, together with reproduction instructions and checksums. Directory structure and submitted source bytes are preserved inside the ZIP, rather than flattened by browser file upload.
- [Revision release](https://github.com/xhr-CHN/causal-rgb-pose-fall-event-detection/releases/tag/revision-materials-20261019): the exact submitted Analysis S2 data ZIP, the code snapshot, the 12-page S2 PDF and asset checksums.
- [Reproduction instructions](REPRODUCTION.md).
- [Source-member manifest](CODE_SOURCE_MANIFEST.csv): paths refer to members inside the source/data archives, not to individually expanded files in this Git directory.
- [Data archive manifest](SOURCE_DATA_FILE_MANIFEST.csv) and [publication check summary](PUBLICATION_AUDIT.md).

The supplement covers fixed OmniFall Le2i labels and timestamp mapping, verifier correction and retrospective reassessment, paired fusion statistics, URFD exposure correction, verifier operating points, sensitivity studies and descriptive false-alarm analysis. It also contains exploratory source-only calibrated fusion, bounded-gate regularization, optimized constants, training-mean controls and frozen feature stress.

## Interpretation

URFD fixed/gated fusion both detect 28/30 events, with 24/26 false alarms. Under corrected Le2i labels, fixed fusion detects 122/130 with 103 false alarms; gated fusion detects 124/130 with 101. These comparisons do not establish a consistent learned-gate advantage.

The historical GMDCSA24 final result remains 60/79 detections and 14 false alarms (53.32/h over approximately 0.262573 h negative exposure). The corrected eight-feature pre-candidate verifier matches those aggregate counts without rescue, but changes individual errors; its reassessment is retrospective, not a new blind test.

The exploratory source CV reuses six people and 29 falls. At the primary 0.85 point the calibrated bounded zero-anchor head changes the original-structure reference from 25 detections/10 false alarms to 27/4; at 0.90 it changes 25/9 to 29/1, with increased conditional delay. Calibrated equal fusion reaches the same detection/alarm counts, and optimized constants closely match local timing gains. Repeated seeds are not additional independent events; feature stress is not physical-camera or deployment validation.

No raw participant media, full source feature tensors, full fold-specific temporal checkpoints or confidential reviewer/editor correspondence is supplied. Dataset-derived artifacts remain subject to provider terms. Historical paths in archived procedures are provenance, not credentials or portable installation paths. This supplement supports saved-prediction reconstruction, not a claim that the full training history was independently repeated.
