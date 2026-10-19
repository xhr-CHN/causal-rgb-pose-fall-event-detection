# Reconstructing revision metrics

Download `Analysis_S2_Source_Data.zip` from the revision release and extract it into a fresh directory. The archive contains the analysis procedures, cached records and intact result archives used by Analysis S2. Original Python scripts and directory structures are also available in `Revision_Analysis_Code_20261019.zip`.

From the extracted analysis-data root, using Python and NumPy:

```bash
python evidence/source_calibration_cv29_20261018/reproduce_calibrated_cv29_metrics.py --archive evidence/source_calibration_cv29_20261018/archives/Gate_Constant_Controls_CV29_Results_20261018.zip --all-policies
```

This reads saved predictions and checks 372 fold/policy records; it does not train or run neural inference. Replace the archive basename with `Gate_Modality_Stress_Side_Results_20261018.zip` to check saved stress records. Omitting `--all-policies` checks only the primary 0.85 point. The helper looks for `runtimes/stress/` next to itself; use `--runtime-path` when relocating files.

The recorded model-work environment used Python 3.9.25, NumPy 1.26.4 and PyTorch 2.1.0+cu121/cuDNN 8902. Other procedures may require additional recorded scientific packages, licensed datasets, source features and separately archived branch checkpoints. Read each script's protocol before executing it; fixed server paths and no-overwrite guards remain for provenance. Do not reuse historical output directories or call new work the original blind evaluation.

Verify assets with the release `SHA256SUMS.txt`. The code ZIP has its own file manifest. Full training reproduction and independent validation are not claimed by a successful saved-prediction check.
