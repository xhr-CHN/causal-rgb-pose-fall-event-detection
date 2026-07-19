# Third-party and licensing notices

## Repository license

Unless a file states otherwise, the source code and included checkpoints in
this release are distributed under GNU Affero General Public License v3.0 only
(`AGPL-3.0-only`). See `LICENSE`.

Copyright (C) 2026 Haoran Xiao, Zehao Li, and Peishun Liu.

## Ultralytics

The pipeline uses the Ultralytics Python package and pretrained YOLO model
components. Ultralytics publishes its open-source code and models under
AGPL-3.0, with a separate enterprise licensing option. This release uses the
open-source route and therefore adopts AGPL-3.0. Ultralytics source code and
the pose foundation checkpoint are not copied into this repository.

- Project: https://github.com/ultralytics/ultralytics
- License: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- Version used: 8.4.57

## Dataset-derived artifacts

The repository license does not override dataset terms. The checkpoints,
annotations, split manifests, and result summaries were produced using the
datasets listed in `DATASETS.md`. Users must observe the attribution,
non-commercial, and share-alike requirements of the corresponding source
datasets. No permission to redistribute raw dataset media is granted here.

## Python dependencies

Python dependencies are not vendored. They remain under their respective
licenses. Exact package names and versions are recorded in
`environment/conda_environment.yml` and `environment/pip_freeze.txt`.

