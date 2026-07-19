# Public-release audit

Audit date: 2026-07-18

## Completed checks

- Archive transfer hash matched the server-generated SHA-256 value.
- Every candidate file passed its internal SHA-256 check before cleanup.
- All 50 Python scripts parsed with the Python 3.9 grammar.
- No raw image, video, cached embedding, or temporal-window array is included.
- No private key, API key, access token, credential-bearing URL, or email
  address was detected by the release scan.
- Personal conda/user paths were removed from public environment records.
- CAUCAFall split paths were converted to dataset-relative paths.
- Generated metadata paths use `${PROJECT_ROOT}` rather than the server root.
- The 18 verifier features are documented in `VERIFIER_FEATURES.md`.
- Official dataset sources and terms are documented in `DATASETS.md`.
- AGPL-3.0-only was selected because open-source Ultralytics components were
  used and the authors held no enterprise license.

## Preserved provenance

Python files under `code/` are the exact executed scripts and retain their
historical `/home/data/yoloA27` root constants. This is intentional and is
documented in `README.md` and `REPRODUCTION.md`; users configure those constants
locally without changing the frozen scientific protocol.

## Actions after repository creation

1. Add the public repository URL to `CITATION.cff`.
2. Add the article DOI after publication.
3. Create an immutable tagged release and archive it in Zenodo.
4. Replace provisional manuscript wording with the final GitHub/Zenodo URL.

