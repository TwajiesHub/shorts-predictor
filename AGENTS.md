# Agent notes

Lite neural-net YouTube Shorts performance predictor. Public metadata and public `i.ytimg.com` thumbnails only. Never fetch URLs from video descriptions, and never open Amazon or affiliate links.

## Layout

- `src/` — shared config and feature code (`features.py`, `thumb_features.py`, `yt_utils.py`)
- `scripts/` — collect, download thumbnails, extract features, train, predict
- `models/model_v1.joblib` and `models/model_v2.joblib` — trained bundles (include each once; do not add duplicate `models/v1/` or `models/model.joblib` copies)
- `models/cv/face_detection_yunet_2023mar.onnx` — YuNet weights used for face features
- `data/videos.parquet`, `data/channels.csv`, `data/thumb_features.parquet`, `data/thumb_embeddings.npz` — training tables (precomputed; images are not in git)
- `data/thumbnails/` — local JPEG cache, gitignored; recreate with `python scripts/download_thumbnails.py`
- `reports/` — held-out metrics and TestRun score tables
- `requirements.txt` — pinned versions used for the published metrics

The example target channel is TestRun (`@testru7`, `UCExmWKJ2B0tAC0FT3c0YHeg`). It must stay **held out of training**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CPU is enough. For CPU-only PyTorch wheels:

```bash
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
```

## Common commands

```bash
# Score draft titles (v2 is the default when models/model_v2.joblib exists)
python scripts/predict.py "Why Does My 128GB iPhone Only Show 113GB Free?" --duration 30 --model v2

# Compare thumbnails for one title
python scripts/predict.py "Example title" --thumbnail a.jpg --thumbnail b.jpg --model v2

# Score the held-out TestRun Shorts
python scripts/predict.py --score-existing --model v2

# Retrain v1 (does not need thumbnails)
python scripts/train.py

# Re-download public thumbnails, then retrain v2 (leaves v1 files untouched)
python scripts/download_thumbnails.py
python scripts/extract_thumb_features.py
python scripts/train_v2.py
```

`scripts/train_v2.py` must use the same per-channel held-out split (seed 42) and the same v1 age-effect coefficients so v1 / v2 / thumbnail-only metrics stay comparable. Do not invent metrics; write the exact Spearman and pairwise numbers the training script prints.

## Guardrails

- Do not add description text, affiliate URLs, API keys, tokens, or absolute machine paths to the repo.
- Do not commit `data/thumbnails/`, `data/testrun_thumbs/` images, `data/raw/`, `*_before_missing*`, `.venv/`, or files over 50 MB.
- Thumbnail fetches: `i.ytimg.com` only, polite `--sleep` (default 0.4s), refuse Amazon hosts.
- Prefer editing this repo in place. If you add a new model version, save it as `models/model_v3.joblib` and leave v1 and v2 intact.
