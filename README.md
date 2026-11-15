# Shorts performance predictor

A lite neural net that scores a **draft YouTube Shorts title** (plus duration, planned post time, and optionally a thumbnail) before you publish. The score is a predicted log-views residual versus that channel's typical Short, with a simple age adjustment so newer and older videos can be compared. The CLI also reports a **niche percentile** and a rough 14-day implied view count, using the held-out example channel's own median.

This is a small, reusable pipeline: collect public metadata, train a compact model, and score drafts locally. It is not a guaranteed view forecast.

Two model versions ship in this repo:

- **v1** — title text + numeric features only (`models/model_v1.joblib`)
- **v2** (default) — v1 features plus thumbnail embeddings and interpretable image traits (`models/model_v2.joblib`)

The example target channel, [TestRun](https://www.youtube.com/@testru7) (`UCExmWKJ2B0tAC0FT3c0YHeg`), was **held out of training**. Public view counts in the reports are as collected on 15 Nov 2026.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional, if you want CPU-only PyTorch wheels:
# pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
```

Python 3.11+ is recommended. Training and CLIP feature extraction run on CPU.

## Predict

```bash
# Score one or more draft titles.
# Times are Africa/Lagos unless you pass a timezone offset.
# Without --thumbnail, v2 uses the training-set average thumbnail and says so.
python scripts/predict.py \
  "Why Does My 128GB iPhone Only Show 113GB Free?" \
  "Your 128GB iPhone Is Lying About Its Storage" \
  "Why 128GB on iPhone Isn't Really 128GB" \
  --duration 30 \
  --when "2026-11-16 10:00" \
  --model v2

# Compare several thumbnails for one title (local path or i.ytimg.com URL).
# Amazon / affiliate URLs are refused.
python scripts/predict.py \
  "Why Does a 1TB Drive Only Show 931GB? #Shorts" \
  --thumbnail path/to/hook.jpg \
  --thumbnail path/to/other-frame.jpg \
  --duration 30 \
  --when "2026-11-16 10:00" \
  --model v2

# Score TestRun's existing Shorts with their stored thumbnail features
python scripts/predict.py --score-existing --model v2

# Title-only v1
python scripts/predict.py --score-existing --model v1
```

## Download thumbnails (optional, needed to retrain v2)

Thumbnail **images** are not in the repo (thousands of JPEGs). Precomputed features and CLIP embeddings are included, so you can score titles with v2 and retrain v2 without re-downloading. To fetch the public images from `i.ytimg.com` again (polite rate limits, no login, no description or Amazon URLs):

```bash
python scripts/download_thumbnails.py
# Optional: also save TestRun hqdefault hooks under data/testrun_thumbs/
python scripts/download_thumbnails.py --testrun-hooks
python scripts/extract_thumb_features.py
```

The downloader prefers `oar2.jpg` (often a 1080×1920 Shorts preview, which may be an auto-picked mid-video frame) over letterboxed `hqdefault.jpg`.

## Train

Rebuild from public yt-dlp / RSS metadata only. Rate limits are conservative. Description and affiliate URLs are never fetched.

```bash
python scripts/collect_data.py
python scripts/collect_missing.py   # recovers seed channels dropped in the first pass
python scripts/train.py             # writes models/model_v1.joblib
python scripts/download_thumbnails.py
python scripts/extract_thumb_features.py
python scripts/train_v2.py          # writes models/model_v2.joblib; leaves v1 untouched
```

## Features

**Known before posting (v1 and v2)**

- Title text (word + character TF-IDF, compressed with Truncated SVD)
- Title length, question / number / emoji / hashtag flags, “why / how / your” openings, topic keywords
- Duration
- Planned post hour and weekday, when a real timestamp is available (often it is not; see limitations)
- Channel subscriber count

**Thumbnail features (v2 only)**

- Public `i.ytimg.com` images only
- OpenCLIP ViT-B/32 (OpenAI weights) 512-d embeddings, compressed with PCA to 32 dimensions (51.9% of CLIP variance explained on the train set)
- Interpretable OpenCV traits: brightness, contrast, colorfulness, saturation, letterbox fraction, content aspect, edge density, a morphological text-region detector (not OCR), YuNet face detection, dominant HSV color, plus CLIP prompt similarities (text overlay, face, phone screen, code screen, product shot, dark/blurry, bright/bold)

**Target (unchanged between v1 and v2)**

- `log(views+1)` minus `log(channel median views+1)`, then minus a small age term fit only on the v1 training split and reused for v2
- Shorts younger than 3 days are excluded from training
- TestRun is excluded from training

## Held-out metrics

All v1 vs v2 vs thumbnail-only numbers use the **same per-channel random 20% held-out split** (seed 42) and the same age-adjusted target (v1 age coefficients reused). The comparison set is 3,912 train / 980 held-out videos from 32 channels. Every eligible row already had a thumbnail. TestRun is held out.

| Model | Held-out Spearman | Pairwise, all pairs | Pairwise, 8,000-pair sample | MAE | R² |
|---|---:|---:|---:|---:|---:|
| v1 MLP (title + numeric) | 0.14235487108832698 | 0.547691223563947 | 0.5456706706706707 | 1.1764662659016296 | 0.0038096532328687482 |
| Thumbnail-only MLP | 0.13582072949423024 | 0.5454103842818067 | 0.5362282567888875 | 1.18463830860578 | 0.016377919461472268 |
| Thumbnail-only Ridge | 0.12714228093815524 | 0.5420620955367269 | 0.5414267834793491 | 1.1682901748159646 | 0.00992910428141447 |
| v2 Ridge (title + numeric + thumb) | 0.1705905017408485 | 0.557583894856228 | 0.5553886594066841 | 1.1871290873861013 | −0.008188606572214985 |
| **v2 MLP (deployed)** | **0.18930098198462808** | **0.5641074287811061** | **0.5697834522468394** | 1.1750650379103653 | −0.0061629604780346625 |

Spearman p-values: v1 MLP 7.694016293289793e-06; v2 MLP 2.3353577520634047e-09; thumbnail-only MLP 1.9881405233179537e-05.

v2 MLP improves Spearman from 0.14235 to 0.18930 and all-pairs pairwise accuracy from 0.54769 to 0.56411. A thumbnail-only model is about as strong as v1, not stronger; the gain comes from **combining** title, numeric, and thumbnail features. MAE is essentially unchanged. R² is slightly worse for v2 MLP (−0.006 vs +0.004): thumbnails helped **ranking**, not absolute residual fit. 56.4% pairwise is only a bit above a coin flip.

The originally published v1 pairwise figure in `models/metrics_v1.json` was 0.5413693829014895. Spearman, MAE, and R² from the v2 re-evaluation match that file exactly; the pairwise gap is only the random 8,000-pair draw. Use the all-pairs column, or the v2-run 8,000-pair column, when comparing versions.

Full precision lives in `models/metrics_v1.json`, `models/metrics_v2.json`, `reports/train_summary_v1.json`, `reports/train_summary_v2.json`, and `reports/metrics_v2_exact_pairwise.json`.

## Dataset

Public yt-dlp playlist metadata and channel RSS, collected 15 Nov 2026 (no login, no YouTube API key). Thumbnails were downloaded the same day from `i.ytimg.com`.

| | Count |
|---|---:|
| Videos collected | 6,492 |
| Thumbnails downloaded in the original run | 6,492 (6,154 upgraded past first-pass hqdefault; 338 kept; 4,059 vertical) |
| Channels with at least one Short | 35 |
| Eligible English-script Shorts used for train/eval, excluding TestRun | 4,892 |
| TestRun short-form videos collected | 36 (33 current-series Shorts + 3 older non-series uploads) |

Likes, comments, tags, and description length were filled for a subset only and are **not** used as features (they are not known before posting). The channel table is `reports/channels_used.csv` and `data/channels.csv`.

Seed / comparable handles include `@testru7` (held out), `@zackdfilms`, `@athome-y1o`, `@FixWindowsNow`, `@PhoneRepairGuru`, `@MaviGadgets` (reach ceiling only), `@mavigadgetoriginals`, `@SpeedMcQueen`, `@yusuftechtalks`, and `@Keebo100`. `@athome-y1o` and `@FixWindowsNow` have no Shorts. Almost no training rows came from `@zackdfilms` or `@MaviGadgets` after duration/date backfill failed on the Shorts tab (complete rows were under 3 days old). Arya Vale and DATARK could not be resolved to a matching public channel. Some discovered channels are on-camera, bilingual, or official brand accounts, so the “faceless coding Shorts” niche is broader than ideal.

## Thumbnail traits linked to better residuals

Held-out associations only, each effect small. Shuffling the interpretable thumbnail block drops v2 Spearman by 0.08560 ± 0.01693, similar to numeric title metadata (0.08410) or title SVD (0.07897). The CLIP PCA block drops it by 0.05145 ± 0.01480.

1. Do not let one flat color take over the frame (`thumb_dom_frac` Spearman −0.0914910356071423, p=0.00415).
2. Avoid large blown-out bright patches (`thumb_bright_frac` is the top individual permutation feature).
3. Taller, vertical previews have a modest edge over wide 16:9 files.
4. Richer color and contrast help a little (mean saturation, contrast, CLIP “bright bold”).
5. Phone-screen and code-screen imagery fit this niche.
6. Text is mixed, and there was no real OCR. The morphological “text present” flag fires on 99.6% of images. A few clear hook words are fine; a heavy caption stack at the top is not clearly helpful.

Faces are mixed and not a recipe for a faceless channel. TestRun's public previews are typically a very dark navy field with one color owning most of the pixels; many mid-video `oar2` frames leave the lower third empty, while the Studio custom hook (visible in letterboxed `hqdefault`) is often the stronger question/payoff. Putting the hook on an early or key video frame, not only in the custom thumbnail slot, is the practical takeaway. v2 is **not** a reliable ranker of TestRun's already-observed views (current-series Spearman of the oar2 score vs raw views: −0.10596, p=0.557). Full tables: `reports/testrun_scores_v2.csv` and `reports/testrun_thumb_profile.csv`.

## Example drafts

Three title options for a “tech numbers that lie” Short (planned slot 16 Nov 2026, 10:00 Africa/Lagos, 30 seconds). v2 with the training-set average thumbnail still picks the first title; the spread is tighter than v1 because the thumbnail is held constant.

| Rank | v2 score | Niche pctl | Rough implied views at 14d | Title |
|---:|---:|---:|---:|---|
| 1 | +0.697 | 90 | ~222 | Why Does My 128GB iPhone Only Show 113GB Free? |
| 2 | +0.509 | 83 | ~184 | Your 128GB iPhone Is Lying About Its Storage |
| 3 | +0.383 | 77 | ~162 | Why 128GB on iPhone Isn't Really 128GB |

Treat the ranking as the useful part, not the exact view number.

## Layout

```
data/videos.parquet              # master video table
data/channels.csv
data/thumb_features.parquet      # precomputed interpretable + CLIP prompt features
data/thumb_embeddings.npz        # 512-d OpenCLIP embeddings
data/thumbnails/                 # not in git; re-download with scripts/download_thumbnails.py
models/model_v1.joblib
models/model_v2.joblib
models/metrics_v1.json
models/metrics_v2.json
models/cv/face_detection_yunet_2023mar.onnx
reports/                         # held-out metrics and TestRun score tables
scripts/collect_data.py
scripts/collect_missing.py
scripts/download_thumbnails.py
scripts/extract_thumb_features.py
scripts/train.py                 # v1
scripts/train_v2.py              # v2
scripts/predict.py
src/features.py
src/thumb_features.py
```

## Limitations

- **Predictive power is modest.** v2 is a real improvement on ranking, not a trustworthy point forecast. R² is near zero (slightly negative for the deployed v2 MLP).
- **Shorts thumbnails are often auto-picked frames.** Training used the best public Shorts-oriented image, usually `oar2.jpg`, which is frequently a mid-video frame rather than the Studio custom hook.
- **`hqdefault` is 480×360** and usually letterboxes a vertical custom hook.
- **No real OCR.** Text amount is a morphological region detector plus CLIP “text overlay” similarity.
- **Letterbox and content-aspect features can misfire** on very dark backgrounds (they may be treated as bars).
- **Face detection uses YuNet** (OpenCV 5 has no Haar `CascadeClassifier`).
- **Most upload dates are approximate.** yt-dlp's playlist `approximate_date` often snaps “N months/years ago” to the collection calendar day. Hour and weekday were treated as missing in those rows.
- YouTube Shorts can be up to 3 minutes; training allowed that for flagged Shorts, but the example channel's format is 25–30s.
- Affiliate / description URLs are never fetched or opened. Thumbnail URL fetches also refuse Amazon hosts.

## License

MIT. See `LICENSE`.
