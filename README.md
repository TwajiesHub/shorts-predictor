# Shorts performance predictor

A lite neural net that scores a **draft YouTube Shorts title** (plus duration, planned post time, and optionally a thumbnail) before you publish. The score is a predicted log-views residual versus that channel's typical Short, with a simple age adjustment so newer and older videos can be compared. The CLI also reports a **niche percentile** and a rough 14-day implied view count, using the held-out example channel's own median.

This is a small, reusable pipeline: collect public metadata, train a compact model, and score drafts locally. It is not a guaranteed view forecast.

Five compatible model versions ship in this repo:

- **v1** — title text + numeric features only (`models/model_v1.joblib`)
- **v2** (default) — v1 features plus thumbnail embeddings and interpretable image traits (`models/model_v2.joblib`)
- **v3** — lightweight title and thumbnail ranking features.
- **v4 / v5** (best validated ranking) — **60.54% held-out all-pair accuracy**; v5 preserves the v4 head because later additions did not pass selection. Use `python scripts/predict.py "Title A" "Title B" --model v5`. Optional proxy rules remain off by default.

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

## Lightweight ranking model (v3, opt-in)

Use the same CLI with `--model v3`; `auto`, v1, and v2 retain their existing behavior:

```bash
python scripts/predict.py "Why Is My iPhone Storage Missing?" "Your iPhone Storage Is Lying" --model v3 --duration 30
python scripts/predict.py "Example title" --model v3 --thumbnail a.jpg --thumbnail b.jpg --json
# Supply a known recent baseline for your channel instead of the default TestRun context:
python scripts/predict.py "Example title" --model v3 --channel-id MY_CHANNEL --subscribers 1000 --recent-median-views 500
python scripts/train_v3.py                  # train, select, calibrate, evaluate + ablations
python scripts/train_v3.py --evaluate-only  # evaluate saved v3 without refitting
python scripts/benchmark_v3.py              # CPU latency/memory, including image processing
python -m unittest discover -s tests -v
```

V3 adds a fixed-size word/bigram and character 3–5-gram hashing representation,
compressed to 64 dimensions, plus power-word counts, individual emoji counts,
number/question counts, length buckets, masked cyclical posting times, subscriber
tiers/interactions, and title-length/duration differences from channel history.
Baselines use fitting rows only, excluding the video itself. Recent history uses
up to 20 strictly earlier exact-dated rows; coarse dates use channel context without
claiming a chronology. The optional recent median overrides this fallback at inference.
TestRun stays excluded from training, selection, and calibration; its cached recent
context is used only for draft inference. Snapshot view counts cannot reconstruct
what a channel's baseline actually was at a past posting time.

Thumbnail processing uses existing cached brightness, contrast, faces, edge density,
and colorfulness, plus exposure-adjusted contrast, face prominence, edge/color
interaction, and usable-exposure fraction. It uses small OpenCV measurements only:
**no CLIP/torch loading or weight downloads in v3 inference**. Repeated image paths
are measured once per request; missing thumbnails are imputed, with matching dropout
during training. Existing v2 thumbnail/embedding caches and models remain intact.

The training code adds channel-balanced RankNet logistic pair loss on shallow trees,
including a gap-weighted variant that downweights near-tied outcomes. It starts from
a small robust regression head. A separate inner selection split chooses the number
of pairwise corrections, including zero; a disjoint calibration split fits a positive
affine relative-score mapping and an empirical pair-reliability model. In this dataset,
**selection rejected both pairwise variants and retained the regression head**.
The ranking loss is available explicitly with `python scripts/train_v3.py --objective pairwise`
(replaces only v3); it has not demonstrated a gain here. No new dependencies are needed.

`ranking_score` orders drafts; `score` / `relative_score` is the calibrated log-view
residual, and `niche_percentile` uses inner-validation predictions. `--json` also returns
empirical comparison reliability (against the top alternative, or runner-up for the
winner), an 80% validation absolute-error radius, and low-confidence reasons such as
close scores, missing thumbnails, little history, or an unseen channel. A single draft
has no pair reliability (`null`); identical drafts get 50%. These diagnostics and the
rough 14-day view translation are not guarantees of future views.

On this CPU, three drafts took about 46 ms after loading, or 250 ms with three
1920×1080 thumbnail files (about 1.8 s import/load, 348 MiB peak benchmark RSS,
7.25 MiB model). Measurements and the reproducible benchmark are in
`reports/benchmark_v3.json`; other machines and images will differ.

The seed-42 held-out set has 980 videos; v3 reserves 3,130 fitting, 391 selection,
and 391 calibration rows from the original 3,912 training rows. Exact pair accuracies:

| Model / input | All pairs | Within-channel pairs |
|---|---:|---:|
| v1 | 54.77% | 53.06% |
| v2 | 56.41% | 56.38% |
| v3 selected head | 57.84% | 58.00% |
| v2, no thumbnail input | 53.24% | 52.92% |
| v3, no thumbnail input | 53.90% | 54.29% |

These are **modest improvements**, not the substantial lift originally sought.
V3 Spearman is 0.22508926609182808 versus v2's 0.18930098198462808.
The no-thumbnail-input comparison holds thumbnail features at a common imputed
reference; it is a proxy for title comparison, not a randomized title experiment.
The outer development split was reused during implementation; fresh prospective
validation is still needed. Most dates are approximate, so this is not a future-time
validation. Full precision, calibration diagnostics, channel-bootstrap uncertainty,
objective selection, and retrained feature ablations are in `reports/metrics_v3.json`;
`--evaluate-only` writes `reports/evaluation_v3.json` and checks the dataset fingerprint.
Model selection does not use outer scores, and no ablation is silently promoted.

## Research-backed lightweight features (v4, opt-in)

V4 adds **sentiment/title-structure summaries** from a small
[VADER valence lexicon](https://github.com/cjhutto/vaderSentiment), plus
**cross-fitted historical title priors** from similar Shorts in the training set.
This uses sparse word/bigram similarity, a small Ridge prior, and a shallow tree
head. No LLM, new vision model, API call, or new dependency is needed at inference.
V1/v2/v3 remain available and `auto` still selects v2.

```bash
python scripts/predict.py "Title A" "Title B" --model v4
python scripts/predict.py "Title A" --thumbnail a.jpg --thumbnail b.jpg --model v4 --json
python scripts/train_v4.py                  # train-only selection, calibration, held-out comparison
python scripts/train_v4.py --evaluate-only  # reproduce saved-model evaluation
python scripts/benchmark_v4.py              # CPU timings, including thumbnail processing
```

On the same 980 held-out videos, **v3 → v4 all-pairs accuracy is 57.84% → 60.54%**;
within-channel accuracy is **58.00% → 59.74%**. Without thumbnail input, all-pairs
accuracy improves **53.90% → 58.84%**. Spearman rises from 0.22508926609182808 to
0.3037112176128674. Exact metrics, a repeated-title overlap audit, calibration,
and channel-bootstrap uncertainty are in `reports/metrics_v4.json`. This is a
reused development holdout; prospective gains are not yet established.
On this CPU, three drafts took about **81 ms** after loading, or **426 ms** with
three full-size thumbnails; import/loading took 2.8 seconds. The model is 7.7 MiB,
with about 348 MiB peak benchmark RSS and no torch/CLIP loaded. See
`reports/benchmark_v4.json` for the measurements.

We screened public datasets, including a [synthetic Shorts/TikTok dataset](https://github.com/tarekmasryo/youtube-tiktok-trends-dataset-2025)
and a [larger research Shorts dataset](https://github.com/JamilAL01/YouTube-Shorts-Dataset),
and did not merge incompatible outcomes. Novelty heuristics did not win selection.
Sentiment alone did not help; the selected language-plus-priors bundle did.
No universal "power word" or thumbnail-CTR claim is assumed.
See [research sources, exclusions, cross-fitting, and ablations](docs/research_v4.md).
The pinned external lexicon has its MIT license and checksum manifest under
`data/lexicons/`; `python scripts/collect_v4_resources.py` rebuilds it. Normal
prediction works offline from the bundled resources and retains v3's reliability
notes, calibrated relative score, percentile, and rough view estimate.

## Public-data and emotional-cue experiment (v5)

```bash
python scripts/predict.py "Title A" "Title B" --model v5 --json
python scripts/collect_v5_public.py --source snapshot
python scripts/train_v5.py
python scripts/train_v5.py --evaluate-only
python scripts/benchmark_v5.py
```

V5 implements expanded cross-fitted title priors and ten interpretable emotional
cue scores: curiosity, urgency/scarcity, FOMO, surprise, relatability, status,
humor, outrage, direct address, and numbers/lists. Editable regex weights and
normalization are in `data/lexicons/emotional_triggers_v5.json`. They approximate
title cues, not actual human reactions or guaranteed performance effects.

**No new ranking feature passed selection**, so shipped v5 keeps v4 predictions:
**60.54% all-pairs, 59.74% within-channel, and 58.84% all-pairs without thumbnail**.
JSON exposes the descriptive cue scores with
`emotional_triggers_used_for_ranking: false`. No accuracy gain is claimed.
The retrained emotional-only candidate gained 0.47 percentage point on inner
selection, below the unchanged 0.50-point retention threshold; pairwise corrections
also failed retention. V1–v4 and the `auto` default remain unchanged.

The public fallback supplied **542 additional examples / 120 channels** after
screening 68,699 records from [JamilAL01's Shorts snapshot](https://github.com/JamilAL01/YouTube-Shorts-Dataset).
They were published May 2023–April 2025, not recently collected 2026 counts.
No API key was configured and API/RSS access was proxy-blocked. Fresh API
collection requires `YOUTUBE_API_KEY` and access to `www.googleapis.com`; the
collector avoids search calls and bounds uploads/statistics requests. The retained
public data has **CC BY-NC 4.0** terms and attribution in `data/PUBLIC_DATA_V5.md`.

See [sources, exact collection counts, cue rules, API limits, and results](docs/research_v5.md),
`reports/metrics_v5.json`, and `reports/benchmark_v5.json`. The new priors are
available for controlled retraining but are inactive in the shipped ranking path.
On this CPU, shipped v5 took about **55 ms for three titles** or **272 ms with
three full-size thumbnails** after loading; import/loading took 2.3 seconds.
The 7.7 MiB bundle loaded no torch/CLIP; peak benchmark RSS was about 361 MiB.

## Optional proxy score adjustments (v4/v5)

```bash
python scripts/predict.py "Title A" "Title B" --model v5 --apply-rules --json
python scripts/evaluate_proxy_rules.py
python scripts/benchmark_proxy_rules.py
```

`--apply-rules` adds `adjusted_score`, `adjusted_rank`, matched rules, and proxy
context. It is **off by default** and never retrains or changes the core model.
Original scores, ranking/order, percentiles, confidence, and view estimates remain
unchanged. Adjusted scores are uncalibrated heuristic scores, clipped to **[−3, 3]**.

| Explicit proxy rule | Score increment |
|---|---:|
| Emotional strength ≥0.7 + channel prior ≥70th percentile | +0.08 |
| Emotional strength ≤0.3 + channel prior ≤30th percentile | −0.06 |
| High curiosity + length within channel's historical sweet spot | +0.05 |
| High outrage + low historical outrage-style proxy | −0.04 |
| At least three triggers + poor measured thumbnail stats | −0.03 |
| No triggers + very similar to recent channel-average title | −0.04 |

Strength is the maximum existing trigger score; “high” means ≥0.7. Channel prior
uses existing same-channel title priors and tied midpoint percentiles. Missing
context skips the corresponding gate; missing thumbnails are not labeled poor.
The outrage-style proxy is not a measured toxicity label.

| Held-out metric | Original | Adjusted | Change |
|---|---:|---:|---:|
| All pairs | 60.538% | 60.620% | +0.082 pp |
| Within-channel | 59.737% | 59.945% | +0.208 pp |
| No-thumbnail, all pairs | 58.844% | 58.950% | +0.106 pp |
| No-thumbnail, within-channel | 57.582% | 57.740% | +0.157 pp |

The gains are small and equal-channel macro accuracy falls slightly with
thumbnails, so the rules remain opt-in. Only the two emotion/prior gates fired in
this evaluation. Warm CPU overhead was about **9 ms for three drafts**, reusing
thumbnail stats. See [thresholds, proxy definitions, cache provenance, and limits](docs/proxy_rules.md),
`reports/proxy_rules_evaluation.json`, and `reports/proxy_rules_benchmark.json`.
The 980 frozen v5 predictions are cached for reproducibility; no models were changed.

## Feature importance and group ablation

```bash
python scripts/feature_importance.py --model v5
# Fast masking alternative (separate report):
python scripts/feature_importance.py --model v5 --ablation mask
```

The current v4/v5 head has 172 active columns. On the original 980 held-out videos:

| Removed group (retrained head) | Overall accuracy | Drop |
|---|---:|---:|
| Nothing — full model | 60.538% | — |
| Channel context + performance priors | 58.314% | 2.224 pp |
| Cheap thumbnail features | 58.771% | 1.767 pp |
| Title structure/readability | 59.667% | 0.871 pp |
| Subscriber size/interactions | 60.006% | 0.532 pp |
| Standalone title n-grams/SVD | 60.446% | 0.092 pp |
| Posting time | 60.470% | 0.068 pp |
| Sentiment lexicon | 60.511% | 0.027 pp |
| Duration | 60.540% | −0.002 pp |
| Explicit v5 emotional triggers | 60.538% | 0.000 pp — inactive |

Ten-repeat permutation importance also ranks **priors, cheap thumbnails, and title
structure** highest. Sentiment adds little; explicit v5 triggers, extra public-data
priors, novelty, and CLIP embeddings are absent from the shipped head. SVD removal
improves within-channel accuracy but hurts overall no-thumbnail accuracy: title
information still reaches the model through priors, so this is not a title-input
removal or a recommendation to drop text.

The analysis includes global/within-channel permutations, retrained and masked
ablations, native split importance, and auxiliary Ridge coefficients in
`reports/feature_importance_v5.json`. Full-head reproduction matched saved
predictions exactly; no production models or default behavior changed. Runtime
was about 2.5 minutes on CPU. See [feature groups, full results, split, and limitations](docs/feature_importance.md).

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

## Thumbnail palette research

Run `python scripts/analyze_palettes.py --source cached` for reproducible distributions and a temporary v5 ablation. Cached-color encodings reduced held-out all-pair accuracy **60.538% → 60.118% (−0.420 pp)**; they are excluded from prediction. No color statistic showed a consistent channel-balanced difference after multiple-testing checks.

A deterministic, CPU-only top-five palette extractor is implemented, but its real-image ranking evaluation remains pending: thumbnail files are absent and `i.ytimg.com` is still blocked in the running environment (the authorized allowlist change is saved in the environment draft). After applying access, run `python scripts/analyze_palettes.py --download --source pixels`. v1–v5 defaults remain unchanged. See [palette research notes](docs/palettes.md) for exact features, split, timing caveats and results; full numbers are in [JSON](reports/palette_analysis_cached.json) and [CSV](reports/palette_distributions_cached.csv).
