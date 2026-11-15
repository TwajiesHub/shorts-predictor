# Thumbnail palette experiment

## Result and scope

The **cached-color proxy experiment is negative**: overall held-out pairwise accuracy fell from **60.538% to 60.118% (−0.420 percentage points)**. These additions remain outside production. v1–v5 model files, prediction code, and default behavior are unchanged.

The requested **pixel-level top-five palette experiment is not yet measured**. No thumbnail JPEGs were present. Fetching `https://i.ytimg.com` returned a proxy tunnel 403. The authorized host addition was saved to the environment configuration draft, but the running policy still does not include it. Nothing was published. Existing scalar thumbnail statistics cannot reconstruct missing pixels or harmony; the cached experiment below must not be interpreted as evidence against the unmeasured pixel features.

## Reproduce

```bash
# Available now: cached-statistics distributions, proxy ablation, timings
python scripts/analyze_palettes.py --source cached
# After i.ytimg.com access is applied to the running environment:
python scripts/analyze_palettes.py --download --source pixels
# Reuse cached thumbnails, automatically choosing pixels at >=90% coverage:
python scripts/analyze_palettes.py
```

The downloader requests only public YouTube thumbnails, refuses redirects outside HTTPS `i.ytimg.com`, spaces request starts by at least 0.4 seconds, uses three workers, and stops on transport failure or rate limiting. A complete first collection needs 4,501 images and at least 30 minutes; fallback image requests can increase this. Existing JPEGs are reused. Images stay out of git. Collection and extraction record source URLs / image hashes where available. Current thumbnails can differ from those used for the original cached statistics.

Outputs: `reports/palette_analysis_cached.json`, `reports/palette_distributions_cached.csv`; a successful pixel study instead writes `_pixels` reports plus `data/palette_features_v1.parquet` containing numeric features and swatch metadata. The JSON includes exact feature names, thresholds, fitting/held-out IDs, coverage, distributions, permutation repetitions, and model/data fingerprints. No production model is saved.

## Implemented pixel extractor (50 features; not yet evaluated on corpus)

`src/palette_features.py` uses existing NumPy, OpenCV and Pillow dependencies. JPEG draft decoding and resize bound the longest side to 128 pixels. Existing letterbox masks crop only when both remaining dimensions are at least eight pixels and at least half the image area remains. RGB quantization uses four bits per channel (4,096 bins), selecting the five most frequent bins deterministically and computing their actual RGB centroids. No iterative clustering or neural model is needed.

The numeric vector contains:

- **19 global statistics:** circular mean hue sine/cosine and concentration (weighted by saturation × value); mean saturation, HSV value and grayscale brightness; existing colorfulness statistic; warm/cool fractions and smoothed ratio; high-saturation and vivid fractions; red/yellow accent fractions and presence; effective color count (exponential quantization entropy), top-five coverage, retained crop fraction.
- **25 dominant-color statistics:** saturation-weighted hue sine/cosine, saturation, value and pixel fraction for each of five colors. Missing colors are zero padded. RGB/hex swatches are also recorded as descriptive metadata.
- **6 pair proxies:** complementary and analogous scores, chromatic pair mass, maximum brightness contrast and RGB distance, and high-contrast pair mass. Scores are frequency weighted. These are geometric proxies, not established causal measures of visual appeal.

Hue is circular. Warm means <60° or >=330°; cool means 90°–270°. High saturation is S>=0.65; vivid also requires V>=0.5. Red is <=20° or >=340°, yellow is 40°–75°, both vivid; accent presence requires >=2% pixels. Harmony uses swatches with >=2% mass, S>=0.35 and V>=0.2: complementary closeness decreases to zero 30° away from 180° separation, analogous closeness decreases to zero at 45°. Pair mass is 2*p_i*p_j. High contrast includes neutral colors and requires grayscale contrast >=0.5 or normalized RGB distance >=0.65. All exact constants are in the JSON.

Synthetic timing only: full 1920×1080 JPEG decode plus extraction median **23.42 ms**, p95 **34.08 ms** (30 measurements). Predecoded 128×72 RGB extraction median **1.44 ms**. These are fixture timings, not measurements on downloaded thumbnails. Production inference cost is unchanged because extraction is not integrated into v1–v5.

## Available-data distribution analysis

We compared within-channel bottom and top performance quartiles, requiring >=8 observations, varying labels and >=2 examples per quartile. Held-out comparisons cover 22 channels and 245 examples per quartile. Labels are existing channel-relative performance targets, not CTR. We report pooled means for description, equal-channel high-minus-low differences, 2,000 channel bootstrap intervals, Wilcoxon signed-rank tests and Benjamini–Hochberg FDR across the 23 tested statistics per split. A consistent finding must have matching fitting/held-out signs, held-out q<0.05 and a bootstrap interval excluding zero.

| Existing statistic | High quartile mean | Low quartile mean | Equal-channel difference [95% interval] |
|---|---:|---:|---:|
| Saturation | 0.328 | 0.308 | +0.003 [−0.030, +0.033] |
| Brightness | 0.504 | 0.516 | −0.014 [−0.035, +0.010] |
| Colorfulness | 57.427 | 56.292 | −0.179 [−5.730, +5.016] |
| Warm pixel fraction | 0.249 | 0.247 | −0.003 [−0.037, +0.026] |

**No statistic passed the consistency criteria.** Pooled saturation differs slightly, but the channel-balanced comparison does not support a general recommendation. Red/yellow dominant-color proxies also provide no consistent evidence. Top-five harmony and true accent pixel fractions remain unmeasured.

## Temporary-model ablation

The available-data variant adds 12 deterministic encodings of existing statistics plus an availability flag: circular dominant hue, dominant chroma and visual mass, dominant red/yellow chromatic mass proxies, warm×saturation, normalized saturation dispersion, saturation×dark/bright fractions, dominant neutral mass and bright mass. These are transformations of cached scalar features, not new image observations. Exact names are saved in JSON.

We use the saved v5 split: **3,130 fitting, 391 selection, 391 reserved calibration and 980 held-out examples**, seed 42; TestRun is excluded. The original data fingerprint and held-out IDs are checked. Frozen cross-fitted training priors, fitting-only palette imputation, identical seed-42 30% thumbnail dropout, and the same fixed 180-tree head are used. Retraining the control reproduces saved baseline predictions exactly (maximum absolute error zero). The held-out target and pairwise evaluator are unchanged.

| Evaluation | Original / without added proxies | With added proxies | Change (pp) |
|---|---:|---:|---:|
| Held-out all pairs | 60.538% | 60.118% | −0.420 |
| Held-out within channel | 59.737% | 59.694% | −0.042 |
| No thumbnail, all pairs | 58.844% | 58.424% | −0.420 |
| No thumbnail, within channel | 57.582% | 57.610% | +0.028 |
| Selection, all pairs | 56.313% | 56.054% | −0.258 |

Removing the added block therefore **improves overall accuracy by 0.420 pp**. Ten joint feature-block permutations reduce the augmented model's accuracy by 0.155 pp overall and 0.440 pp within channel on average; use by the fitted model does not imply net benefit. The within-channel candidate-minus-control channel-bootstrap interval is [−0.504, +0.419] pp.

No optional inference flag or v-next artifact is introduced. A future actual-pixel candidate must meet the requested >=0.3 pp overall lift, also pass the selection threshold, and have >=90% image coverage per fitting/selection/held-out partition before eligibility is reported. Even then this reused development holdout is not fresh confirmation; the script never deploys automatically. Missing images do not qualify as a completed pixel-palette study.
