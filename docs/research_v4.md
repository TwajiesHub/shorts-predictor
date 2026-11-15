# V4 research, sources, and evaluation

Research and retrieval date: 15 November 2026. General search engines, arXiv,
YouTube support, and the GitHub REST API returned proxy 403 errors in this
machine. Public GitHub repository search and raw files were accessible. We used
those sources directly; this is not an exhaustive literature review.

## Sources reviewed

| Source | What was verified | Decision |
|---|---|---|
| [Hutto & Gilbert, VADER (ICWSM 2014), author repository](https://github.com/cjhutto/vaderSentiment) | Human-rated social-text valences; the README describes polarity/intensity ratings and the included MIT license permits reuse. | Collected 7,506 unique token/valence entries. Use cheap valence summaries, **not** the full VADER sentence analyzer. This is a candidate representation, not evidence that negativity causes Shorts views. |
| [YouTube Shorts Dataset](https://github.com/JamilAL01/YouTube-Shorts-Dataset) | README advertises over 100,000 Shorts with titles, dates, outcomes, descriptions, and CC BY-NC 4.0 terms. | Not merged. No verified compatible collection-time/channel-baseline protocol, broader niche, and noncommercial terms. No description text or video-description links were fetched. |
| [YouTube Shorts & TikTok Trends 2025](https://github.com/tarekmasryo/youtube-tiktok-trends-dataset-2025) | README explicitly identifies the data, creator handles, titles, and comments as synthetic. | Rejected for learning real performance priors. |
| [AI Slop Shorts Dataset](https://github.com/aibuttonfoundation/youtube-ai-slop-dataset) | README describes video IDs and AI-content labels, not title/performance supervision. | Rejected for this ranking task. |
| [Upworthy headline A/B replication](https://github.com/Yubo-Cao/upworthy-lora), linking the [Upworthy Research Archive](https://upworthy.natematias.com/) (Matias et al., Scientific Data, 2021) | The accessible replication README describes grouping headlines by shared image/experiment and warns that training loss need not track ranking. Its reported LLM experiment is not independent Shorts evidence. | Adopt the methodological caution: select on ranking, compare fixed-thumbnail inputs, bootstrap channels. No headline labels, LLM, or Upworthy weights imported; news CTR is not Shorts feed performance. The original bulk archive was not downloaded. |
| [Google 10,000 English words](https://github.com/first20hours/google-10000-english) | Frequency-list provenance described in README. | Not needed: word-length/readability proxies avoid another corpus and its distribution assumptions. |

Search queries included `youtube shorts dataset`, `upworthy research archive`,
`youtube popularity prediction`, `upworthy negativity`, and `headline complexity`.
GitHub repository search exposed the sources above. A search hit alone was not
accepted as evidence: the relevant source READMEs/license were read.

## Three lightweight candidate signal families

1. **Affect and title specificity/structure** (`language`): positive/negative
   valence density and strength; opening-position affect; negation, information-gap
   pronouns, contradiction, direct address, list formats, units/numbers, comparison
   wording, and cheap readability/truncation proxies. All are available in a draft.
   No hand-set assumption says one format must perform better.
2. **Historical title performance priors** (`peers`): a sparse word/bigram TF-IDF
   bank over the 3,130 fitting titles, a small Ridge title prior, and shrunken
   similarity-weighted neighborhood residuals, including same-channel peers and
   support counts. Niche information comes from the existing real Shorts fitting
   cohort, not a generic web power-word list. Only a few scalar features reach the
   shallow tree head; no nearest-neighbor video content is downloaded at inference.
3. **Channel/topic familiarity** (`novelty`): maximum and local title similarity,
   neighborhood familiarity, vocabulary coverage, and channel support. Tested but
   not selected. No unvalidated new thumbnail rules, synthetic outcomes, scraped
   trend counts, or fabricated posting-time histories were added.

`language` + `peers` won inner selection as a bundle. Language features alone
were worse; this does **not** establish an independent causal benefit from sentiment
or any particular word. The novelty block and its runtime features are disabled
in the shipped bundle.

## Data construction and leakage controls

The new external data is the tiny valence dictionary. Its upstream revision is
`44fc044cd877310ee8278a0eadf34bcd50d41d06`. Original-file SHA-256 checksums and
URLs are in `data/lexicons/vader_source.json`; attribution/license are in
`data/lexicons/LICENSE.vader.txt`. `python scripts/collect_v4_resources.py`
rebuilds the derived JSON with TLS and checksum verification. Inference embeds
all selected resources in the v4 bundle and makes no network requests.

Historical peer features are **label-derived**, so fitting rows use five-fold
out-of-fold features. Identical normalized titles share a fold. A video's own
label and labels from its duplicate-title group cannot enter its training prior.
Selection, calibration, TestRun, and outer held-out rows never enter the prior
bank, vocabulary, or Ridge fit. For held-out/inference rows, the fixed fitting bank
is used. The model has no access to those query rows' views, likes, or comments.

The original seed-42 split remains 3,912 training and 980 held-out rows. V4 reuses
v3's 3,130 fitting / 391 selection / 391 calibration IDs and its unchanged base
features. Six prespecified feature combinations were compared with the same
small 180-tree Huber-regression head; the existing v3 head is the baseline.
Selection uses within-channel pairwise accuracy and requires at least 0.5
percentage point lift before accepting additions. Selection froze before outer
evaluation. No candidate was promoted because of its outer score.

| Inner-selection candidate | Within-channel pair accuracy |
|---|---:|
| Unchanged v3 | 55.02% |
| Language only | 53.84% |
| Peer priors only | 56.35% |
| Novelty only | 55.19% |
| Language + peers (selected) | 56.46% |
| All three | 55.72% |

## Held-out results

All-pairs evaluation contains 479,648 unequal-target pairs; within-channel
contains 23,535. Prediction ties receive half credit and target ties are excluded.
All versions use the same rows and original age-adjusted target for comparison.
V4 fitting/calibration baselines use fitting outcomes only, as in v3. TestRun is
excluded from fitting, selection, and calibration.

| Version | All pairs | Within-channel pairs | Spearman |
|---|---:|---:|---:|
| v2 | 56.41% | 56.38% | 0.18930098198462808 |
| v3 | 57.84% | 58.00% | 0.22508926609182808 |
| v4 | 60.54% | 59.74% | 0.3037112176128674 |

For the **no-thumbnail input** path, v3 → v4 all-pairs accuracy is 53.90% →
58.84%; within-channel accuracy is 54.29% → 57.58%. This holds the thumbnail
inputs at the common imputed reference and better approximates title comparison
than comparing videos with different images. It still does not isolate title
causality or test alternate titles for the same actual video.

There are 42 normalized-title overlaps between fitting and outer rows. Excluding
those rows leaves 938 videos: v3 → v4 all-pairs accuracy is 57.35% → 60.27%,
and within-channel accuracy is 57.51% → 59.45%.

A paired channel bootstrap gives a 95% interval of **+0.06 to +3.35 percentage
points** for pooled within-channel lift over v3. Equal-channel macro lift has an
interval of **−3.49 to +4.34 points**: improvements are not established uniformly
across channels. The development holdout was already used during v3 development;
a fresh prospective evaluation is still needed.

Calibrated relative-score MAE is 1.1349125097297652, versus a zero-relative-score
reference of 1.1848854326992024. The calibration-set 80% error radius covers
76.43% of outer outcomes, so it must not be presented as guaranteed 80% future
coverage. Snapshot views, approximate dates, and modest samples remain limitations.

See `reports/metrics_v4.json` for full precision, selection ablations, confidence
intervals, overlap audit, and calibration. `python scripts/train_v4.py --evaluate-only`
reproduces the saved-model results in `reports/evaluation_v4.json` without training
or changing any model. `reports/benchmark_v4.json` records cold loading, warm CPU
latency with and without full-size thumbnails, memory, and model size.

CPU benchmark on this machine: about 81 ms median for three titles, 212 ms with
a shared thumbnail, and 426 ms with three 1920×1080 thumbnails, after loading.
Import/loading took about 2.8 seconds; model size was 7.7 MiB and peak benchmark
RSS was about 348 MiB. The benchmark asserts neither torch nor OpenCLIP is loaded.
These are machine-specific observations, not hard latency or memory guarantees.
