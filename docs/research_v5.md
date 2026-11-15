# V5: public data and emotional-trigger experiment

**Result:** the new implementations were tested but did not clear the preset
ranking-retention threshold. The shipped `--model v5` preserves v4 predictions.
Its JSON exposes descriptive emotional cue scores and explicitly reports
`emotional_triggers_used_for_ranking: false`. No accuracy gain is claimed.

## Public collection — 15 November 2026

No `YOUTUBE_API_KEY`, `YT_API_KEY`, or `GOOGLE_API_KEY` was configured. Managed
configuration had no credential bindings. Direct probes of YouTube Data API
(`www.googleapis.com`) and public RSS (`www.youtube.com`) both failed at the
network proxy with 403. **No authenticated API requests were made**; no API quota
or account usage could be inspected. Fresh/recent-2026 observations were not retrieved.

Fallback: [JamilAL01's public Shorts metadata](https://github.com/JamilAL01/YouTube-Shorts-Dataset),
revision `60d8f38921e43ec1eac227e2b89e2d0c12ec4d15`. The actual JSON contained
**68,699 records / 68,536,032 bytes**, despite the README advertising a larger
collection. SHA-256: `f767db96ddb148a6905a55543fb28db8fe53531cd0637691290e55fe0ead7f15`.
It is CC BY-NC 4.0; attribution and source terms are retained under `data/`.

**542 additional examples from 120 channels** survived filtering. Publication
range: **25 May 2023–8 April 2025**; durations 5–165 seconds; median views 12,308.
They match technical/science niches, not the original channel IDs. Original view
observation times are absent: `observed_at` remains null. They are an older
public snapshot, not fresh scraped performance. The upstream data can be
popularity-selected, and Latin-script filtering does not guarantee English.

Filtering rejected 446 invalid/duration rows; excluded 6,581 original IDs,
normalized titles, or TestRun matches **before computing any outcome baseline**;
removed 10,052 older source examples, 651 non-Latin-script titles, 39,314 rows
without five same-channel/quarter examples, and 11,113 outside matching niches.
No cap removed further rows. Only IDs, titles, channel, dates, duration, views,
category, source, and normalization fields are retained. No description text,
tags, or affiliate links are retained; description links are never followed.
Exact fields, filters, counts, source URL, and output hash are in
`reports/public_collection_v5.json`.

Candidate external labels are clipped log-views relative to their channel and
publication-quarter median, rather than inventing a collection age. Source rows
have weight 0.35. Expanded sparse title priors use five-fold group cross-fitting
across fitting and external titles; normalized duplicates share a fold. Selection,
calibration, outer rows, and TestRun cannot supply prior-bank outcomes. The active
v5 bank remains v4's because expanded priors failed retention.

## Emotional cue rules

The hand-authored, editable resource is `data/lexicons/emotional_triggers_v5.json`.
These are **editorial cue approximations, not measured human reactions or causal
view weights**. Generic mentions of a word need not evoke the named emotion.

| Cue | Representative patterns |
|---|---|
| Curiosity gap | why, how, secret, hidden, what happens, question mark |
| Urgency/scarcity | today, hurry, limited time, last chance, only N left |
| FOMO | don't miss, missing out, everyone is, need to know |
| Surprise/shock | unexpected, shocking, mind-blowing, wait for it |
| Relatability | we all, anyone else, me when, when you |
| Status/superiority | expert, smarter, genius, like a pro, level up |
| Humor | funny, bruh, lol, memes, 😂 / 🤣 |
| Controversy/outrage | scam, exposed, unfair, lying, stop believing |
| Direct address | you, your, you're, yourself |
| Numbers/lists | numeric tokens, leading numbers, N tips/reasons/steps |

Each regex has weight 0.5, 0.7, 1, or 1.5. Phrase weights reflect hand-assigned
cue salience, **not estimated view lift**. For each trigger, sum weighted matches
(each pattern capped at two matches), cap the sum at four, then divide by four.
Case is ignored and curly apostrophes normalized. Ten scores plus mean, maximum,
and nonzero fraction give 13 cheap features. For example, “Only 3 left today”
has urgency 0.625; “Only 128GB of storage” has urgency zero. No cue is added
directly to the view estimate; candidate ranking weights must be learned.

Coverage is sparse: the 3,130 fitting titles have only five FOMO, seven humor,
and 12 surprise/shock matches under these rules. Coverage is documented in
`reports/emotional_trigger_coverage_v5.json`; it limits what these samples can
establish about separate triggers. Curiosity and number cues already overlap v4.

## Selection, evaluation, and limits

The unchanged split is 3,130 fitting / 391 selection / 391 calibration / 980 outer
videos. Acceptance requires **+0.50 percentage point within-channel accuracy** on
selection before outer evaluation. The threshold was not relaxed after seeing results.

| Retrained-head inner ablation | Within-channel accuracy |
|---|---:|
| v4 baseline | 56.46% |
| Emotions only | 56.93% |
| Public priors only | 56.52% |
| Both | 56.65% |

The best gain, +0.47 point, missed retention. A second method froze v4 and fit a
small linear RankNet correction on the new features, with L2 values 1, 0.1, 0.01
selected on the same inner metric. It also failed retention. These exploratory
inner comparisons are not independent proof of a gain. Their results are in
`reports/experiment_v5_retrain.json` and `reports/metrics_v5.json`.

| Outer ranking metric | v4 | Shipped v5 |
|---|---:|---:|
| All unequal-target pairs | 60.54% | 60.54% |
| Within-channel pairs | 59.74% | 59.74% |
| No-thumbnail input, all pairs | 58.84% | 58.84% |
| No-thumbnail input, within-channel | 57.58% | 57.58% |

Ties receive half credit. Predictions match exactly because v5 uses the frozen
v4 head when no addition passes selection. The existing development holdout has
been reused; these measurements are not prospective causal title tests.

## Reproduction and API limits

```bash
python scripts/collect_v5_public.py --source snapshot
python scripts/train_v5.py                      # pairwise correction candidates
python scripts/train_v5.py --head retrain       # alternate whole-head ablations
python scripts/train_v5.py --evaluate-only
python scripts/predict.py "Title A" "Title B" --model v5 --json
python scripts/benchmark_v5.py
```

A fresh API run needs a securely configured `YOUTUBE_API_KEY` and network access
to `www.googleapis.com`; then use `--source api`. The API path is unit-tested with
mock responses, **not verified live**. It discovers uploads from existing channels,
batches statistics in groups of 50, filters to 5–60-second uploads aged 3–180 days,
and excludes TestRun. These uploads are short-form candidates, not proof of vertical
Shorts-feed membership. It avoids expensive `search.list` calls.

[YouTube quota reference](https://developers.google.com/youtube/v3/determine_quota_cost):
`channels.list`, `playlistItems.list`, and `videos.list` are budgeted at one unit
per request. The script uses one upload page per channel by default and permits
1–5 pages. Counts of successful list requests are recorded; failed requests may
also consume quota, and account-specific daily quota is not observable here.
Network failures, missing keys, and empty API results use the public snapshot in
`--source auto` mode. If no eligible records remain, collection stops explicitly.
The original dataset and v1–v4 artifacts are never overwritten.

Measured shipped-v5 CPU performance: 55 ms median for three titles, 129 ms with
a shared thumbnail, and 272 ms with three full-size local thumbnails; 2.3 seconds
import/loading, 7.7 MiB model, and about 361 MiB peak benchmark RSS. No torch or
OpenCLIP was loaded. These measurements include cue explanations but not inactive
public-prior inference; they describe the safe fallback, not a claimed faster new
ranking head. See `reports/benchmark_v5.json`.
