# Optional proxy score rules for v4/v5

This is pure post-processing. It reads the existing emotional cues, same-channel
prior, fitting history, and already-computed cheap thumbnail statistics. It never
fits or changes a model. `--apply-rules` is **off by default**.

```bash
python scripts/predict.py "Title A" "Title B" --model v5 --apply-rules --json
python scripts/predict.py "Title A" --model v4 --apply-rules
python scripts/evaluate_proxy_rules.py
python scripts/benchmark_proxy_rules.py
```

`adjusted_score = clip(R + sum(rule increments), -3, 3)`, where **R is the existing
calibrated relative `score`**, not raw `ranking_score`. Each rule applies once;
independent matching rules stack before clipping. The layer adds
`adjusted_score`, `adjusted_rank`, `rule_delta` (actual change after clipping),
`unclipped_rule_delta`, `rules_fired`, `rule_score_clipped`, and
`proxy_rule_context`. Original rows/order, `score`, `relative_score`, `rank`,
percentile, confidence, and rough view estimate are unchanged. Sort by
`adjusted_rank` explicitly if that ordering is desired. Adjusted scores are not
newly calibrated forecasts. V5's existing `emotional_triggers_used_for_ranking`
field still describes its **core** model; optional rules use those cues separately.

## Fixed thresholds and weights

A trigger fires when its existing v5 cue score is **greater than zero**. Overall
emotional strength is the **maximum** of the ten scores, in [0, 1]. High curiosity
and outrage both mean **at least 0.7**. Rules and context thresholds were not tuned
to held-out results.

| Condition | Increment to R |
|---|---:|
| Strength ≥0.7 and same-channel prior percentile ≥70 | +0.08 |
| Strength ≤0.3 and same-channel prior percentile ≤30 | −0.06 |
| Curiosity ≥0.7 and title length within the historical sweet spot | +0.05 |
| Outrage ≥0.7 and established channel outrage-style proxy is low | −0.04 |
| At least three triggers fire and measured thumbnail stats are poor | −0.03 |
| No triggers fire and title/average similarity is ≥0.8 | −0.04 |

Context is defined explicitly, with unknown gates **skipped**, not assumed true:

- **Channel prior percentile:** reuse the v4/v5 same-channel similarity-weighted
  title prior (the existing peer lookup's third scalar). Compare with stored
  **out-of-fold fitting** prior values from that channel, requiring five examples.
  Percentile is `100 * (count_below + 0.5 * count_equal) / n`. Thus constant tied
  priors are neutral at 50, not artificially high. This is not the niche percentile.
  Unseen channels, including TestRun in the fitting bank, have unknown prior percentile.
- **Sweet spot:** require at least 20 historical rows and five top-quartile
  performers. Select performers using existing age-adjusted fitting residuals;
  their title-length 25th–75th percentile is the inclusive character-length band.
  Constant outcomes or insufficient history leave the band unknown.
- **Outrage-style proxy:** no ground-truth toxicity labels exist. With 20 historical
  rows, classify the usual style as high when at least 20% of the latest 20 titles
  have any outrage cue. Otherwise classify it as low. This is **not a toxicity
  classifier**, nor evidence about a creator's conduct or audience.
- **Poor thumbnail:** at least two of: brightness outside [0.15, 0.85], contrast
  below 0.10, edge density below 0.02. All three observed measurements are required.
  Missing/imputed/no-thumbnail inputs are unknown and do not trigger this penalty.
- **Recent average:** normalized mean of the existing sparse TF-IDF vectors for
  up to 20 historical titles, requiring five. Cosine similarity ≥0.8 is very similar.
  Known timestamps at/after the query are excluded, as is the query video itself.
  Approximate dates remain flagged as approximate, not treated as exact chronology.

Evaluation uses only the existing fitting bank for history and references. Draft
inference can additionally reuse the existing local TestRun history already loaded
by the CLI; it is not added to the model or evaluation bank. Snapshot view counts
still cannot reconstruct what counts were at a historical posting time.

## Evaluation against frozen v5 predictions

Earlier v5 reports contained aggregates only. The evaluation script first
materialized predictions from the **unchanged saved model**, without fitting,
into `reports/heldout_predictions_v5.parquet`. Subsequent evaluation loads that
cache and verifies model, data, row order, labels, and prediction-file hashes.
`--refresh-predictions` explicitly rebuilds it if the model/data legitimately change.
The original R rankings reproduce published v5 metrics exactly.

980 videos; 479,648 unequal-target pairs and 23,535 within-channel pairs. Target
ties are excluded, prediction ties receive half credit. Changes below are
**percentage points**, not relative percentages.

| Metric | Original R | Adjusted R′ | Delta |
|---|---:|---:|---:|
| All pairs | 60.538% | 60.620% | +0.082 pp |
| Within-channel | 59.737% | 59.945% | +0.208 pp |
| No-thumbnail, all pairs | 58.844% | 58.950% | +0.106 pp |
| No-thumbnail, within-channel | 57.582% | 57.740% | +0.157 pp |

Only two rules fired: high-emotion/high-prior on **2 rows**, and low-emotion/low-prior
on **289 rows**. Four further rows changed through clipping; **295 rows** changed
in total. The other four rules had zero firing coverage on this split, so these
results do not establish their ranking benefit; unit tests exercise their behavior.
Clipping-only control results are included in the report and slightly reduce accuracy.

The pooled within-channel channel-bootstrap 95% lift interval is +0.012 to +0.431
pp with thumbnails, and −0.151 to +0.504 pp without thumbnails. Equal-channel macro
accuracy with thumbnails **falls by 0.125 pp**. These are tiny, mixed improvements
on an already reused development split, not a prospective or causal validation.
The layer therefore remains opt-in and is not promoted as a new calibrated model.

Full precision, rule counts, missing-context coverage, clipping controls, and
uncertainty are in `reports/proxy_rules_evaluation.json`. Per-row adjusted results
are separate in `reports/proxy_rules_predictions.parquet`; the frozen base cache
is never overwritten by rule adjustments.

## Runtime and compatibility

Paired CPU timings for three drafts: rules off/on **54.0/63.1 ms**; with three
full-size thumbnails **280.7/289.8 ms**. This adds about **9 ms per request** after
context setup. Initial context construction took 8.4 ms; the standalone API path
with a fresh context took 87.4 ms for three titles. The CLI initializes one context
and reuses it for the request. Results are machine-specific; see
`reports/proxy_rules_benchmark.json`. No extra image extraction, heavy dependency,
torch, or CLIP is introduced.

All five model-file SHA-256 hashes and saved default draft/existing-video outputs
were verified unchanged. The full suite passes 39 tests, including all six rule
branches, exact boundaries, clipping, unknown data, duplicate-title thumbnail
alignment, query-outcome isolation, and opt-in CLI behavior.
