# Feature importance and ablations of the shipped v4/v5 head

**Main result:** historical channel/performance priors, cheap thumbnail measurements,
and title structure/readability provide the clearest incremental ranking signal.
Explicit v5 emotional triggers are **not inputs to the shipped prediction head**.
No production feature, model, calibration, or default prediction was changed.

```bash
python scripts/feature_importance.py --model v5
# Same current head/features; v4 is also supported:
python scripts/feature_importance.py --model v4
# Faster conditional masking, without any analysis-head fitting:
python scripts/feature_importance.py --model v5 --ablation mask
```

Full analysis writes `reports/feature_importance_v5.json`. Mask mode writes a
separate `reports/feature_importance_v5_mask.json`; it does not overwrite the
retraining report. Optional `--permutation-repeats`, `--seed`, and `--output`
control shuffle repeats, shuffle seed, and destination. The default full run took
about **147 seconds**, reusing feature matrices with at most two numerical-library
threads. There are no new dependencies or image/model downloads.

## What was analyzed

The saved v5 head is the same 180-tree Huber `GradientBoostingRegressor` as v4;
their transformed features and predictions were checked equal. There are **172
active transformed columns**, assigned once to eight non-overlapping groups.
`src/feature_catalog.py` and the report list every name, index, group, and
cross-input dependency. Dimension and raw-input perturbation tests validate this
manifest against the actual saved transform.

| Group | Columns | Includes |
|---|---:|---|
| Title structure | 47 | Length, punctuation, numbers, emoji, topic/opening flags, linguistic formats, readability/case |
| Title n-grams | 64 | Saved SVD of hashed word/bigrams and character 3–5-grams |
| Sentiment lexicon | 8 | VADER summaries, negation, manual power-word count |
| Channel context and performance priors | 8 | Historical median/support, relative title length, cross-fitted Ridge/neighbor/same-channel priors and support |
| Cheap thumbnail statistics | 30 | Brightness, contrast, faces/area, edges, color, morphology/text/layout, geometry, derived exposure and observed indicator |
| Posting time | 8 | Hour, missing/coarse flags, cyclical hour/weekday, weekend |
| Subscriber size | 4 | Log subscribers, tier, tier×power-word and tier×title-length |
| Duration | 3 | Duration, log duration, ratio to historical channel duration |

**Inactive, zero columns:** the ten v5 emotional-trigger scores and their three
summaries, the extra public-source priors, v4 novelty candidates, and CLIP
embeddings/PCA/prompt similarities. The latter belong to v2, not this head.
Optional proxy-score rules are also **off** during this analysis. Inactive group
removals are structural no-ops, not fabricated permutation measurements.

## Held-out split and methods

The original eligible pool has 4,892 rows. Its seed-42 per-channel random split
has 3,912 training and **980 held-out** rows, excluding TestRun. The saved cohort
IDs partition training into **3,130 fitting / 391 selection / 391 calibration**.
The exact fitting and held-out IDs, input hash, and model hash are in the report.
The script reuses `scripts/train_v3.py`'s eligibility, target, and split functions
and refuses changed data or held-out order.

All ranking checks use the same original age-adjusted channel-relative held-out
target: **479,648** unequal-target pairs globally and **23,535** within-channel.
Prediction ties get half credit; target ties are excluded. The positive affine
relative-score calibration preserves raw-head ranking, so no calibration refit is
needed. Baselines reproduce the published v5 results exactly.

1. **Joint block permutation:** ten shuffles, seed 42, using the same shuffle
   orders for each group. All columns of a group move together. Both global row
   shuffles and shuffles restricted to each channel are reported, for both normal
   and no-thumbnail input. Standard deviations below describe shuffle variation,
   **not** confidence intervals over future videos.
2. **Leave-one-group-out retraining:** freeze the saved feature transforms and
   priors, remove the group's columns, and clone/refit only the head with the
   original parameters/seed on the same 3,130 fitting rows. Training uses saved
   group-fold **out-of-fold priors**, never in-sample target lookups, and the same
   seed-42 30% thumbnail dropout. No selection, calibration, or held-out labels
   train these temporary heads. A full-column retraining control matched the saved
   predictions with **maximum absolute difference 0.0**. Analysis heads are never saved.
3. **Median masking and native importance:** mask a group to fitting-column medians
   in the frozen model, and collect normalized split-impurity importances from the
   original head. Both are in JSON. Split importance is not a causal ranking gain.
   The auxiliary Ridge prior's top positive/negative TF-IDF coefficients are also
   provided; they are **not** final-model word effects or title-writing prescriptions.

## Leave-one-group-out results

Full model: **60.538% overall / 59.737% within-channel**. Drops are percentage
points; a negative drop means removal improved that metric.

| Removed group | Overall accuracy | Overall drop | Within-channel accuracy | Within drop |
|---|---:|---:|---:|---:|
| Channel context + priors | 58.314% | 2.224 | 57.213% | 2.524 |
| Cheap thumbnails | 58.771% | 1.767 | 57.604% | 2.133 |
| Title structure | 59.667% | 0.871 | 59.346% | 0.391 |
| Subscriber size/interactions | 60.006% | 0.532 | 59.363% | 0.374 |
| Standalone title n-grams/SVD | 60.446% | 0.092 | 61.003% | −1.266 |
| Posting time | 60.470% | 0.068 | 59.822% | −0.085 |
| Sentiment lexicon | 60.511% | 0.027 | 59.754% | −0.017 |
| Duration | 60.540% | −0.002 | 59.932% | −0.195 |
| Each inactive group | 60.538% | 0.000 | 59.737% | 0.000 |

The no-thumbnail baseline is **58.844% overall / 57.582% within-channel**. Removing
channel context/priors lowers overall accuracy to **55.155%**; removing title
structure gives **57.870%**, and removing standalone n-grams gives **57.956%**.
Thus n-gram removal has a different tradeoff without image input. The JSON contains
both views for every group. Removing all thumbnail columns and retraining is
not the same experiment as scoring the full model with its imputed thumbnail.

## Ranked importance and interpretation

Sorted by global-permutation drop in overall accuracy:

| Group | Mean drop ± shuffle SD (pp) | Native split share |
|---|---:|---:|
| Channel context + priors | 6.709 ± 0.625 | 49.67% |
| Cheap thumbnails | 2.742 ± 0.413 | 10.73% |
| Title structure | 0.995 ± 0.174 | 6.14% |
| Title n-grams/SVD | 0.396 ± 0.373 | 27.89% |
| Subscriber size/interactions | 0.286 ± 0.110 | 2.08% |
| Duration | 0.205 ± 0.116 | 2.52% |
| Posting time | 0.067 ± 0.024 | 0.18% |
| Sentiment lexicon | 0.039 ± 0.055 | 0.78% |

Within-channel permutation ranks **priors → thumbnails → structure → duration →
subscriber interactions → posting time → sentiment → standalone SVD**. Its mean
within-channel drops for the top three are **5.369, 2.405, 0.422 pp**. SVD has a
small negative mean drop (−0.183 ± 0.420 pp), consistent with redundancy/noise on
that metric, not proof of universally harmful title text.

- **Most useful:** context/priors and cheap image traits lead both permutation
  and retrained ablations; title structure is next. Historical baseline and
  same-channel neighbor residual alone account for 24.04% and 19.60% of split
  importance. In thumbnail traits, content aspect and height stand out.
- **Little incremental contribution:** explicit sentiment is close to neutral;
  time is weak, with only **109/3,130 fitting and 44/980 held-out** rows having a
  usable hour feature. Duration is used by the frozen model but mostly redundant
  after retraining. This does not prove posting time or duration never matters.
- **Possible redundancy/overfit:** SVD has large native importance but much weaker
  held-out incremental importance and mixed removal results. It still helps
  overall no-thumbnail ranking. No feature was removed from production.
- **Unused:** 64 of 172 columns have zero split importance in this fitted head.
  Examples include direct question/number/emoji flags, raw length counts,
  face-detected/count indicators, and manual power-word count. Derived length
  interactions and face-area features are nevertheless used. `topic_python` and
  `topic_javascript` are constant in the fitting matrix. Full lists are in JSON.

These are **conditional representation-block analyses**, not causal raw-input
knockouts. Text also feeds the performance priors, and cross-block interactions
remain unless assigned to the removed group. Global shuffles can create mismatched
feature combinations; within-channel shuffles preserve channel membership.
Correlated blocks can substitute for one another, impurity importance favors
some high-dimensional/continuous blocks, and dropping columns can change tree
split tie-breaking even with the same seed. Retraining uses one original head
seed. The existing development holdout has been reused; small gains or “harmful”
labels need fresh validation. Default v1–v5 behavior remains unchanged.
