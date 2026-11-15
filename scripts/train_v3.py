#!/usr/bin/env python3
"""Train/evaluate v3 on the original seed-42 split; TestRun never enters fitting.

Inner validation alone fits score and pair calibration. The outer set is report-only.
Run with --evaluate-only to reproduce evaluation of the saved bundle.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import VIDEOS_PATH, THUMB_FEATURES_PATH, TESTRUN_CHANNEL_ID
from src.features import eligible_training_mask, fill_channel_medians, add_targets
from src.ranking import RankFeatures, PairwiseRanker, sample_pairs, attach_thumbnails, predict_frame
from scripts.train import channel_stratified_split, latin_letter_share, apply_age_effect


def load_data():
    df = pd.read_parquet(VIDEOS_PATH)
    eligible = eligible_training_mask(df)
    df = add_targets(fill_channel_medians(df, eligible))
    v1 = joblib.load(ROOT / 'models/model_v1.joblib')
    df['target'] = apply_age_effect(df.rel_log_views.to_numpy(), df.age_days.to_numpy(), v1['age_a'], v1['age_b'])
    keep = eligible & (df.channel_id != TESTRUN_CHANNEL_ID) & (df.title.map(latin_letter_share) >= .8)
    pool = df.loc[keep].copy()
    train, test = channel_stratified_split(pool, .2, 42)
    return df, pool.loc[train].copy(), pool.loc[test].copy(), v1


def exact_accuracy(y, p):
    i, j = np.triu_indices(len(y), 1)
    signs = np.sign(y[i] - y[j]); keep = signs != 0
    hit = ((np.sign(p[i]-p[j]) * signs + 1) / 2)[keep]  # ties get half credit
    return float(hit.mean()) if len(hit) else None, len(hit)


def evaluate(y, p, channels):
    acc, n = exact_accuracy(y, p)
    macro, correct, count = [], 0., 0
    for channel in np.unique(channels):
        idx = np.flatnonzero(channels == channel)
        a, pairs = exact_accuracy(y[idx], p[idx])
        if a is not None:
            macro.append(a); correct += a * pairs; count += pairs
    return dict(n=len(y), spearman=float(spearmanr(y, p).statistic),
                all_pairs_accuracy=acc, all_pairs=n,
                within_channel_accuracy=correct/count, within_channel_pairs=count,
                channel_macro_accuracy=float(np.mean(macro)))


def leakage_safe_target(frame, fitting, v1):
    # Baselines use fitting outcomes only, with self exclusion for fitting rows.
    histories = fitting.groupby('channel_id')
    meds = []
    for _, row in frame.iterrows():
        g = histories.get_group(row.channel_id) if row.channel_id in histories.groups else fitting
        g = g[g.video_id != row.video_id]
        meds.append(float(g.views.median()) if len(g) else float(fitting.views.median()))
    return apply_age_effect(np.log1p(frame.views.to_numpy())-np.log1p(meds),
                            frame.age_days.to_numpy(), v1['age_a'], v1['age_b'])


def legacy_predictions(frame, version):
    from scripts.predict import title_block, stored_thumb_block
    from src.features import build_numeric_frame
    b = joblib.load(ROOT / f'models/model_{version}.joblib')
    blocks = [build_numeric_frame(frame).to_numpy(float), title_block(b, frame.title.tolist())]
    if version == 'v2':
        blocks.append(stored_thumb_block(b, frame.video_id.astype(str).tolist()))
    model = b['mlp'] if b.get('deployed', 'mlp') == 'mlp' else b['ridge_full']
    return model.predict(b['scaler'].transform(np.hstack(blocks)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluate-only', action='store_true')
    ap.add_argument('--skip-ablations', action='store_true')
    ap.add_argument('--objective', choices=['auto', 'pairwise', 'regression'], default='auto',
                    help='auto selects regression or RankNet corrections using inner selection only')
    args = ap.parse_args()
    start = time.perf_counter()
    df, train, test, v1 = load_data()
    cache = pd.read_parquet(THUMB_FEATURES_PATH)
    train = attach_thumbnails(train, cache=cache)
    test = attach_thumbnails(test, cache=cache)
    path = ROOT / 'models/model_v3.joblib'
    fingerprint = hashlib.sha256(VIDEOS_PATH.read_bytes() + THUMB_FEATURES_PATH.read_bytes()).hexdigest()
    results = {}
    for version in ('v1', 'v2'):
        results[version] = evaluate(test.target.to_numpy(), legacy_predictions(test, version), test.channel_id.to_numpy())
    print('Legacy baselines:', json.dumps(results), flush=True)
    if args.evaluate_only:
        bundle = joblib.load(path)
        if bundle['data_fingerprint'] != fingerprint or bundle['test_ids'] != test.video_id.tolist():
            raise ValueError('Dataset/split changed: retrain before evaluating')
    else:
        inner_train, inner_val = channel_stratified_split(train, .2, 43)
        fitting = train.loc[inner_train].copy()
        held_inner = train.loc[inner_val].copy()
        select_idx, calibration_idx = channel_stratified_split(held_inner, .5, 44)
        selection = held_inner.loc[select_idx].copy()
        validation = held_inner.loc[calibration_idx].copy()
        # Learn the missing-thumbnail path too, rather than assuming every draft has one.
        from src.ranking import THUMB_COLS
        augmented = fitting.copy()
        mask = np.random.default_rng(42).random(len(augmented)) < .3
        augmented.loc[mask, THUMB_COLS] = np.nan
        features = RankFeatures().fit(fitting)
        X = features.transform(augmented)
        V = features.transform(validation)
        y = leakage_safe_target(fitting, fitting, v1)
        yv = leakage_safe_target(validation, fitting, v1)
        channels = fitting.channel_id.to_numpy()
        initial = GradientBoostingRegressor(n_estimators=180, max_depth=3, min_samples_leaf=35,
            learning_rate=.05, random_state=42, loss='huber').fit(X, y)
        # Compare uniform and gap-weighted pairwise losses on inner selection.
        # Calibration and outer rows are never used to select the objective.
        S = features.transform(selection)
        ys = leakage_safe_target(selection, fitting, v1)
        base_accuracy = evaluate(ys, initial.predict(S), selection.channel_id.to_numpy())['within_channel_accuracy']
        candidates = [(base_accuracy, 0, False)]
        rankers = {}
        for weighted in (False, True):
            candidate = PairwiseRanker(initial_model=initial, gap_weighted=weighted).fit(X, y, channels)
            rankers[weighted] = candidate
            current = initial.predict(S)
            for step, tree in enumerate(candidate.trees_, 1):
                current += candidate.learning_rate * tree.predict(S)
                if step in (30, 60, 120, 180):
                    candidates.append((evaluate(ys, current, selection.channel_id.to_numpy())['within_channel_accuracy'], step, weighted))
        eligible_candidates = candidates
        if args.objective == 'pairwise':
            eligible_candidates = [c for c in candidates if c[1] > 0]
        elif args.objective == 'regression':
            eligible_candidates = candidates[:1]
        best_accuracy, best_steps, best_weighted = max(eligible_candidates, key=lambda item: (item[0], -item[1]))
        ranker = rankers[best_weighted]
        ranker.trees_ = ranker.trees_[:best_steps]
        print('Inner selection (accuracy, correction trees, gap weighted):', candidates,
              'selected', (best_steps, best_weighted), flush=True)
        validation_raw = ranker.predict(V)
        # Positive affine calibration preserves the learned ordering; never calibrate
        # on the outer test. Report residual error rather than a false precise forecast.
        calibration = LinearRegression(positive=True).fit(validation_raw[:, None], yv)
        if calibration.coef_[0] == 0:
            print('WARNING: validation has no positive calibration slope', flush=True)
        i, j = sample_pairs(yv, validation.channel_id.to_numpy(), seed=77)
        gaps = validation_raw[i] - validation_raw[j]
        nonzero = gaps != 0
        gaps, i, j = gaps[nonzero], i[nonzero], j[nonzero]
        correct = (np.sign(gaps) == np.sign(yv[i]-yv[j])).astype(int)
        pair_calibration = LogisticRegression(C=1.).fit(abs(gaps).reshape(-1, 1), correct)
        # If observed reliability decreases with gap, report the constant base rate.
        if pair_calibration.coef_[0, 0] < 0:
            pair_calibration.coef_[:] = 0
            rate = np.clip(correct.mean(), .001, .999)
            pair_calibration.intercept_[:] = np.log(rate/(1-rate))
        bundle = dict(version='v3', deployed='hybrid_ranknet' if best_steps else 'regression_trees', features=features, ranker=ranker,
                      calibration=calibration, pair_calibration=pair_calibration,
                      reference_rank_scores=np.sort(validation_raw),
                      relative_error_80pct=float(np.quantile(abs(yv-calibration.predict(validation_raw[:, None])), .8)),
                      title_length_p99=float(fitting.title.str.len().quantile(.99)),
                      age_a=v1['age_a'], age_b=v1['age_b'],
                      testrun_channel_id=TESTRUN_CHANNEL_ID,
                      testrun_subscribers=float(df.loc[df.channel_id == TESTRUN_CHANNEL_ID, 'channel_subscribers'].median()),
                      data_fingerprint=fingerprint, test_ids=test.video_id.tolist(),
                      fitting_ids=fitting.video_id.tolist(), validation_ids=validation.video_id.tolist(),
                      selection_ids=selection.video_id.tolist(), correction_trees=best_steps, selection_candidates=candidates, gap_weighted=best_weighted, objective=args.objective)
        path.parent.mkdir(exist_ok=True)
        joblib.dump(bundle, path, compress=3)
        results['regression_same_features'] = evaluate(test.target.to_numpy(),
            initial.predict(features.transform(test)),
            test.channel_id.to_numpy())
        print('Regression ablation:', json.dumps(results['regression_same_features']), flush=True)
        if not args.skip_ablations:
            for name, options in [('no_title_hash', dict(text=False)), ('no_thumbnail', dict(thumbnail=False)),
                                  ('no_channel_context', dict(context=False))]:
                f = RankFeatures(**options).fit(fitting)
                A = f.transform(augmented)
                initial_a = GradientBoostingRegressor(n_estimators=180, max_depth=3, min_samples_leaf=35,
                    learning_rate=.05, random_state=42, loss='huber').fit(A, y)
                # Same correction budget as full model; ablations never tune on outer test.
                m = PairwiseRanker(n_estimators=best_steps, initial_model=initial_a, gap_weighted=best_weighted).fit(A, y, channels)
                results[name] = evaluate(test.target.to_numpy(), m.predict(f.transform(test)), test.channel_id.to_numpy())
                print(name, json.dumps(results[name]), flush=True)
    raw, relative, _ = predict_frame(bundle, test)
    results['v3'] = evaluate(test.target.to_numpy(), raw, test.channel_id.to_numpy())
    fitting = df[df.video_id.isin(bundle['fitting_ids'])]
    yt = leakage_safe_target(test, fitting, v1)
    # Title comparison path: hold thumbnails fixed at the imputed reference for
    # every video, rather than crediting ranking changes to varying images.
    from src.ranking import THUMB_COLS
    title_only = test.copy()
    title_only[THUMB_COLS] = np.nan
    title_raw, _, _ = predict_frame(bundle, title_only)
    results['v3_without_thumbnail_input'] = evaluate(test.target.to_numpy(), title_raw, test.channel_id.to_numpy())
    legacy = joblib.load(ROOT / 'models/model_v2.joblib')
    from src.features import build_numeric_frame
    from scripts.predict import title_block, thumb_block_from_paths
    title_X = np.hstack([build_numeric_frame(test).to_numpy(float), title_block(legacy, test.title.tolist()),
                        thumb_block_from_paths(legacy, [None] * len(test))])
    legacy_model = legacy['mlp'] if legacy.get('deployed', 'mlp') == 'mlp' else legacy['ridge_full']
    results['v2_without_thumbnail_input'] = evaluate(test.target.to_numpy(),
        legacy_model.predict(legacy['scaler'].transform(title_X)), test.channel_id.to_numpy())
    # Empirical pair-probability quality on untouched within-channel test pairs.
    i, j = sample_pairs(yt, test.channel_id.to_numpy(), seed=81)
    gaps = raw[i]-raw[j]
    probability = bundle['pair_calibration'].predict_proba(abs(gaps).reshape(-1, 1))[:, 1]
    correct = (np.sign(gaps) == np.sign(yt[i]-yt[j])).astype(float)
    results['v3']['pair_reliability_brier'] = float(np.mean((probability-correct)**2))
    results['v3']['pair_reliability_mean'] = float(probability.mean())
    results['v3']['sampled_pair_actual_accuracy'] = float(correct.mean())
    results['v3']['calibrated_mae'] = float(np.mean(abs(yt-relative)))
    results['v3']['zero_relative_mae'] = float(np.mean(abs(yt)))
    results['v3']['empirical_80pct_error_coverage'] = float(np.mean(abs(yt-relative) <= bundle['relative_error_80pct']))
    # Paired channel bootstrap for lift: channels, not millions of dependent pairs,
    # are the resampling units. This is an uncertainty diagnostic, not a guarantee.
    base = legacy_predictions(test, 'v2')
    deltas = []
    for c in test.channel_id.unique():
        ix = np.flatnonzero(test.channel_id.to_numpy() == c)
        a, _ = exact_accuracy(test.target.to_numpy()[ix], raw[ix])
        b, _ = exact_accuracy(test.target.to_numpy()[ix], base[ix])
        if a is not None: deltas.append(a-b)
    rng = np.random.default_rng(12)
    boot = np.mean(rng.choice(deltas, size=(2000, len(deltas))), axis=1)
    results['channel_macro_lift_vs_v2_95pct_interval'] = np.quantile(boot, [.025, .975]).tolist()
    results['protocol'] = dict(outer_seed=42, outer_train=len(train), outer_test=len(test),
        fitting=len(bundle['fitting_ids']), calibration=len(bundle['validation_ids']), selection=len(bundle.get('selection_ids', [])),
        pairwise_correction_trees=bundle.get('correction_trees', 180),
        testrun_excluded=True, objective='inner-selected regression warm-start plus channel-balanced RankNet corrections',
        selected_head=bundle['deployed'], gap_weighted=bundle.get('gap_weighted', False),
        target='v1 age coefficients; training-only baseline for v3 fitting/calibration; legacy target for common ranking comparison',
        caveats=['Random per-channel split, not future performance validation',
                 'Development holdout reused across implementation iterations; fresh prospective validation still needed',
                 'Most timestamps approximate; recent exact history often unavailable',
                 'Snapshot views cannot reconstruct point-in-time channel outcomes',
                 'Legacy v1/v2 trained on more rows than v3, which reserves inner calibration rows'])
    results['inner_selection_candidates'] = bundle.get('selection_candidates', [])
    results['data_fingerprint'] = fingerprint
    results['runtime_seconds'] = time.perf_counter()-start
    report = ROOT / 'reports/metrics_v3.json'
    if args.evaluate_only:
        report = ROOT / 'reports/evaluation_v3.json'
    report.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
