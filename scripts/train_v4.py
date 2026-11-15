#!/usr/bin/env python3
"""Select lightweight research features on training-only validation, then evaluate.

Preserves v1/v2/v3 artifacts and the original 980-row seed-42 development holdout.
No held-out outcomes enter title priors, vocabulary, candidate selection, or calibration.
"""
import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_v3 import load_data, evaluate, leakage_safe_target, exact_accuracy, legacy_predictions
from src.config import THUMB_FEATURES_PATH, VIDEOS_PATH, TESTRUN_CHANNEL_ID
from src.ranking import attach_thumbnails, sample_pairs, predict_frame, THUMB_COLS
from src.research_features import ResearchFeatures, language_features, normalized_title


def data_fingerprint():
    return hashlib.sha256(VIDEOS_PATH.read_bytes()+THUMB_FEATURES_PATH.read_bytes()).hexdigest()


def by_ids(frame, ids):
    return frame.set_index('video_id', drop=False).loc[ids].reset_index(drop=True)


def paired_channel_interval(frame, baseline, candidate, seed=42):
    channels = []
    for channel, group in frame.groupby('channel_id'):
        ix = group.index.to_numpy()
        y = group.target.to_numpy()
        a, pairs = exact_accuracy(y, baseline[ix])
        b, _ = exact_accuracy(y, candidate[ix])
        if a is not None:
            channels.append([a*pairs, b*pairs, pairs])
    channels = np.asarray(channels)
    rng = np.random.default_rng(seed)
    sample = channels[rng.integers(len(channels), size=(2000, len(channels)))]
    pooled = (sample[:, :, 1].sum(axis=1)-sample[:, :, 0].sum(axis=1))/sample[:, :, 2].sum(axis=1)
    macro = ((sample[:, :, 1]-sample[:, :, 0])/sample[:, :, 2]).mean(axis=1)
    return dict(within_channel_lift_95pct=np.quantile(pooled, [.025, .975]).tolist(),
                channel_macro_lift_95pct=np.quantile(macro, [.025, .975]).tolist())


def fit_model(X, y):
    return GradientBoostingRegressor(n_estimators=180, max_depth=3, min_samples_leaf=35,
                                    learning_rate=.05, random_state=42, loss='huber').fit(X, y)


def calibrate(raw, y, channels):
    relative = LinearRegression(positive=True).fit(raw[:, None], y)
    i, j = sample_pairs(y, channels, seed=77)
    gaps = raw[i]-raw[j]
    keep = abs(gaps) > 1e-10
    i, j, gaps = i[keep], j[keep], gaps[keep]
    correct = (np.sign(gaps) == np.sign(y[i]-y[j])).astype(int)
    reliability = LogisticRegression(C=1.).fit(abs(gaps)[:, None], correct)
    if reliability.coef_[0, 0] < 0:
        reliability.coef_[:] = 0
        rate = np.clip(correct.mean(), .001, .999)
        reliability.intercept_[:] = np.log(rate/(1-rate))
    return relative, reliability


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluate-only', action='store_true')
    args = ap.parse_args()
    start = time.perf_counter()
    df, train, test, v1 = load_data()
    v3 = joblib.load(ROOT/'models/model_v3.joblib')
    cache = pd.read_parquet(THUMB_FEATURES_PATH)
    train = attach_thumbnails(train, cache=cache)
    test = attach_thumbnails(test, cache=cache)
    fingerprint = data_fingerprint()
    if v3['data_fingerprint'] != fingerprint:
        raise ValueError('Dataset changed since v3: regenerate comparable splits first')
    if test.video_id.tolist() != v3['test_ids']:
        raise ValueError('V3 held-out cohort changed')
    fitting = by_ids(train, v3['fitting_ids'])
    selection = by_ids(train, v3['selection_ids'])
    calibration = by_ids(train, v3['validation_ids'])
    y = leakage_safe_target(fitting, fitting, v1)
    path = ROOT/'models/model_v4.joblib'
    if args.evaluate_only:
        bundle = joblib.load(path)
        if bundle['data_fingerprint'] != fingerprint or bundle['test_ids'] != test.video_id.tolist():
            raise ValueError('Dataset/split differs from saved v4')
    else:
        full = ResearchFeatures(base=v3['features']).fit(fitting, y)
        augmented = fitting.copy()
        augmented.loc[np.random.default_rng(42).random(len(augmented)) < .3, THUMB_COLS] = np.nan
        X = full.training_matrix(augmented)
        S = full.transform(selection)
        ys = leakage_safe_target(selection, fitting, v1)
        base_width = v3['features'].transform(fitting.iloc[:1]).shape[1]
        lang_width = len(language_features('', {}))
        ranges = {'language': list(range(base_width, base_width+lang_width)),
                  'peers': list(range(base_width+lang_width, base_width+lang_width+5)),
                  'novelty': list(range(base_width+lang_width+5, X.shape[1]))}
        configs = [(), ('language',), ('peers',), ('novelty',), ('language', 'peers'),
                   ('language', 'peers', 'novelty')]
        models, selections = {}, []
        for groups in configs:
            columns = list(range(base_width))+list(itertools.chain.from_iterable(ranges[g] for g in groups))
            model = v3['ranker'] if not groups else fit_model(X[:, columns], y)
            pred = model.predict(S[:, columns])
            metrics = evaluate(ys, pred, selection.channel_id.to_numpy())
            models[groups] = (model, columns)
            selections.append(dict(groups=list(groups), **metrics))
            print('Inner selection:', json.dumps(selections[-1]), flush=True)
        # Freeze selection before evaluating any candidate on the outer holdout.
        best = max(selections, key=lambda m: (m['within_channel_accuracy'], -len(m['groups'])))
        baseline = selections[0]['within_channel_accuracy']
        if best['within_channel_accuracy'] < baseline+.005:
            best = selections[0]
        groups = tuple(best['groups'])
        model, columns = models[groups]
        features = (ResearchFeatures(groups=groups, base=v3['features']).fit(fitting, y)
                    if groups else v3['features'])
        # Assert that the compact selected feature path equals the scored candidate.
        np.testing.assert_allclose(features.transform(selection), S[:, columns], rtol=1e-5, atol=1e-6)
        raw = model.predict(features.transform(calibration))
        yc = leakage_safe_target(calibration, fitting, v1)
        relative, reliability = calibrate(raw, yc, calibration.channel_id.to_numpy())
        bundle = dict(version='v4', deployed='research_features_trees' if groups else 'v3_fallback',
            features=features, ranker=model, calibration=relative, pair_calibration=reliability,
            reference_rank_scores=np.sort(raw), relative_error_80pct=float(np.quantile(abs(yc-relative.predict(raw[:, None])), .8)),
            title_length_p99=v3['title_length_p99'], age_a=v3['age_a'], age_b=v3['age_b'],
            testrun_channel_id=TESTRUN_CHANNEL_ID, testrun_subscribers=v3['testrun_subscribers'],
            data_fingerprint=fingerprint, test_ids=v3['test_ids'], fitting_ids=v3['fitting_ids'],
            selection_ids=v3['selection_ids'], validation_ids=v3['validation_ids'],
            selected_groups=list(groups), candidate_selection=selections,
            resource_sha256=hashlib.sha256((ROOT/'data/lexicons/vader_valence.json').read_bytes()).hexdigest())
        joblib.dump(bundle, path, compress=3)
        print('Frozen v4 feature groups:', groups, flush=True)
    report = {'protocol':dict(outer_train=len(train), outer_test=len(test), fitting=len(fitting),
        selection=len(selection), calibration=len(calibration), original_outer_seed=42,
        selection_metric='within-channel pairwise accuracy; minimum selection lift 0.5 percentage point',
        testrun_excluded=True, cross_fitting='5 GroupKFold folds on normalized titles, fitting cohort only',
        caveats=['Reused development holdout; prospective validation still needed',
                 'Snapshot views and coarse timestamps do not establish causal title effects',
                 'No external video outcome rows added; VADER valences are the external data resource']),
        'selected_groups':bundle['selected_groups'], 'candidate_selection':bundle['candidate_selection'],
        'data_fingerprint':fingerprint}
    predictions = {}
    for version in ('v1', 'v2', 'v3', 'v4'):
        if version in ('v1', 'v2'):
            raw = legacy_predictions(test, version)
        else:
            raw, _, _ = predict_frame(v3 if version == 'v3' else bundle, test)
        predictions[version] = raw
        report[version] = evaluate(test.target.to_numpy(), raw, test.channel_id.to_numpy())
    no_thumb = test.copy(); no_thumb[THUMB_COLS] = np.nan
    for version, b in [('v3', v3), ('v4', bundle)]:
        raw, _, _ = predict_frame(b, no_thumb)
        report[version+'_without_thumbnail_input'] = evaluate(test.target.to_numpy(), raw, test.channel_id.to_numpy())
    report['paired_channel_bootstrap_vs_v3'] = paired_channel_interval(test, predictions['v3'], predictions['v4'])
    # Secondary audit: avoid crediting exact repeated-title overlap with fitting data.
    fitting_titles = set(fitting.title.map(normalized_title))
    overlap = test.title.map(normalized_title).isin(fitting_titles).to_numpy()
    report['normalized_title_overlap_count'] = int(overlap.sum())
    report['without_normalized_title_overlap'] = {}
    for version in ('v3', 'v4'):
        subset = test.loc[~overlap]
        report['without_normalized_title_overlap'][version] = evaluate(subset.target.to_numpy(),
            predictions[version][~overlap], subset.channel_id.to_numpy())
    _, relative, _ = predict_frame(bundle, test)
    yt = leakage_safe_target(test, fitting, v1)
    report['calibration'] = dict(mae=float(np.mean(abs(yt-relative))),
        zero_relative_mae=float(np.mean(abs(yt))),
        error_radius_80pct=bundle['relative_error_80pct'],
        empirical_error_radius_coverage=float(np.mean(abs(yt-relative) <= bundle['relative_error_80pct'])))
    report['runtime_seconds'] = time.perf_counter()-start
    name = 'evaluation_v4.json' if args.evaluate_only else 'metrics_v4.json'
    (ROOT/'reports'/name).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
