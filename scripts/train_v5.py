#!/usr/bin/env python3
"""V5 ablations: public title priors and emotional cues, selected before holdout."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_v3 import load_data, evaluate, leakage_safe_target
from scripts.train_v4 import by_ids, fit_model, calibrate, data_fingerprint, paired_channel_interval
from src.config import THUMB_FEATURES_PATH, TESTRUN_CHANNEL_ID
from src.ranking import attach_thumbnails, predict_frame, THUMB_COLS
from src.public_priors import V5Features, LinearRankCorrection
from src.research_features import normalized_title
from src.emotional_triggers import RESOURCE, emotion_matrix

PUBLIC = ROOT/'data/public_title_examples_v5.parquet'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--head',choices=['pairwise','retrain'],default='pairwise',
                        help='pairwise keeps v4 frozen and learns a tiny linear RankNet correction')
    args = parser.parse_args()
    start = time.perf_counter()
    df, train, test, v1 = load_data()
    v4 = joblib.load(ROOT/'models/model_v4.joblib')
    fingerprint = data_fingerprint()
    if v4['data_fingerprint'] != fingerprint or v4['test_ids'] != test.video_id.tolist():
        raise ValueError('Original data/split differs from v4')
    external = pd.read_parquet(PUBLIC)
    collection = json.loads((ROOT/'reports/public_collection_v5.json').read_text())
    public_sha = hashlib.sha256(PUBLIC.read_bytes()).hexdigest()
    rules_sha = hashlib.sha256(RESOURCE.read_bytes()).hexdigest()
    if public_sha != collection['output_sha256']:
        raise ValueError('Public data fingerprint differs from collection manifest')
    # Exclusion precedes all source cohort normalization in the collector; verify
    # the resulting bank again before any supervised feature is fitted.
    if (set(external.video_id) & set(df.video_id) or
        set(external.title.map(normalized_title)) & set(df.title.map(normalized_title)) or
        TESTRUN_CHANNEL_ID in set(external.channel_id)):
        raise ValueError('External corpus overlaps original benchmark IDs/titles/TestRun')
    cache = pd.read_parquet(THUMB_FEATURES_PATH)
    train = attach_thumbnails(train,cache=cache)
    test = attach_thumbnails(test,cache=cache)
    fitting = by_ids(train,v4['fitting_ids'])
    selection = by_ids(train,v4['selection_ids'])
    calibration = by_ids(train,v4['validation_ids'])
    y = leakage_safe_target(fitting,fitting,v1)
    path = ROOT/'models/model_v5.joblib'
    if args.evaluate_only:
        bundle = joblib.load(path)
        if (bundle['data_fingerprint'] != fingerprint or bundle['public_data_sha256'] != public_sha or
            bundle['emotional_rules_sha256'] != rules_sha or bundle['test_ids'] != test.video_id.tolist()):
            raise ValueError('Saved v5 does not match data/rules/split')
    else:
        full = V5Features(v4['features']).fit(fitting,y,external)
        augmented = fitting.copy()
        augmented.loc[np.random.default_rng(42).random(len(augmented)) < .3,THUMB_COLS] = np.nan
        X = full.training_matrix(augmented)
        S = full.transform(selection)
        width = v4['features'].transform(fitting.iloc[:1]).shape[1]
        ranges = {'public_priors':list(range(width,width+7)),
                  'emotions':list(range(width+7,X.shape[1]))}
        configs = [(),('emotions',),('public_priors',),('public_priors','emotions')]
        ys = leakage_safe_target(selection,fitting,v1)
        candidates = [];models = {}
        for groups in configs:
            columns = list(range(width))+[c for group in groups for c in ranges[group]]
            regularizers = (1.,.1,.01) if groups and args.head == 'pairwise' else (None,)
            for l2 in regularizers:
                if not groups:
                    model = v4['ranker']
                elif args.head == 'retrain':
                    model = fit_model(X[:,columns],y)
                else:
                    model = LinearRankCorrection(v4['ranker'],width,l2).fit(X[:,columns],y,fitting.channel_id.to_numpy())
                metrics = evaluate(ys,model.predict(S[:,columns]),selection.channel_id.to_numpy())
                candidate = dict(groups=list(groups),l2=l2,head=args.head,**metrics)
                candidates.append(candidate);models[(groups,l2)] = model,columns
                print('Inner selection:',json.dumps(candidate),flush=True)
        best = max(candidates,key=lambda c:(c['within_channel_accuracy'],-len(c['groups']),c['l2'] or 0))
        if best['within_channel_accuracy'] < candidates[0]['within_channel_accuracy']+.005:
            best = candidates[0]
        groups = tuple(best['groups'])
        model,columns = models[(groups,best['l2'])]
        features = V5Features(v4['features'],groups).fit(fitting,y,external)
        np.testing.assert_allclose(features.transform(selection),S[:,columns],atol=1e-6,rtol=1e-5)
        raw = model.predict(features.transform(calibration))
        yc = leakage_safe_target(calibration,fitting,v1)
        relative,reliability = calibrate(raw,yc,calibration.channel_id.to_numpy())
        bundle = dict(version='v5',deployed=('pairwise_linear_correction' if args.head == 'pairwise' else 'public_emotional_trees') if groups else 'v4_fallback',
            features=features,ranker=model,calibration=relative,pair_calibration=reliability,
            reference_rank_scores=np.sort(raw),relative_error_80pct=float(np.quantile(abs(yc-relative.predict(raw[:,None])),.8)),
            title_length_p99=v4['title_length_p99'],age_a=v4['age_a'],age_b=v4['age_b'],
            testrun_channel_id=TESTRUN_CHANNEL_ID,testrun_subscribers=v4['testrun_subscribers'],
            data_fingerprint=fingerprint,test_ids=v4['test_ids'],fitting_ids=v4['fitting_ids'],
            selection_ids=v4['selection_ids'],validation_ids=v4['validation_ids'],
            selected_groups=list(groups),head=args.head,selected_l2=best['l2'],candidate_selection=candidates,public_data_sha256=public_sha,
            emotional_rules_sha256=rules_sha,collection_summary=collection)
        joblib.dump(bundle,path,compress=3)
        print('Frozen v5 groups:',groups,flush=True)
    report = dict(protocol=dict(outer_test=len(test),fitting=len(fitting),selection=len(selection),calibration=len(calibration),
        testrun_excluded=True,selection_metric='within-channel pairwise accuracy; minimum inner lift 0.5 percentage point',
        cross_fitting='5 GroupKFold folds across original fitting + external normalized titles',
        external_source_weight=.35,external_rows=len(external),original_data_unchanged=True,
        caveats=['Reused development holdout; fresh prospective evaluation still needed',
                 'External snapshot is older and has unknown count-observation timestamps',
                 'Emotional scores are hand-authored cue heuristics, not measured human reactions']),
        data_fingerprint=fingerprint,public_data_sha256=public_sha,emotional_rules_sha256=rules_sha,
        selected_groups=bundle['selected_groups'],head=bundle.get('head','retrain'),selected_l2=bundle.get('selected_l2'),
        candidate_selection=bundle['candidate_selection'],
        collection=collection)
    predictions = {}
    for version,b in [('v4',v4),('v5',bundle)]:
        raw,relative,_ = predict_frame(b,test)
        predictions[version] = raw
        report[version] = evaluate(test.target.to_numpy(),raw,test.channel_id.to_numpy())
        no_thumb = test.copy();no_thumb[THUMB_COLS] = np.nan
        missing,_,_ = predict_frame(b,no_thumb)
        report[version+'_without_thumbnail_input'] = evaluate(test.target.to_numpy(),missing,test.channel_id.to_numpy())
    report['paired_channel_bootstrap_vs_v4'] = paired_channel_interval(test,predictions['v4'],predictions['v5'])
    _,relative,_ = predict_frame(bundle,test)
    yt = leakage_safe_target(test,fitting,v1)
    report['calibration'] = dict(mae=float(np.mean(abs(yt-relative))),zero_relative_mae=float(np.mean(abs(yt))),
        error_radius_80pct=bundle['relative_error_80pct'],empirical_error_radius_coverage=float(np.mean(abs(yt-relative) <= bundle['relative_error_80pct'])))
    report['runtime_seconds'] = time.perf_counter()-start
    report_path = ROOT/'reports'/('evaluation_v5.json' if args.evaluate_only else 'metrics_v5.json')
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
