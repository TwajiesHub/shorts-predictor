#!/usr/bin/env python3
"""Evaluate optional rules from cached v5 predictions. No fitting or model writes.

Existing v5 reports contain aggregates only. First use materializes row-level
predictions from the frozen model; subsequent runs read/verify the saved cache.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_v3 import load_data,evaluate
from scripts.train_v4 import data_fingerprint,paired_channel_interval
from src.config import THUMB_FEATURES_PATH
from src.ranking import attach_thumbnails,predict_frame,THUMB_COLS
from src.proxy_rules import ProxyRuleContext,THRESHOLDS,WEIGHTS

CACHE = ROOT/'reports/heldout_predictions_v5.parquet'
META = ROOT/'reports/heldout_predictions_v5.json'


def cached_predictions(bundle,test,refresh=False):
    model = ROOT/'models/model_v5.joblib'
    expected = dict(model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    data_fingerprint=data_fingerprint(),rows=len(test))
    if expected['data_fingerprint'] != bundle['data_fingerprint'] or test.video_id.tolist() != bundle['test_ids']:
        raise ValueError('Saved model, dataset, and original split differ')
    generated = False
    if refresh or not (CACHE.exists() or META.exists()):
        raw,relative,percentile = predict_frame(bundle,test)
        no_thumb = test.copy();no_thumb[THUMB_COLS] = np.nan
        missing_raw,missing_relative,_ = predict_frame(bundle,no_thumb)
        saved = test[['video_id','channel_id','title','target']].copy()
        saved['original_R'] = relative
        saved['original_ranking_score'] = raw
        saved['original_niche_percentile'] = percentile
        saved['original_R_no_thumbnail'] = missing_relative
        saved['original_ranking_score_no_thumbnail'] = missing_raw
        saved.to_parquet(CACHE,index=False)
        META.write_text(json.dumps(dict(expected,predictions_sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),
            provenance='Materialized from unchanged saved v5; no training',
            score_definition='R is calibrated relative score, not raw ranking_score'),indent=2)+'\n')
        generated = True
    if not CACHE.exists() or not META.exists():
        raise ValueError('Incomplete prediction cache; use --refresh-predictions to rebuild explicitly')
    meta = json.loads(META.read_text())
    if any(meta.get(k) != v for k,v in expected.items()) or meta['predictions_sha256'] != hashlib.sha256(CACHE.read_bytes()).hexdigest():
        raise ValueError('Prediction cache stale/corrupt; use --refresh-predictions explicitly')
    saved = pd.read_parquet(CACHE)
    if saved.video_id.tolist() != test.video_id.tolist() or not np.array_equal(saved.target.to_numpy(),test.target.to_numpy()):
        raise ValueError('Prediction cache has wrong rows/order/targets')
    return saved,generated


def summarize(frame,predictions,context,column):
    adjusted = context.apply(frame,predictions,score_column=column)
    y = predictions.target.to_numpy()
    channel = predictions.channel_id.to_numpy()
    original = evaluate(y,predictions[column].to_numpy(),channel)
    after = evaluate(y,adjusted.adjusted_score.to_numpy(),channel)
    counts = Counter(name for names in adjusted.rules_fired for name in names)
    report = dict(original=original,adjusted=after,delta_percentage_points={
        k:100*(after[k]-original[k]) for k in ['all_pairs_accuracy','within_channel_accuracy','channel_macro_accuracy']},
        adjusted_rows=int((adjusted.rule_delta.abs() > 1e-12).sum()),
        clipped_rows=int(adjusted.rule_score_clipped.sum()),
        rule_firings={name:int(counts[name]) for name in WEIGHTS},
        unavailable_prior_rows=sum(c['channel_prior_percentile'] is None for c in adjusted.proxy_rule_context),
        unavailable_sweet_spot_rows=sum(c['in_sweet_spot'] is None for c in adjusted.proxy_rule_context),
        unavailable_outrage_style_rows=sum(c['high_outrage_style'] is None for c in adjusted.proxy_rule_context),
        clip_only=evaluate(y,np.clip(predictions[column].to_numpy(),THRESHOLDS['clip_low'],THRESHOLDS['clip_high']),channel),
        paired_channel_bootstrap=paired_channel_interval(frame,predictions[column].to_numpy(),adjusted.adjusted_score.to_numpy()))
    return report,adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-predictions',action='store_true')
    args = parser.parse_args()
    _,_,test,_ = load_data()
    bundle = joblib.load(ROOT/'models/model_v5.joblib')
    test = attach_thumbnails(test,cache=pd.read_parquet(THUMB_FEATURES_PATH))
    predictions,generated = cached_predictions(bundle,test,args.refresh_predictions)
    context = ProxyRuleContext(bundle)
    full,adjusted = summarize(test,predictions,context,'original_R')
    no_thumb = test.copy();no_thumb[THUMB_COLS] = np.nan
    missing,adjusted_missing = summarize(no_thumb,predictions,context,'original_R_no_thumbnail')
    # Verify baseline reproduction rather than silently evaluating a changed model.
    old = json.loads((ROOT/'reports/metrics_v5.json').read_text())
    for current,key in [(full,'v5'),(missing,'v5_without_thumbnail_input')]:
        for metric in ['all_pairs_accuracy','within_channel_accuracy']:
            if not np.isclose(current['original'][metric],old[key][metric],atol=1e-12,rtol=0):
                raise ValueError('Frozen relative-score ranking differs from published v5 baseline')
    report = dict(protocol=dict(model='v5',rows=len(test),prediction_cache=CACHE.name,
        prediction_cache_created_this_run=generated,retraining=False,defaults_changed=False,
        score='calibrated relative R; R-prime is independent, uncalibrated post-processing',
        references='Existing fitting-only same-channel prior OOF reference and historical profiles',
        unknown_context='Skip gates needing unavailable data; missing thumbnail is unknown, not poor',
        limitations=['Existing reused development holdout, not a prospective/causal test',
                    'Outrage-style prevalence is a proxy, not a measured toxicity label',
                    'Approximate history dates are not exact historical exposure/outcome times']),
        thresholds=THRESHOLDS,weights=WEIGHTS,with_thumbnail=full,no_thumbnail=missing,
        model_sha256=json.loads(META.read_text())['model_sha256'])
    (ROOT/'reports/proxy_rules_evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Auditable per-row diagnostics remain separate from immutable base predictions.
    diagnostics = predictions[['video_id','channel_id','original_R','original_R_no_thumbnail']].copy()
    diagnostics['adjusted_R'] = adjusted.adjusted_score
    diagnostics['adjusted_R_no_thumbnail'] = adjusted_missing.adjusted_score
    diagnostics['rules_fired'] = [json.dumps(x) for x in adjusted.rules_fired]
    diagnostics['rules_fired_no_thumbnail'] = [json.dumps(x) for x in adjusted_missing.rules_fired]
    diagnostics.to_parquet(ROOT/'reports/proxy_rules_predictions.parquet',index=False)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
