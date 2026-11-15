#!/usr/bin/env python3
"""Analyze cheap palettes and a temporary augmented head; never changes v1–v5.

Auto mode uses real cached images when coverage is >=90%, otherwise explicitly
labeled derived cached-color proxies. --download retrieves only i.ytimg.com images.
"""
import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from threading import Lock,Event
from urllib.error import HTTPError,URLError
from urllib.parse import urlparse
from urllib.request import Request,HTTPRedirectHandler,build_opener

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr,wilcoxon
from sklearn.base import clone
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_v3 import load_data,leakage_safe_target
from scripts.train_v4 import by_ids,data_fingerprint,paired_channel_interval
from scripts.feature_importance import PairwiseTask
from scripts.download_thumbnails import CANDIDATE_NAMES
from src.config import THUMB_FEATURES_PATH,THUMB_DIR,TESTRUN_CHANNEL_ID
from src.ranking import attach_thumbnails,THUMB_COLS
from src.palette_features import CONFIG,PIXEL_NAMES,extract_palette,palette_from_rgb,cached_color_proxies


class OnlyThumbnailRedirects(HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        p=urlparse(newurl)
        if p.scheme!='https' or p.hostname!='i.ytimg.com':
            raise URLError('Refusing thumbnail redirect outside https://i.ytimg.com')
        return super().redirect_request(request,fp,code,msg,headers,newurl)


def thumbnail_probe(video_id):
    url=f'https://i.ytimg.com/vi/{video_id}/hqdefault.jpg'
    try:
        with build_opener(OnlyThumbnailRedirects()).open(Request(url,headers={'User-Agent':'shorts-predictor-palettes/1.0'}),timeout=20) as r:
            r.read(128)
            return dict(accessible=True,http_status=r.status)
    except HTTPError as exc:
        return dict(accessible=exc.code==404,http_status=exc.code,error='Thumbnail service HTTP response')
    except (URLError,TimeoutError) as exc:
        return dict(accessible=False,error=str(exc))


def download_thumbnails(ids,directory):
    probe=thumbnail_probe(ids[0])
    if not probe['accessible']:return dict(status='blocked',probe=probe,downloaded=0)
    directory.mkdir(parents=True,exist_ok=True)
    lock=Lock();clock=[0.];stopped=Event()
    def polite_request(url):
        with lock:
            if stopped.is_set():raise RuntimeError('Thumbnail collection stopped')
            time.sleep(max(0,.4-(time.monotonic()-clock[0])))
            clock[0]=time.monotonic()
        with build_opener(OnlyThumbnailRedirects()).open(Request(url,headers={
                'User-Agent':'shorts-predictor-palettes/1.0','Accept':'image/*'}),timeout=25) as response:
            data=response.read(4*1024*1024+1)
        if len(data)>4*1024*1024:raise ValueError('Thumbnail exceeds 4 MiB limit')
        return data
    def one(vid):
        if stopped.is_set():return dict(video_id=vid,status='cancelled')
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}',str(vid)):raise ValueError('Invalid video ID')
        path=directory/f'{vid}.jpg'
        if path.exists():return dict(video_id=vid,status='existing')
        for name in CANDIDATE_NAMES:
            url=f'https://i.ytimg.com/vi/{vid}/{name}'
            try:
                data=polite_request(url)
                import io
                with Image.open(io.BytesIO(data)) as image:image.verify()
                path.write_bytes(data)
                return dict(video_id=vid,status='downloaded',url=url,sha256=hashlib.sha256(data).hexdigest())
            except HTTPError as exc:
                if exc.code==429:
                    stopped.set()
                    raise RuntimeError('i.ytimg.com rate-limited requests; respect Retry-After before rerunning') from None
                if exc.code not in (403,404):raise
            except URLError:
                stopped.set()
                raise RuntimeError('Thumbnail transport unavailable; preserve the cache and retry after access is restored') from None
            except (TimeoutError,ValueError,OSError):
                continue
        return dict(video_id=vid,status='unavailable')
    records=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(one,vid) for vid in ids]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%100==0:print(f'Thumbnail collection {len(records)}/{len(ids)}',flush=True)
    (ROOT/'reports/palette_download_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    return dict(status='complete',probe=probe,**dict(Counter(r['status'] for r in records)))


def qvalues(pvalues):
    """Benjamini–Hochberg adjusted p-values; None means insufficient evidence."""
    valid=[i for i,p in enumerate(pvalues) if p is not None]
    ordered=sorted(valid,key=lambda i:pvalues[i]);out=[None]*len(pvalues);last=1.
    for rank in range(len(ordered),0,-1):
        i=ordered[rank-1];last=min(last,pvalues[i]*len(ordered)/rank);out[i]=float(min(1,last))
    return out


def distribution(frame,values):
    frame=frame.reset_index(drop=True);values=values.reset_index(drop=True)
    groups=[]
    for channel,group in frame.groupby('channel_id'):
        if len(group)<8:continue
        lo,hi=group.target.quantile([.25,.75])
        if hi<=lo:continue
        low=group.index[group.target<=lo].to_numpy();high=group.index[group.target>=hi].to_numpy()
        if len(low)>=2 and len(high)>=2:groups.append((channel,low,high))
    records=[]
    for column in values:
        x=values[column].to_numpy(float);deltas=[];highs=[];lows=[]
        for _,low,high in groups:
            a=x[high];b=x[low];a=a[np.isfinite(a)];b=b[np.isfinite(b)]
            if len(a)>=2 and len(b)>=2:
                deltas.append(float(a.mean()-b.mean()));highs.extend(a);lows.extend(b)
        finite=np.isfinite(x)&np.isfinite(frame.target.to_numpy())
        corr=float(spearmanr(x[finite],frame.target.to_numpy()[finite]).statistic) if finite.sum()>2 and np.ptp(x[finite])>0 else None
        ci=None;p=None
        if len(deltas)>=5:
            d=np.array(deltas);rng=np.random.default_rng(42)
            boot=rng.choice(d,size=(2000,len(d)),replace=True).mean(axis=1)
            ci=np.quantile(boot,[.025,.975]).tolist()
            p=float(wilcoxon(d).pvalue) if np.any(np.abs(d)>1e-12) else 1.
        records.append(dict(feature=column,channels=len(deltas),high_n=len(highs),low_n=len(lows),
            mean_high=float(np.mean(highs)) if highs else None,mean_low=float(np.mean(lows)) if lows else None,
            median_high=float(np.median(highs)) if highs else None,median_low=float(np.median(lows)) if lows else None,
            mean_channel_difference=float(np.mean(deltas)) if deltas else None,
            mean_channel_difference_95pct=ci,channel_signed_rank_p=p,spearman=corr))
    for row,q in zip(records,qvalues([r['channel_signed_rank_p'] for r in records])):row['fdr_q']=q
    return records


def palette_arrays(frame,table,names,medians=None,dropout=None):
    raw=table.set_index('video_id').reindex(frame.video_id)[names].to_numpy(float,copy=True)
    available=np.isfinite(raw).all(axis=1)
    if medians is None:
        medians=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0. for col in raw.T])
    missing=np.where(~np.isfinite(raw));raw[missing]=medians[missing[1]]
    if dropout is not None:raw[dropout]=medians;available[dropout]=False
    return np.column_stack([raw,available]).astype(np.float32),medians


def extract_images(ids,directory):
    records=[];timings=[]
    for vid in ids:
        path=directory/f'{vid}.jpg'
        if not path.exists():continue
        try:
            start=time.perf_counter();features,swatches=extract_palette(path);timings.append(time.perf_counter()-start)
            records.append(dict(video_id=vid,**features,dominant_swatches=json.dumps(swatches),
                                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        except (OSError,ValueError):continue
    table=pd.DataFrame(records,columns=['video_id']+PIXEL_NAMES+['dominant_swatches','image_sha256'])
    if len(table):table.to_parquet(ROOT/'data/palette_features_v1.parquet',index=False)
    return table,timings


def extraction_benchmark(directory,ids):
    paths=[directory/f'{v}.jpg' for v in ids if (directory/f'{v}.jpg').exists()][:40]
    if paths:source='real_local_thumbnails'
    else:source='synthetic_benchmark_only'
    with tempfile.TemporaryDirectory() as tmp:
        if not paths:
            pixels=np.random.default_rng(42).integers(0,256,(1920,1080,3),dtype=np.uint8)
            for i in range(10):
                path=Path(tmp)/f'{i}.jpg';Image.fromarray(pixels).save(path);paths.append(path)
        times=[]
        for _ in range(3):
            for path in paths:
                start=time.perf_counter();extract_palette(path);times.append(1000*(time.perf_counter()-start))
    raw=np.random.default_rng(9).integers(0,256,(128,72,3),dtype=np.uint8)
    numeric=[]
    for _ in range(50):
        start=time.perf_counter();palette_from_rgb(raw);numeric.append(1000*(time.perf_counter()-start))
    return dict(source=source,measurements=len(times),median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),
                includes='image decode, resize, crop, quantization and all 50 pixel features',
                predecoded_128x72_rgb_median_ms=float(np.median(numeric)),predecoded_source='synthetic_benchmark_only')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=['auto','pixels','cached'],default='auto')
    parser.add_argument('--download',action='store_true',help='Polite i.ytimg.com collection; full split can take >30 minutes')
    parser.add_argument('--thumbnail-dir',type=Path,default=THUMB_DIR)
    parser.add_argument('--permutation-repeats',type=int,default=10)
    args=parser.parse_args()
    if args.permutation_repeats<1:parser.error('--permutation-repeats must be positive')
    started=time.perf_counter()
    protected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'models').glob('model_v[1-5].joblib')}
    _,train,test,v1=load_data();bundle=joblib.load(ROOT/'models/model_v5.joblib')
    if bundle['data_fingerprint']!=data_fingerprint() or test.video_id.tolist()!=bundle['test_ids']:
        raise ValueError('Saved v5 data/split changed')
    cache=pd.read_parquet(THUMB_FEATURES_PATH)
    train=attach_thumbnails(train,cache=cache);test=attach_thumbnails(test,cache=cache)
    fitting=by_ids(train,bundle['fitting_ids']);selection=by_ids(train,bundle['selection_ids'])
    needed=pd.concat([fitting,selection,test],ignore_index=True)
    if TESTRUN_CHANNEL_ID in set(needed.channel_id):raise ValueError('TestRun must remain excluded')
    ids=needed.video_id.astype(str).tolist()
    collection=download_thumbnails(ids,args.thumbnail_dir) if args.download else dict(status='not_requested')
    pixels,timings=extract_images(ids,args.thumbnail_dir)
    if pixels.empty and not args.download:
        collection=dict(status='no_local_images',probe=thumbnail_probe(ids[0]),
                        retry_command='python scripts/analyze_palettes.py --download --source pixels')
    coverage={name:float(frame.video_id.isin(pixels.video_id).mean()) for name,frame in [('fitting',fitting),('selection',selection),('heldout',test)]}
    use_pixels=args.source=='pixels' or (args.source=='auto' and min(coverage.values())>=.9)
    if use_pixels and pixels.empty:
        use_pixels=False;collection['pixel_requested_but_unavailable']=True
    source='pixels' if use_pixels else 'cached_color_proxies'
    if not use_pixels:
        proxies=cached_color_proxies(needed)
        table=pd.concat([needed[['video_id']].reset_index(drop=True),proxies.reset_index(drop=True)],axis=1)
        names=proxies.columns.tolist()
    else:table=pixels;names=PIXEL_NAMES
    print('Palette source:',source,'image coverage:',json.dumps(coverage),flush=True)
    # Existing scalar colors are analyzable even when source images are absent.
    base_stats=['thumb_saturation_mean','thumb_saturation_std','thumb_brightness','thumb_contrast',
                'thumb_colorfulness','thumb_warm_frac','thumb_dom_s','thumb_dom_v','thumb_dom_frac',
                'thumb_bright_frac','thumb_dark_frac']
    summaries={}
    for label,frame in [('fitting',fitting),('heldout',test)]:
        values=frame[base_stats].reset_index(drop=True).copy()
        palette_values=table.set_index('video_id').reindex(frame.video_id)[names].reset_index(drop=True)
        values=pd.concat([values,palette_values],axis=1)
        summaries[label]=distribution(frame.reset_index(drop=True),values)
    fit_stats={r['feature']:r for r in summaries['fitting']}
    consistent=[]
    for row in summaries['heldout']:
        earlier=fit_stats[row['feature']];delta=row['mean_channel_difference'];other=earlier['mean_channel_difference']
        ci=row['mean_channel_difference_95pct'];q=row['fdr_q']
        if delta is not None and other is not None and ci and q is not None and q<.05 and ci[0]*ci[1]>0 and delta*other>0:
            consistent.append(row['feature'])
    y=leakage_safe_target(fitting,fitting,v1)
    dropout=np.random.default_rng(42).random(len(fitting))<.3
    augmented=fitting.copy();augmented.loc[dropout,THUMB_COLS]=np.nan
    X=bundle['features'].training_matrix(augmented);T=bundle['features'].transform(test)
    S=bundle['features'].transform(selection)
    P,medians=palette_arrays(fitting,table,names,dropout=dropout)
    PT,_=palette_arrays(test,table,names,medians);PS,_=palette_arrays(selection,table,names,medians)
    control=clone(bundle['ranker']).fit(X,y)
    error=float(np.max(np.abs(control.predict(T)-bundle['ranker'].predict(T))))
    if error>1e-10:raise ValueError('Baseline full-head reproduction failed')
    candidate=clone(bundle['ranker']).fit(np.hstack([X,P]),y)
    task=PairwiseTask(test.target.to_numpy(),test.channel_id.to_numpy())
    before=task.score(control.predict(T));pred=candidate.predict(np.hstack([T,PT]));after=task.score(pred)
    selection_task=PairwiseTask(leakage_safe_target(selection,fitting,v1),selection.channel_id.to_numpy())
    selection_before=selection_task.score(control.predict(S));selection_after=selection_task.score(candidate.predict(np.hstack([S,PS])))
    delta={m:100*(after[m]-before[m]) for m in before}
    selection_delta=100*(selection_after['all_pairs_accuracy']-selection_before['all_pairs_accuracy'])
    # Fixed group permutation; no feature subset selected using held-out outcomes.
    rng=np.random.default_rng(42);permutation=[]
    for _ in range(args.permutation_repeats):
        order=rng.permutation(len(test));shuffled=task.score(candidate.predict(np.hstack([T,PT[order]])))
        permutation.append({m:100*(after[m]-shuffled[m]) for m in after})
    no_thumb=test.copy();no_thumb[THUMB_COLS]=np.nan
    NT=bundle['features'].transform(no_thumb);NP=np.column_stack([np.broadcast_to(medians,(len(test),len(names))),np.zeros(len(test))]).astype(np.float32)
    missing_before=task.score(control.predict(NT));missing_after=task.score(candidate.predict(np.hstack([NT,NP])))
    eligible=bool(use_pixels and min(coverage.values())>=.9 and delta['all_pairs_accuracy']>=.3 and selection_delta>=.3)
    permutation_summary={m:dict(mean_drop_pp=float(np.mean([p[m] for p in permutation])),
        std_drop_pp=float(np.std([p[m] for p in permutation],ddof=1)) if len(permutation)>1 else 0.) for m in after}
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),source=source,real_images_extracted_count=len(pixels),pixel_palette_status='measured' if use_pixels else 'not_measured_no_sufficient_images',
        collection=collection,image_coverage=coverage,extraction_config=CONFIG,
        exact_augmented_features=names+['palette_observed'],pixel_features_supported=PIXEL_NAMES,
        unmeasured_pixel_features=[] if use_pixels else PIXEL_NAMES,
        summary_outputs='Full JSON and CSV statistics; no plotting dependency added',
        distributions=summaries,consistent_direction_with_fdr_and_bootstrap=consistent,
        distribution_method='within-channel performance quartiles, >=8 rows/channel; channel-equal mean high-minus-low differences, 2000 channel bootstraps; signed-rank p and BH FDR per split',
        model_evaluation=dict(baseline=before,with_palette=after,
            delta_pp=delta,drop_when_palette_removed_pp=delta,
            no_thumbnail_original=missing_before,no_thumbnail_variant=missing_after,
            selection_original=selection_before,selection_variant=selection_after,selection_lift_pp=selection_delta,
            baseline_control_max_abs_error=error,palette_permutation_drop_pp=permutation,
            palette_permutation_summary=permutation_summary,
            within_channel_lift_bootstrap=paired_channel_interval(test,control.predict(T),pred)),
        promotion=dict(threshold_pp=.3,eligible_by_selection_and_development_metrics=eligible,
            deployed=False,reason='Temporary analysis only; no v1–v5 changes. Missing/insufficient pixels cannot validate a new palette model.' if not eligible else 'Selection and development thresholds passed; independent validation still needed before recommending deployment.'),
        extraction_benchmark=extraction_benchmark(args.thumbnail_dir,ids),
        protocol=dict(fitting=len(fitting),selection=len(selection),calibration_reserved=len(bundle['validation_ids']),heldout=len(test),
            original_split_seed=42,heldout_ids=bundle['test_ids'],fitting_ids=bundle['fitting_ids'],testrun_excluded=True,
            data_fingerprint=bundle['data_fingerprint'],model_sha256=protected['model_v5.joblib'],
            feature_imputation='fitting medians only; identical seed-42 thumbnail dropout for old and new features',
            model='same fixed 180-tree head and params; temporary models only',
            caveats=['Cached proxies do not contain top-five swatches, high-saturation pixel fractions or harmony',
                     'Thumbnails fetched now may differ from original cached-image versions',
                     'Distribution differences are observational, not causal CTR or views effects',
                     'Existing development holdout reused; a positive small lift still needs fresh validation']),
        runtime_seconds=time.perf_counter()-started)
    for name,digest in protected.items():
        if hashlib.sha256((ROOT/'models'/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('A production model changed')
    suffix='pixels' if use_pixels else 'cached'
    (ROOT/f'reports/palette_analysis_{suffix}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    pd.DataFrame([dict(split=split,**row) for split,rows in summaries.items() for row in rows]).to_csv(ROOT/f'reports/palette_distributions_{suffix}.csv',index=False)
    print(json.dumps({key:report[key] for key in ['source','pixel_palette_status','consistent_direction_with_fdr_and_bootstrap',
         'model_evaluation','promotion','extraction_benchmark','runtime_seconds']},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
