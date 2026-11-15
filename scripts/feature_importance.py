#!/usr/bin/env python3
"""Group permutation, leave-one-group-out and split importance for frozen v4/v5.

Only temporary in-memory analysis heads are fitted; production models, feature
transformers, calibration and default predictions are never changed or saved.
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
from sklearn.base import clone
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_v3 import load_data,leakage_safe_target
from scripts.train_v4 import by_ids,data_fingerprint
from src.config import THUMB_FEATURES_PATH,TESTRUN_CHANNEL_ID
from src.ranking import attach_thumbnails,THUMB_COLS
from src.features import build_numeric_frame
from src.feature_catalog import feature_catalog


class PairwiseTask:
    """Reuse comparable-pair arrays across all importance calculations."""
    def __init__(self,y,channels):
        y=np.asarray(y,float);channels=np.asarray(channels)
        self.pairs={}
        i,j=np.triu_indices(len(y),1)
        sign=np.sign(y[i]-y[j]);keep=sign!=0
        self.pairs['all_pairs_accuracy']=(i[keep],j[keep],sign[keep])
        ii=[];jj=[]
        for channel in np.unique(channels):
            ix=np.flatnonzero(channels==channel)
            a,b=np.triu_indices(len(ix),1);ii.extend(ix[a]);jj.extend(ix[b])
        i,j=np.asarray(ii,int),np.asarray(jj,int)
        sign=np.sign(y[i]-y[j]);keep=sign!=0
        self.pairs['within_channel_accuracy']=(i[keep],j[keep],sign[keep])
        if any(len(p[0])==0 for p in self.pairs.values()):
            raise ValueError('Evaluation requires comparable global and within-channel pairs')
        self.channels=channels
        self.channel_indices=[np.flatnonzero(channels==c) for c in np.unique(channels)]

    def score(self,predictions):
        p=np.asarray(predictions,float)
        if not np.isfinite(p).all():raise ValueError('Nonfinite predictions')
        return {name:float(((np.sign(p[i]-p[j])*truth+1)/2).mean())
                for name,(i,j,truth) in self.pairs.items()}

    def permutations(self,repeats,seed,scope):
        rng=np.random.default_rng(seed)
        result=[]
        for _ in range(repeats):
            order=np.arange(len(self.channels))
            if scope=='global':order=rng.permutation(order)
            else:
                for ix in self.channel_indices:order[ix]=rng.permutation(ix)
            result.append(order)
        return result


def views_score(model,matrices,task):
    return {view:task.score(model.predict(X)) for view,X in matrices.items()}


def drops(baseline,changed):
    return {view:{metric:100*(baseline[view][metric]-changed[view][metric])
                  for metric in baseline[view]} for view in baseline}


def permutation_summary(model,matrices,task,columns,orders,baseline):
    runs=[]
    for order in orders:
        metrics={}
        for view,X in matrices.items():
            shuffled=X.copy();shuffled[:,columns]=X[order][:,columns]
            metrics[view]=task.score(model.predict(shuffled))
        runs.append(metrics)
    summary={}
    for view in baseline:
        summary[view]={}
        for metric,value in baseline[view].items():
            accuracies=np.asarray([run[view][metric] for run in runs])
            delta=100*(value-accuracies)
            summary[view][metric]=dict(mean_accuracy=float(accuracies.mean()),
                mean_drop_pp=float(delta.mean()),std_drop_pp=float(delta.std(ddof=1)) if len(delta)>1 else 0.,
                repeat_accuracies=accuracies.tolist())
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['v4','v5'],default='v5')
    parser.add_argument('--permutation-repeats',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42,help='Permutation seed; retrained heads retain the saved estimator seed')
    parser.add_argument('--ablation',choices=['retrain','mask'],default='retrain',
                        help='retrain remaining-column heads, or fast training-median masking only')
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    if args.permutation_repeats<1:parser.error('--permutation-repeats must be positive')
    started=time.perf_counter()
    protected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'models').glob('model_v[1-5].joblib')}
    bundle=joblib.load(ROOT/f'models/model_{args.model}.joblib')
    model=bundle['ranker']
    # Current best v4/v5 use this exact regression head; refuse silent approximations.
    if type(model).__name__!='GradientBoostingRegressor':
        raise ValueError('This analysis expects the current GradientBoostingRegressor head')
    if data_fingerprint()!=bundle['data_fingerprint']:
        raise ValueError('Data changed since the model was saved')
    _,train,test,v1=load_data()
    if test.video_id.tolist()!=bundle['test_ids']:
        raise ValueError('Held-out IDs/order differ from the saved original split')
    if any(set(bundle['fitting_ids']) & set(bundle[k]) for k in ['selection_ids','validation_ids','test_ids']):
        raise ValueError('Fitting and evaluation cohorts overlap')
    cache=pd.read_parquet(THUMB_FEATURES_PATH)
    train=attach_thumbnails(train,cache=cache);test=attach_thumbnails(test,cache=cache)
    fitting=by_ids(train,bundle['fitting_ids'])
    if TESTRUN_CHANNEL_ID in set(fitting.channel_id) or TESTRUN_CHANNEL_ID in set(test.channel_id):
        raise ValueError('TestRun must remain excluded')
    y=leakage_safe_target(fitting,fitting,v1)
    features=bundle['features']
    augmented=fitting.copy()
    dropout=np.random.default_rng(42).random(len(augmented))<.3
    augmented.loc[dropout,THUMB_COLS]=np.nan
    # Critical: use saved out-of-fold priors for training, never in-sample lookup scores.
    Xtrain=features.training_matrix(augmented)
    Xtest=features.transform(test)
    no_thumb=test.copy();no_thumb[THUMB_COLS]=np.nan
    matrices={'with_thumbnail':Xtest,'no_thumbnail_input':features.transform(no_thumb)}
    catalog,groups,peer=feature_catalog(bundle)
    if len(catalog)!=Xtrain.shape[1] or Xtest.shape[1]!=model.n_features_in_ or len(catalog)!=model.n_features_in_:
        raise ValueError('Feature manifest and actual fitted matrix dimensions disagree')
    task=PairwiseTask(test.target.to_numpy(),test.channel_id.to_numpy())
    baseline=views_score(model,matrices,task)
    expected=json.loads((ROOT/f'reports/metrics_{args.model}.json').read_text())
    for view,key in [('with_thumbnail',args.model),('no_thumbnail_input',args.model+'_without_thumbnail_input')]:
        for metric,value in baseline[view].items():
            if not np.isclose(value,expected[key][metric],atol=1e-12,rtol=0):
                raise ValueError('Saved full model does not reproduce published baseline')
    print('Full saved model:',json.dumps(baseline),flush=True)
    control={'performed':False}
    if args.ablation=='retrain':
        replica=clone(model).fit(Xtrain,y)
        difference=float(np.max(np.abs(replica.predict(Xtest)-model.predict(Xtest))))
        if difference>1e-10:
            raise ValueError(f'Full-head retraining does not reproduce saved model: max error {difference}')
        control=dict(performed=True,max_absolute_prediction_difference=difference,
                     metrics=views_score(replica,matrices,task))
        print('Full-head reproduction verified:',difference,flush=True)
    orders={scope:task.permutations(args.permutation_repeats,args.seed,scope) for scope in ['global','within_channel']}
    medians=np.median(Xtrain,axis=0)
    importance=np.asarray(model.feature_importances_,float)
    results={}
    for group,info in groups.items():
        cols=info['columns']
        entry=dict(info,n_features=len(cols),native_impurity_share=float(importance[cols].sum()))
        if not cols:
            entry.update(status='not_used_by_current_model',permutation=None,
                         ablation=dict(method='structural_noop',metrics=baseline,drop_pp=drops(baseline,baseline)))
        else:
            entry['status']='active'
            entry['permutation']={scope:permutation_summary(model,matrices,task,cols,indices,baseline)
                                  for scope,indices in orders.items()}
            masked={}
            for view,X in matrices.items():
                values=X.copy();values[:,cols]=medians[cols]
                masked[view]=task.score(model.predict(values))
            entry['median_mask']=dict(metrics=masked,drop_pp=drops(baseline,masked))
            if args.ablation=='retrain':
                keep=np.ones(Xtrain.shape[1],bool);keep[cols]=False
                reduced=clone(model).fit(Xtrain[:,keep],y)
                metrics=views_score(reduced,{view:X[:,keep] for view,X in matrices.items()},task)
                entry['ablation']=dict(method='retrained_head_remaining_columns',metrics=metrics,drop_pp=drops(baseline,metrics))
            else:
                entry['ablation']=dict(method='training_median_mask',metrics=masked,drop_pp=drops(baseline,masked))
        results[group]=entry
        print(group,json.dumps(entry['ablation']),flush=True)
    # Native importance is split-impurity reduction, not a causal gain or ranking-loss coefficient.
    for row in catalog:
        row['native_impurity_share']=float(importance[row['index']])
        row['constant_in_training']=bool(np.ptp(Xtrain[:,row['index']])==0)
    ranking=lambda scope,metric:sorted([g for g,v in results.items() if v['active']],
        key=lambda g:results[g]['permutation'][scope]['with_thumbnail'][metric]['mean_drop_pp'],reverse=True)
    ridge=peer.ridge
    vocabulary=peer.vectorizer.get_feature_names_out()
    coefficients=np.asarray(ridge.coef_).ravel()
    ordered=np.argsort(coefficients)
    prior_coefficients={direction:[dict(term=str(vocabulary[i]),coefficient=float(coefficients[i])) for i in indices]
        for direction,indices in [('negative',ordered[:20]),('positive',ordered[-20:][::-1])]}
    numeric_fit=build_numeric_frame(fitting);numeric_test=build_numeric_frame(test)
    report=dict(model=args.model,baseline=baseline,groups=results,feature_catalog=catalog,
        ranked_groups_overall=ranking('global','all_pairs_accuracy'),
        ranked_groups_within_channel=ranking('within_channel','within_channel_accuracy'),
        native_top_features=sorted(catalog,key=lambda row:row['native_impurity_share'],reverse=True)[:25],
        unused_split_features=[r['name'] for r in catalog if r['native_impurity_share']==0],
        constant_training_features=[r['name'] for r in catalog if r['constant_in_training']],
        prior_ridge_coefficients=dict(description='Auxiliary TF-IDF prior coefficients, not final-head word effects; correlated, observational features',
                                      top_terms=prior_coefficients),
        retrain_control=control,
        protocol=dict(outer_seed=42,outer_train=len(train),held_out=len(test),fitting=len(fitting),
            selection=len(bundle['selection_ids']),calibration=len(bundle['validation_ids']),
            fitting_ids=bundle['fitting_ids'],held_out_ids=bundle['test_ids'],
            data_fingerprint=bundle['data_fingerprint'],model_sha256=protected[f'model_{args.model}.joblib'],
            all_comparable_pairs=len(task.pairs['all_pairs_accuracy'][0]),within_channel_comparable_pairs=len(task.pairs['within_channel_accuracy'][0]),
            testrun_excluded=True,proxy_rules=False,permutation_repeats=args.permutation_repeats,
            permutation_seed=args.seed,ablation_method=args.ablation,training_thumbnail_dropout_fraction=float(dropout.mean()),
            training_priors='saved GroupKFold out-of-fold values; fitting cohort only',
            estimators=model.get_params(deep=False),native_importance='normalized split impurity reduction',
            nonapproximate_timestamp_flag_fitting=int((fitting.hour_is_approximate==False).sum()),
            nonapproximate_timestamp_flag_heldout=int((test.hour_is_approximate==False).sum()),
            hour_feature_known_fitting=int(numeric_fit.hour_missing.eq(0).sum()),
            hour_feature_known_heldout=int(numeric_test.hour_missing.eq(0).sum()),
            weekday_feature_known_fitting=int(numeric_fit.date_is_coarse.eq(0).sum()),
            weekday_feature_known_heldout=int(numeric_test.date_is_coarse.eq(0).sum()),
            assumptions=['Fixed transformed-feature block removal, not causal removal of raw inputs',
                'Correlated information remains in other blocks (title information also reaches performance priors)',
                'Cross-block interactions assigned once and listed in feature dependencies',
                'Transforms and priors stay frozen; only temporary remaining-column heads are fitted',
                'One original head seed; retrain differences can include tie-breaking/random-order effects',
                'Permutation SD measures shuffle variation, not a confidence interval over future data',
                'Permutation breaks cross-block correlations and can create off-distribution combinations',
                'Existing reused random per-channel development holdout, not future-time or causal validation']),
        runtime_seconds=time.perf_counter()-started)
    for name,digest in protected.items():
        if hashlib.sha256((ROOT/'models'/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Production model unexpectedly changed')
    suffix='_mask' if args.ablation=='mask' else ''
    output=args.output or ROOT/f'reports/feature_importance_{args.model}{suffix}.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Saved',str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),flush=True)
    print('Ranked overall groups:',', '.join(report['ranked_groups_overall']),flush=True)
    print('Runtime seconds:',round(report['runtime_seconds'],2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
