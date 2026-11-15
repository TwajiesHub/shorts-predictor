import unittest

import numpy as np
import pandas as pd

from scripts.feature_importance import PairwiseTask,permutation_summary
from scripts.train_v3 import exact_accuracy
from scripts.predict import load_bundle
from src.feature_catalog import feature_catalog
from src.ranking import attach_thumbnails


class PairwiseImportanceTests(unittest.TestCase):
    def test_task_matches_existing_tie_aware_evaluator(self):
        y=np.array([1.,1.,2.,3.,4.]);p=np.array([0.,0.,3.,3.,1.])
        channels=np.array(['a','a','b','b','b'])
        task=PairwiseTask(y,channels);scores=task.score(p)
        self.assertEqual(scores['all_pairs_accuracy'],exact_accuracy(y,p)[0])
        self.assertEqual(scores['within_channel_accuracy'],exact_accuracy(y[2:],p[2:])[0])

    def test_within_channel_shuffle_preserves_membership_and_is_reproducible(self):
        channels=np.repeat(['a','b'],4)
        task=PairwiseTask(np.arange(8),channels)
        first=task.permutations(3,42,'within_channel')
        second=task.permutations(3,42,'within_channel')
        for a,b in zip(first,second):
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(channels,channels[a])

    def test_joint_block_permutation_detects_predictive_column(self):
        class FirstColumn:
            def predict(self,X):return X[:,0]
        X=np.column_stack([np.arange(10),np.zeros(10)])
        task=PairwiseTask(np.arange(10),np.repeat('a',10))
        baseline={'view':task.score(X[:,0])}
        orders=task.permutations(5,1,'global')
        useful=permutation_summary(FirstColumn(),{'view':X},task,[0],orders,baseline)
        unused=permutation_summary(FirstColumn(),{'view':X},task,[1],orders,baseline)
        self.assertGreater(useful['view']['all_pairs_accuracy']['mean_drop_pp'],0)
        self.assertEqual(unused['view']['all_pairs_accuracy']['mean_drop_pp'],0)


class FeatureCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=load_bundle('v5')
        cls.catalog,cls.groups,cls.research=feature_catalog(cls.bundle)
        cls.frame=attach_thumbnails(pd.DataFrame([dict(video_id='unknown-catalog-row',title='Why your phone is bad? 3 tricks',
            duration=30.,channel_id='unknown-catalog-channel',channel_subscribers=1000,
            published_ts=1794823200.,hour_is_approximate=False)]),paths=[None])

    def test_catalog_is_a_disjoint_complete_partition_and_inactive_means_absent(self):
        columns=[c for group in self.groups.values() for c in group['columns']]
        self.assertEqual(sorted(columns),list(range(self.bundle['ranker'].n_features_in_)))
        self.assertEqual(len(columns),len(set(columns)))
        for name in ['emotional_triggers_v5','external_public_priors_v5','thumbnail_embeddings','novelty']:
            self.assertFalse(self.groups[name]['active'])
            self.assertEqual(self.groups[name]['columns'],[])

    def changed_groups(self,changed):
        a=self.bundle['features'].transform(self.frame)
        b=self.bundle['features'].transform(changed)
        cols=set(np.flatnonzero(np.abs(a[0]-b[0])>1e-6))
        return {r['group'] for r in self.catalog if r['index'] in cols}

    def test_duration_and_subscriber_interactions_are_in_the_correct_groups(self):
        frame=self.frame.copy();frame['duration']=60.
        self.assertEqual(self.changed_groups(frame),{'duration'})
        frame=self.frame.copy();frame['channel_subscribers']=1e6
        self.assertEqual(self.changed_groups(frame),{'subscriber_size'})

    def test_thumbnail_and_time_columns_match_actual_transform(self):
        frame=self.frame.copy()
        for name in frame.columns:
            if name.startswith('thumb_'):frame[name]=.2
        self.assertEqual(self.changed_groups(frame),{'thumbnail_cheap_stats'})
        frame=self.frame.copy();frame['published_ts']+=3600*26
        self.assertEqual(self.changed_groups(frame),{'posting_time'})

    def test_current_v4_and_v5_share_the_same_features_and_head(self):
        old=load_bundle('v4')
        catalog,groups,_=feature_catalog(old)
        self.assertEqual(catalog,self.catalog)
        np.testing.assert_array_equal(old['features'].transform(self.frame),self.bundle['features'].transform(self.frame))
        np.testing.assert_array_equal(old['ranker'].feature_importances_,self.bundle['ranker'].feature_importances_)
        np.testing.assert_array_equal(old['ranker'].predict(old['features'].transform(self.frame)),
                                      self.bundle['ranker'].predict(self.bundle['features'].transform(self.frame)))

    def test_title_svd_manifest_matches_the_saved_transform(self):
        base=self.research.base
        full=self.bundle['features'].transform(self.frame)
        expected=base.svd.transform(base._text(self.frame))
        np.testing.assert_allclose(full[:,self.groups['title_ngrams']['columns']],expected,rtol=1e-5,atol=1e-6)


if __name__=='__main__':unittest.main()
