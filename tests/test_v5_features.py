import unittest

import numpy as np
import pandas as pd

from scripts.collect_v5_public import prepare_rows, video_id_from_url, iso_duration
from src.emotional_triggers import emotion_scores, emotion_matrix
from src.public_priors import ExpandedTitlePriors


class V5FeatureTests(unittest.TestCase):
    def test_trigger_rules_are_bounded_and_interpretable(self):
        neutral = emotion_scores('A plain statement')
        self.assertEqual(len(neutral),10)
        self.assertEqual(emotion_scores('YouTube tutorial')['direct_address'],0)
        self.assertGreater(emotion_scores('Why does your phone do this?')['curiosity_gap'],0)
        self.assertGreater(emotion_scores('YOUR phone')['direct_address'],0)
        self.assertEqual(emotion_scores('Only 128GB of storage')['urgency_scarcity'],0)
        self.assertGreater(emotion_scores('Only 3 left today')['urgency_scarcity'],0)
        self.assertGreater(emotion_scores('BRUH 😂')['humor'],0)
        scores = emotion_matrix(['','secret! '*100,'3 surprising facts you need to know'])
        self.assertEqual(scores.shape,(3,13))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_collector_excludes_benchmark_outcomes_before_source_baselines(self):
        rows = [dict(video_id=f'{i:011d}',title=f'Battery tech example {i}',channel_id='external',
                published_at=f'2025-01-{i+1:02d}T00:00:00Z',duration=30,views=10*(i+1),category_name='') for i in range(6)]
        original = pd.DataFrame([dict(video_id='00000000005',title='Excluded benchmark row',channel_id='heldout')])
        rows[-1]['views'] = 10**12
        out,_ = prepare_rows(rows,original)
        self.assertEqual(len(out),5)
        self.assertTrue((out.cohort_median_views == 30).all())
        self.assertNotIn('description',out.columns)

    def test_public_url_and_duration_parsing(self):
        self.assertEqual(video_id_from_url('https://www.youtube.com/shorts/abcdefghijk'),'abcdefghijk')
        self.assertIsNone(video_id_from_url('https://example.com/shorts/abcdefghijk'))
        self.assertEqual(iso_duration('PT1M30S'),90)
        self.assertIsNone(iso_duration('garbage'))

    def test_expanded_prior_cross_fit_excludes_duplicate_group_labels(self):
        fitting = pd.DataFrame([dict(video_id=f'f{i}',title=f'Battery charging tip {i//2}',channel_id='a') for i in range(40)])
        external = pd.DataFrame([dict(video_id=f'e{i}',title=f'Phone storage tip {i}',channel_id='b',prior_target=i/10) for i in range(10)])
        y = np.linspace(-1,1,40)
        a = ExpandedTitlePriors().fit(fitting,y,external)
        altered = y.copy();altered[:2] = 1000
        b = ExpandedTitlePriors().fit(fitting,altered,external)
        np.testing.assert_allclose(a.oof[:2],b.oof[:2],atol=1e-6)
        self.assertEqual(a.fold_assignments[0],a.fold_assignments[1])
        self.assertTrue((a.source_weights[a.is_external] == .35).all())


class LinearCorrectionTests(unittest.TestCase):
    def test_regularized_pairwise_correction_learns_order(self):
        from src.public_priors import LinearRankCorrection
        class Zero:
            def predict(self,X):return np.zeros(len(X))
        X = np.column_stack([np.zeros(100),np.linspace(-2,2,100)])
        y = X[:,1]
        model = LinearRankCorrection(Zero(),1,l2=.1).fit(X,y,np.repeat('a',100))
        self.assertLess(model.final_loss,model.initial_loss)
        self.assertGreater(model.predict(X)[-1],model.predict(X)[0])


class ApiCollectorTests(unittest.TestCase):
    def test_api_path_filters_and_accounts_without_live_requests(self):
        import io,json
        from datetime import datetime,timedelta,timezone
        from unittest.mock import patch
        from scripts.collect_v5_public import fetch_api
        published = (datetime.now(timezone.utc)-timedelta(days=10)).isoformat()
        payloads = [
            {'items':[{'contentDetails':{'relatedPlaylists':{'uploads':'UUexample'}}}]},
            {'items':[{'contentDetails':{'videoId':'abcdefghijk'}}]},
            {'items':[{'id':'abcdefghijk','snippet':{'title':'Battery tip','channelId':'channel',
              'publishedAt':published,'categoryId':'28','description':'Never persist this field'},
              'contentDetails':{'duration':'PT30S'},'statistics':{'viewCount':'123'}}]},
        ]
        responses = [io.BytesIO(json.dumps(p).encode()) for p in payloads]
        with patch('scripts.collect_v5_public.urlopen',side_effect=responses), patch('scripts.collect_v5_public.time.sleep'):
            rows,usage = fetch_api(['channel'],'unit-test-value')
        self.assertEqual(len(rows),1)
        self.assertNotIn('description',rows[0])
        self.assertEqual(usage['successful_request_quota_units'],3)
        self.assertEqual(usage['search_requests'],0)


if __name__ == "__main__":
    unittest.main()
