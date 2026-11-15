import json
import subprocess
import sys
import unittest

import numpy as np

from scripts.predict import load_bundle, predict_scores, parse_when
from src.config import ROOT, TESTRUN_CHANNEL_ID
from src.emotional_triggers import emotion_scores


class V5PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle('v5')
        cls.when = parse_when('2026-11-16 10:00')

    def test_explanations_are_cues_not_a_claim_of_model_influence(self):
        out = predict_scores(self.bundle,['Why is your phone doing this?'],30,self.when)
        self.assertEqual(out.iloc[0].emotional_triggers,emotion_scores(out.iloc[0].title))
        self.assertEqual(out.iloc[0].emotional_triggers_used_for_ranking,
                         'emotions' in self.bundle['selected_groups'])
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('open_clip',sys.modules)

    def test_retention_guard_preserves_v4_when_additions_are_rejected(self):
        titles = ['Why is my battery low?','3 tricks to charge your phone']
        out = predict_scores(self.bundle,titles,30,self.when).set_index('title')
        if not self.bundle['selected_groups']:
            old = predict_scores(load_bundle('v4'),titles,30,self.when).set_index('title')
            np.testing.assert_allclose(out.loc[titles].ranking_score,old.loc[titles].ranking_score,atol=1e-10)
        self.assertTrue(np.isfinite(out.relative_score).all())
        self.assertNotIn(TESTRUN_CHANNEL_ID,set(self.bundle['features'].history.channel_id))

    def test_v5_cli_json_stays_strict(self):
        run = subprocess.run([sys.executable,str(ROOT/'scripts/predict.py'),'Only 3 left today!',
                              '--model','v5','--json'],capture_output=True,text=True,check=True)
        def reject(value):
            raise AssertionError(f'Invalid JSON constant {value}')
        result = json.loads(run.stdout,parse_constant=reject)
        self.assertEqual(result['model_version'],'v5')
        row = result['drafts'][0]
        self.assertIsNone(row['comparison_reliability'])
        self.assertGreater(row['emotional_triggers']['urgency_scarcity'],0)


if __name__ == '__main__':
    unittest.main()
