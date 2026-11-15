import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.predict import load_bundle, predict_scores, parse_when
from src.config import ROOT, TESTRUN_CHANNEL_ID


class V4PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle('v4')
        cls.when = parse_when('2026-11-16 10:00')

    def test_title_prior_bank_has_no_held_out_or_testrun_outcomes(self):
        features = self.bundle['features']
        self.assertFalse(set(features.ids) & set(self.bundle['test_ids']))
        self.assertFalse(set(features.ids) & set(self.bundle['selection_ids']))
        self.assertFalse(set(features.ids) & set(self.bundle['validation_ids']))
        self.assertNotIn(TESTRUN_CHANNEL_ID, set(features.channels))
        for title in set(features.bank_titles):
            folds = features.fold_assignments[features.bank_titles == title]
            self.assertEqual(len(set(folds)), 1)

    def test_comparison_with_image_stays_local_and_has_no_false_tie_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'frame.jpg'
            Image.new('RGB', (320, 180), (20, 60, 130)).save(path)
            out = predict_scores(self.bundle, ['Same title', 'Same title'], 30, self.when, thumb_paths=[path])
            self.assertEqual(out.ranking_score.nunique(), 1)
            self.assertTrue((out.comparison_reliability == .5).all())
            self.assertTrue(np.isfinite(out.relative_score).all())
            self.assertNotIn('torch', sys.modules)
            self.assertNotIn('open_clip', sys.modules)

    def test_v4_cli_single_title_json_and_baseline_override(self):
        result = subprocess.run([sys.executable, str(ROOT/'scripts/predict.py'), 'Example title',
                                 '--model', 'v4', '--json', '--recent-median-views', '100'],
                                capture_output=True, text=True, check=True)
        def reject(value):
            raise AssertionError(f'Invalid JSON constant: {value}')
        data = json.loads(result.stdout, parse_constant=reject)
        row = data['drafts'][0]
        self.assertEqual(data['model_version'], 'v4')
        self.assertIsNone(row['comparison_reliability'])
        self.assertEqual(row['channel_median_views'], 100)
        self.assertGreaterEqual(row['implied_views_at_14d'], 0)


if __name__ == '__main__':
    unittest.main()
