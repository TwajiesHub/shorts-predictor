import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.predict import load_bundle, parse_when, predict_scores
from src.config import ROOT, TESTRUN_CHANNEL_ID


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle('v3')
        cls.when = parse_when('2026-11-16 10:00')
        cv2.setNumThreads(2)

    def test_calibration_and_selection_are_disjoint(self):
        b = self.bundle
        cohorts = [set(b[k]) for k in ('fitting_ids', 'selection_ids', 'validation_ids', 'test_ids')]
        for i, cohort in enumerate(cohorts):
            self.assertTrue(cohort)
            for other in cohorts[i+1:]:
                self.assertFalse(cohort & other)
        self.assertNotIn(TESTRUN_CHANNEL_ID, set(b['features'].history.channel_id))

    def test_identical_drafts_have_no_ordering_evidence(self):
        out = predict_scores(self.bundle, ['Same title', 'Same title'], 30, self.when)
        self.assertEqual(out.ranking_score.nunique(), 1)
        self.assertTrue((out.comparison_reliability == .5).all())
        self.assertTrue((out.confidence == 'low').all())

    def test_local_thumbnail_comparison_preserves_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/'a.jpg', Path(tmp)/'b.jpg']
            Image.new('RGB', (320, 180), (40, 80, 120)).save(paths[0])
            Image.new('RGB', (180, 320), (240, 180, 40)).save(paths[1])
            out = predict_scores(self.bundle, ['Example', 'Example'], 30, self.when, thumb_paths=paths)
            self.assertEqual(set(out.thumbnail), set(map(str, paths)))
            self.assertTrue(np.isfinite(out.ranking_score).all())
            self.assertNotIn('torch', sys.modules)
            self.assertNotIn('open_clip', sys.modules)

    def test_cli_single_title_json_is_strict(self):
        result = subprocess.run([sys.executable, str(ROOT/'scripts/predict.py'), 'Example title',
                                 '--model', 'v3', '--json', '--recent-median-views', '0'],
                                capture_output=True, text=True, check=True)
        def reject(value):
            raise AssertionError(f'Invalid JSON constant {value}')
        data = json.loads(result.stdout, parse_constant=reject)
        row = data['drafts'][0]
        self.assertIsNone(row['comparison_reliability'])
        self.assertGreaterEqual(row['implied_views_at_14d'], 0)
        self.assertEqual(row['channel_median_views'], 0)


if __name__ == '__main__':
    unittest.main()
