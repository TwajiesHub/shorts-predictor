import unittest

import numpy as np
import pandas as pd

from src.research_features import ResearchFeatures, language_features, normalized_title


class ZeroBase:
    def __init__(self, frame):
        self.history = frame

    def transform(self, frame):
        return np.zeros((len(frame), 1), np.float32)


class ResearchFeatureTests(unittest.TestCase):
    @staticmethod
    def frame():
        # Repeated normalized titles test group-level cross-fitting, not just row exclusion.
        return pd.DataFrame([dict(video_id=str(i), title=f'iPhone battery charging tip {i//2}',
                                  channel_id='a' if i % 3 else 'b') for i in range(40)])

    def test_title_normalization_groups_cosmetic_duplicates(self):
        self.assertEqual(normalized_title('Why This Works? #Shorts'), normalized_title('WHY this works!'))

    def test_lexical_candidates_are_finite_for_empty_input(self):
        self.assertTrue(np.isfinite(language_features('', {})).all())
        a = language_features('bad bad phone', {'bad': -2.5})
        b = language_features('good good phone', {'good': 2.5})
        self.assertGreater(a[1], b[1])
        self.assertGreater(b[0], a[0])

    def test_oof_prior_excludes_own_and_duplicate_title_labels(self):
        df = self.frame()
        y = np.linspace(-1, 1, len(df))
        a = ResearchFeatures(groups=('peers', 'novelty'), base=ZeroBase(df)).fit(df, y)
        changed = y.copy(); changed[:2] = 1000
        b = ResearchFeatures(groups=('peers', 'novelty'), base=ZeroBase(df)).fit(df, changed)
        np.testing.assert_allclose(a.oof[:2], b.oof[:2], atol=1e-6)
        self.assertEqual(a.fold_assignments[0], a.fold_assignments[1])
        self.assertEqual(a.training_matrix(df.iloc[[1, 0]]).shape, (2, 11))

    def test_neighbor_retrieval_masks_own_row(self):
        df = self.frame()
        y = np.zeros(len(df)); y[0] = 100
        features = ResearchFeatures(groups=('peers',), base=ZeroBase(df)).fit(df, y)
        # The second peer feature is the neighborhood target prior (not the Ridge score).
        self.assertEqual(features._lookup(df.iloc[:1])[0, 1], 0)


if __name__ == '__main__':
    unittest.main()
