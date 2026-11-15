import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.ranking import PairwiseRanker, RankFeatures, sample_pairs, attach_thumbnails, THUMB_COLS
from scripts.train_v3 import exact_accuracy


class RankingTests(unittest.TestCase):
    def test_pairs_never_cross_channels_and_ignore_ties(self):
        y = np.array([0., 0., 1., 5., 9.])
        channels = np.array(['a', 'a', 'a', 'b', 'b'])
        i, j = sample_pairs(y, channels, count=1000)
        self.assertTrue(np.all(channels[i] == channels[j]))
        self.assertTrue(np.all(i != j))
        self.assertTrue(np.all(y[i] != y[j]))

    def test_ranker_learns_order_instead_of_absolute_scale(self):
        X = np.linspace(-2, 2, 120)[:, None]
        y = X[:, 0] ** 3
        channels = np.repeat('a', 120)
        model = PairwiseRanker(n_estimators=40).fit(X, y, channels)
        scores = model.predict(X)
        self.assertGreater(exact_accuracy(y, scores)[0], .9)

    def test_pairwise_loss_is_invariant_to_channel_offsets(self):
        X = np.linspace(-2, 2, 120)[:, None]
        y = X[:, 0] ** 3
        channels = np.repeat(['a', 'b'], 60)
        shifted = y + np.where(channels == 'a', 100., -100.)
        a = PairwiseRanker(n_estimators=10, gap_weighted=True).fit(X, y, channels)
        b = PairwiseRanker(n_estimators=10, gap_weighted=True).fit(X, shifted, channels)
        np.testing.assert_allclose(a.predict(X), b.predict(X), atol=1e-6)

    def test_tie_scoring(self):
        self.assertEqual(exact_accuracy(np.array([1., 2.]), np.array([0., 0.]))[0], .5)
        self.assertIsNone(exact_accuracy(np.array([1., 1.]), np.array([0., 1.]))[0])

    def test_context_excludes_self_future_and_other_channels(self):
        features = RankFeatures()
        features.global_views = 12.
        features.history = pd.DataFrame([
            dict(video_id='old', channel_id='a', views=10, duration=20, title='old', published_ts=1, hour_is_approximate=False),
            dict(video_id='self', channel_id='a', views=999, duration=20, title='self', published_ts=2, hour_is_approximate=False),
            dict(video_id='future', channel_id='a', views=9999, duration=20, title='future', published_ts=3, hour_is_approximate=False),
        ])
        row = pd.Series(dict(video_id='self', channel_id='a', published_ts=2, hour_is_approximate=False))
        self.assertEqual(features.history_context(row)[0], 10)
        self.assertEqual(features.history_context(pd.Series(dict(channel_id='unseen')))[0], 12)

    def test_coarse_time_is_masked_and_emoji_count_is_per_codepoint(self):
        f = RankFeatures(text=False, thumbnail=False, context=False)
        df = pd.DataFrame([dict(title='Secret? 12 😀😀', duration=30, channel_subscribers=100,
                               published_ts=100000, hour_is_approximate=True)])
        raw = f._dense(df)
        self.assertEqual(raw[0, -2:].tolist(), [0., 0.])
        from src.features import NUMERIC_FEATURES
        self.assertEqual(raw[0, len(NUMERIC_FEATURES)+2], 2)

    def test_exact_posting_time_uses_local_cycle(self):
        f = RankFeatures(text=False, thumbnail=False, context=False)
        ts = datetime(2026, 11, 16, 9, tzinfo=timezone.utc).timestamp()
        df = pd.DataFrame([dict(title='Example', duration=30, channel_subscribers=100,
                               published_ts=ts, hour_is_approximate=False)])
        np.testing.assert_allclose(f._dense(df)[0, -2:], [np.sin(2*np.pi*10/24), np.cos(2*np.pi*10/24)], atol=1e-6)

    def test_thumbnail_network_allowlist(self):
        from src.thumb_features import is_allowed_thumb_url
        self.assertTrue(is_allowed_thumb_url('https://i.ytimg.com/vi/example/hqdefault.jpg'))
        for url in ['https://amazon.com/x', 'https://i.ytimg.com.evil.test/x', 'https://example.com/x', 'file:///tmp/x']:
            self.assertFalse(is_allowed_thumb_url(url))

    def test_missing_thumbnails_and_bad_input(self):
        df = pd.DataFrame([dict(title='hello')])
        out = attach_thumbnails(df, paths=[None])
        self.assertTrue(out[THUMB_COLS].isna().all().all())
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            attach_thumbnails(df, paths=[Path(tmp) / 'nonexistent.jpg'])


if __name__ == '__main__':
    unittest.main()
