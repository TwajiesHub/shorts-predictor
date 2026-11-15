"""Small pairwise tree ranker and pre-post features. No torch/CLIP inference."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from src.features import build_numeric_frame, EMOJI_RE, NUMERIC_FEATURES
from src.thumb_features import INTERPRETABLE_FEATURES, interpretable_features

THUMB_COLS = [c for c in INTERPRETABLE_FEATURES if not c.startswith('clip_')]
POWER = {'secret', 'hidden', 'mistake', 'never', 'stop', 'free', 'easy', 'best', 'worst', 'why', 'surprising', 'actually'}


def sample_pairs(y, channels, count=30000, seed=42):
    """Equal sampling per channel, then per video; never compare a video to itself."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(np.asarray(channels) == c) for c in np.unique(channels)]
    groups = [g for g in groups if len(g) > 1]
    if not groups:
        raise ValueError('Need at least one channel with two distinct outcomes')
    i, j = [], []
    for g in groups:
        n = max(1, count // len(groups))
        a = rng.integers(len(g), size=n)
        b = (a + rng.integers(1, len(g), size=n)) % len(g)
        i.extend(g[a]); j.extend(g[b])
    i, j = np.asarray(i), np.asarray(j)
    keep = np.asarray(y)[i] != np.asarray(y)[j]
    if not keep.any():
        raise ValueError('Need unequal within-channel outcomes')
    return i[keep], j[keep]


class PairwiseRanker:
    """Gradient boosting on channel-balanced RankNet logistic pair loss.

    Each shallow tree fits the negative gradient aggregated at video level.
    Normalizing by pair appearances prevents large channels dominating updates.
    """
    def __init__(self, n_estimators=180, learning_rate=0.15, seed=42, initial_model=None, gap_weighted=False):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.seed = seed
        self.initial_model = initial_model
        self.gap_weighted = gap_weighted

    def fit(self, X, y, channels):
        self.trees_ = []
        scores = self.initial_model.predict(X) if self.initial_model is not None else np.zeros(len(y))
        for step in range(self.n_estimators):
            i, j = sample_pairs(y, channels, seed=self.seed + step)
            sign = np.sign(y[i] - y[j])
            grad = sign * expit(-sign * (scores[i] - scores[j]))
            if self.gap_weighted:
                # Tiny view differences are noisy ranking labels. Cap large gaps
                # so one viral observation cannot dominate the RankNet objective.
                grad *= np.minimum(np.abs(y[i] - y[j]), 3.) / 3.
            counts = np.bincount(np.r_[i, j], minlength=len(y))
            sums = np.bincount(i, grad, minlength=len(y)) - np.bincount(j, grad, minlength=len(y))
            target = sums / np.maximum(counts, 1)
            tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=35, max_features=0.85,
                                         random_state=self.seed + step)
            tree.fit(X, target, sample_weight=counts.astype(float))
            scores += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        initial = self.initial_model.predict(X) if self.initial_model is not None else np.zeros(len(X))
        return sum((self.learning_rate * tree.predict(X) for tree in self.trees_), initial)


class RankFeatures:
    """Fixed-size title hashing compressed to 64 dimensions + cheap traits.

    History contains fitting rows only. Recent context uses earlier *exact* dates;
    coarse dates use leave-self-out channel context, never an invented chronology.
    Outcomes are snapshot views, not historical point-in-time observations.
    """
    def __init__(self, text=True, thumbnail=True, context=True):
        self.text, self.thumbnail, self.context = text, thumbnail, context
        self.word = HashingVectorizer(n_features=8192, ngram_range=(1, 2), alternate_sign=False)
        self.char = HashingVectorizer(n_features=8192, analyzer='char_wb', ngram_range=(3, 5), alternate_sign=False)

    def _text(self, df):
        from scipy.sparse import hstack
        titles = df.title.fillna('').astype(str)
        return hstack([self.word.transform(titles), self.char.transform(titles)]).tocsr()

    def fit(self, df):
        self.history = df[['video_id', 'channel_id', 'duration', 'title', 'views', 'published_ts', 'hour_is_approximate']].copy()
        self.global_views = float(df.views.median())
        if self.text:
            self.svd = TruncatedSVD(n_components=min(64, len(df) - 1), random_state=42)
            self.svd.fit(self._text(df))
        raw = self._dense(df)
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(raw)
        self.scaler = StandardScaler().fit(self.imputer.transform(raw))
        return self

    def history_context(self, row):
        if pd.notna(row.get('recent_median_views')):
            return (float(row.recent_median_views), float(row.get('recent_duration', 30)),
                    float(row.get('recent_title_length', 50)), float(row.get('recent_count', 0)))
        history = self.history[self.history.channel_id == row.get('channel_id')]
        if 'video_id' in row:
            history = history[history.video_id != row.video_id]
        exact = row.get('hour_is_approximate') == False  # noqa: E712
        if exact and pd.notna(row.get('published_ts')):
            history = history[(history.hour_is_approximate == False) &  # noqa: E712
                              (history.published_ts < row.published_ts)].sort_values('published_ts').tail(20)
        if history.empty:
            return self.global_views, 30., 50., 0
        return (float(history.views.median()), float(history.duration.median()),
                float(history.title.str.len().median()), len(history))

    def _dense(self, df):
        numeric = build_numeric_frame(df).to_numpy(float)
        rows = []
        for (_, row), num in zip(df.iterrows(), numeric):
            title = str(row.get('title') or '')
            words = re.findall(r'\w+', title.lower())
            # Count emoji code points rather than contiguous emoji runs.
            extras = [title.count('?'), len(re.findall(r'\d+', title)),
                      sum(len(s) for s in EMOJI_RE.findall(title)),
                      sum(w in POWER for w in words), len(title) < 35, 35 <= len(title) < 65,
                      len(title) >= 65, sum(c.isupper() for c in title) / max(len(title), 1)]
            # build_numeric_frame masks coarse dates. Cyclical hour is also masked.
            names = NUMERIC_FEATURES
            hour = num[names.index('upload_hour')]
            known = 1 - num[names.index('hour_missing')]
            extras += [known * np.sin(2*np.pi*hour/24), known * np.cos(2*np.pi*hour/24)]
            if self.context:
                baseline, dur, length, count = self.history_context(row)
                logsubs = num[names.index('log_subscribers')]
                tier = np.digitize(np.expm1(logsubs), [1000, 10000, 100000, 1000000])
                extras += [np.log1p(baseline), count, float(row.duration)/max(dur, 1),
                           len(title)-length, tier, tier * extras[3], tier * len(title)/100]
            if self.thumbnail:
                vals = np.array([row.get(c, np.nan) for c in THUMB_COLS], float)
                v = dict(zip(THUMB_COLS, vals))
                # Exposure-adjusted contrast and face prominence are computable from
                # both old caches and new images: no train/serve feature mismatch.
                extras += list(vals) + [
                    v['thumb_contrast'] / (0.05 + v['thumb_brightness']),
                    v['thumb_face_area_ratio'] / max(v['thumb_face_count'], 1),
                    v['thumb_edge_density'] * v['thumb_colorfulness'] / 100,
                    1 - v['thumb_dark_frac'] - v['thumb_bright_frac'],
                    float(np.isfinite(vals).any())]
            rows.append(np.r_[num, extras])
        return np.asarray(rows, dtype=np.float32)

    def transform(self, df):
        dense = self.scaler.transform(self.imputer.transform(self._dense(df)))
        if self.text:
            dense = np.hstack([dense, self.svd.transform(self._text(df))])
        return np.nan_to_num(dense).astype(np.float32)


def attach_thumbnails(df, paths=None, cache=None):
    """Explicit paths take precedence. Never silently replace a bad image with cache."""
    out = df.copy()
    if paths is not None:
        if len(paths) != len(out):
            raise ValueError('Thumbnail paths must match the number of rows')
        import cv2
        cv2.setNumThreads(min(2, cv2.getNumThreads()))
        measurements = {}
        for idx, path in zip(out.index, paths):
            key = str(path) if path is not None else None
            if key not in measurements:
                measurements[key] = interpretable_features(path) if path is not None else None
            feats = measurements[key]
            if path is not None and feats is None:
                raise ValueError(f'Cannot decode thumbnail: {path}')
            for col in THUMB_COLS:
                out.loc[idx, col] = (feats or {}).get(col, np.nan)
    elif cache is not None and 'video_id' in out:
        out = out.drop(columns=THUMB_COLS, errors='ignore').merge(
            cache[['video_id'] + THUMB_COLS], on='video_id', how='left', validate='many_to_one')
    for col in THUMB_COLS:
        if col not in out:
            out[col] = np.nan
    return out


def predict_frame(bundle, df):
    with threadpool_limits(limits=2):
        X = bundle['features'].transform(df)
        raw = bundle['ranker'].predict(X)
        relative = bundle['calibration'].predict(raw.reshape(-1, 1))
        reference = bundle['reference_rank_scores']
        percentile = 100 * np.searchsorted(reference, raw, side='right') / len(reference)
    return raw, relative, percentile



def draft_output(bundle, df):
    raw, relative, percentile = predict_frame(bundle, df)
    out = df[[c for c in ['title', 'thumbnail'] if c in df]].copy()
    out['ranking_score'] = raw
    out['score'] = relative
    out['relative_score'] = relative
    out['niche_percentile'] = percentile
    # Pair reliability is empirically calibrated on inner-validation pairs, not
    # a claim that an individual video's eventual view count is predictable.
    order = np.argsort(-raw, kind='stable')
    competitor = np.full(len(raw), np.nan)
    if len(raw) > 1:
        for idx in range(len(raw)):
            other = order[0] if idx != order[0] else order[1]
            gap = abs(raw[idx] - raw[other])
            competitor[idx] = (.5 if gap < 1e-10 else
                bundle['pair_calibration'].predict_proba([[gap]])[0, 1])
    out['comparison_reliability'] = competitor
    notes = []
    for pos, (_, row) in enumerate(df.iterrows()):
        reasons = []
        if not np.isfinite(competitor[pos]) or competitor[pos] < .65:
            reasons.append('weak or unavailable pair evidence')
        if row[THUMB_COLS].isna().all():
            reasons.append('thumbnail missing')
        if not str(row.get('channel_id')) in set(bundle['features'].history.channel_id):
            reasons.append('channel unseen during fitting')
        if bundle['features'].history_context(row)[3] < 5:
            reasons.append('little usable channel history')
        if len(str(row.title)) > bundle['title_length_p99']:
            reasons.append('unusual title length')
        notes.append('; '.join(reasons) or 'within observed feature support; outcomes remain noisy')
    out['reliability_note'] = notes
    out['confidence'] = ['low' if n.startswith('weak') or 'unseen' in n or 'unusual' in n or 'little' in n else 'moderate' for n in notes]
    out['relative_error_80pct'] = bundle['relative_error_80pct']
    out['rank'] = out.ranking_score.rank(ascending=False, method='first').astype(int)
    medians = [bundle['features'].history_context(row)[0] for _, row in df.iterrows()]
    out['channel_median_views'] = medians
    out['implied_views_at_14d'] = np.maximum(0, np.expm1(np.clip(relative + np.log1p(medians) +
        bundle['age_a'] + bundle['age_b'] * np.log1p(14), -20, 30)))
    out['better_than_your_typical'] = relative > 0
    return out.sort_values('rank').reset_index(drop=True)
