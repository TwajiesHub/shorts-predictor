"""V4 research candidates: lexical hooks and leakage-controlled title peer priors.

Only sparse bag-of-words, dictionaries, and scalar summaries at inference.
No API calls, video descriptions, heavy embeddings, or new dependencies.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from src.ranking import RankFeatures

ROOT = Path(__file__).resolve().parents[1]
TOKENS = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", re.I)


def normalized_title(title):
    """Group repeated titles regardless of hashtag, punctuation, case or spacing."""
    return ' '.join(TOKENS.findall(re.sub(r'#\w+', ' ', str(title).lower())))


def language_features(title, valence):
    title = str(title)
    words = TOKENS.findall(title.lower())
    n = max(len(words), 1)
    scores = np.array([valence.get(w, 0.) for w in words])
    first = TOKENS.findall(title[:45].lower())
    # Specificity, information-gap and contradiction *candidates*, not universal
    # claims that clickbait or negative words cause Shorts to perform better.
    return np.array([
        np.maximum(scores, 0).sum()/n, -np.minimum(scores, 0).sum()/n,
        (scores > 0).sum()/n, (scores < 0).sum()/n,
        np.abs(scores).max(initial=0),
        sum(valence.get(w, 0) < 0 for w in first)/max(len(first), 1),
        sum(w in {'not', 'no', 'never', "isn't", "don't", "doesn't", "can't", 'without'} for w in words),
        sum(w in {'this', 'these', 'that', 'it', 'here', 'something'} for w in words)/n,
        sum(w in {'because', 'actually', 'instead', 'but', 'wrong', 'lie', 'lying', 'myth', 'truth'} for w in words)/n,
        sum(w in {'you', 'your', "you're"} for w in words)/n,
        sum(w in {'i', 'my', 'we', 'our'} for w in words)/n,
        sum(w in {'why', 'how', 'what', 'which'} for w in first)/max(len(first), 1),
        bool(re.search(r'^\s*\d+\b', title)),
        len(re.findall(r'\d+(?:\.\d+)?\s*(?:gb|tb|mb|mah|w|hz|fps|%|seconds?|minutes?|days?)\b', title, re.I)),
        bool(re.search(r'\b(?:vs\.?|versus|better than|instead of)\b', title, re.I)),
        bool(re.search(r'\b(?:in \d+|under \d+|only \d+|\d+ ways|\d+ reasons|\d+ tips)\b', title, re.I)),
        title.count(':'), title.count('(')+title.count('['),
        np.mean([len(w) for w in words]) if words else 0,
        sum(len(w) >= 8 for w in words)/n, len(set(words))/n,
        sum(c.isupper() for c in title[:45])/max(len(title[:45]), 1),
        max(len(title)-45, 0)/max(len(title), 1),
    ], dtype=np.float32)


class ResearchFeatures:
    """Compose unchanged v3 features with independently selectable candidate blocks.

    Label-derived peer features for fitting must use training_matrix(), which
    supplies group-fold out-of-fold values. transform() is for new/held-out rows.
    Duplicate normalized titles are assigned to the same cross-fitting fold.
    """
    def __init__(self, groups=('language', 'peers', 'novelty'), base=None):
        self.groups = tuple(groups)
        self.base = base

    @property
    def history(self):
        return self.base.history

    def history_context(self, row):
        return self.base.history_context(row)

    def fit(self, df, y):
        if self.base is None:
            self.base = RankFeatures().fit(df)
        elif set(self.base.history.video_id) != set(df.video_id):
            raise ValueError('V3 base and v4 fitting cohorts differ')
        self.valence = {}
        if 'language' in self.groups:
            self.valence = json.loads((ROOT/'data/lexicons/vader_valence.json').read_text())
        self.ids = df.video_id.astype(str).to_numpy()
        self.channels = df.channel_id.astype(str).to_numpy()
        self.targets = np.asarray(y, dtype=np.float64)
        self.bank_titles = np.array([normalized_title(t) for t in df.title])
        self.oof = np.empty((len(df), self._lookup_width()), np.float32)
        if 'peers' in self.groups or 'novelty' in self.groups:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=12000,
                                            sublinear_tf=True, stop_words='english', dtype=np.float32)
            self.bank = self.vectorizer.fit_transform(self.bank_titles).tocsr()
            self.ridge = Ridge(alpha=20., solver='lsqr').fit(self.bank, self.targets)
            nfolds = min(5, len(np.unique(self.bank_titles)))
            if nfolds < 2:
                raise ValueError('Need at least two distinct normalized titles for cross-fitting')
            folds = GroupKFold(n_splits=nfolds)
            for fit, held in folds.split(self.bank, groups=self.bank_titles):
                ridge = Ridge(alpha=20., solver='lsqr').fit(self.bank[fit], self.targets[fit])
                self.oof[held] = self._lookup(df.iloc[held], indices=fit, ridge=ridge)
            self.fold_assignments = np.empty(len(df), dtype=int)
            for fold, (_, held) in enumerate(folds.split(self.bank, groups=self.bank_titles)):
                self.fold_assignments[held] = fold
        else:
            self.oof = np.empty((len(df), 0), np.float32)
            self.fold_assignments = np.full(len(df), -1)
        self.training_positions = {vid: i for i, vid in enumerate(self.ids)}
        return self

    def _lookup_width(self):
        return (5 if 'peers' in self.groups else 0) + (5 if 'novelty' in self.groups else 0)

    def _lookup(self, df, indices=None, ridge=None):
        if self._lookup_width() == 0:
            return np.empty((len(df), 0), np.float32)
        indices = np.arange(len(self.ids)) if indices is None else indices
        bank = self.bank[indices]
        targets = self.targets[indices]
        channels = self.channels[indices]
        ids = self.ids[indices]
        query = self.vectorizer.transform([normalized_title(t) for t in df.title])
        similarities = (query @ bank.T).toarray()
        predictions = (self.ridge if ridge is None else ridge).predict(query)
        rows = []
        for pos, (_, row) in enumerate(df.iterrows()):
            sims = similarities[pos]
            # Never retrieve a row's own outcome, even if transform is called on
            # an existing training video during exploratory use.
            if pd.notna(row.get('video_id')):
                sims[ids == str(row.video_id)] = 0
            top = np.argsort(-sims, kind='stable')[:32]
            weights = np.where(sims[top] >= .15, sims[top] ** 2, 0)
            # Shrink sparse neighborhoods towards zero channel-relative residual.
            prior = np.dot(weights, targets[top])/(1.+weights.sum())
            local = channels[top] == str(row.get('channel_id'))
            lw = weights * local
            same = np.flatnonzero(channels == str(row.get('channel_id')))
            local_sims = sims[same] if len(same) else np.zeros(1)
            values = []
            if 'peers' in self.groups:
                values += [predictions[pos], prior, np.dot(lw, targets[top])/(1.+lw.sum()),
                           weights.sum(), lw.sum()]
            if 'novelty' in self.groups:
                values += [float(sims.max(initial=0)), float(local_sims.max(initial=0)),
                           float(np.mean(np.sort(local_sims)[-5:])),
                           float(query[pos].nnz == 0), float(len(same) > 0)]
            rows.append(values)
        return np.asarray(rows, np.float32)

    def _combine(self, df, lookup):
        blocks = [self.base.transform(df)]
        if 'language' in self.groups:
            blocks.append(np.stack([language_features(t, self.valence) for t in df.title]))
        if self._lookup_width():
            blocks.append(lookup)
        return np.nan_to_num(np.hstack(blocks)).astype(np.float32)

    def training_matrix(self, df):
        positions = [self.training_positions[str(vid)] for vid in df.video_id]
        return self._combine(df, self.oof[positions])

    def transform(self, df):
        return self._combine(df, self._lookup(df))
