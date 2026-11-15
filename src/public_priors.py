"""V5: cross-fitted, source-weighted public title priors + interpretable cues."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from src.emotional_triggers import load_emotion_rules, emotion_scores, emotion_matrix
from src.research_features import normalized_title


class ExpandedTitlePriors:
    """Additional public outcomes influence priors, never original head labels.

    External examples are downweighted to .35 because their snapshot and sampling
    differ. Group-fold exclusion spans the complete bank, including external rows.
    """
    def fit(self, fitting, targets, external):
        fields = ['video_id','title','channel_id']
        combined = pd.concat([fitting[fields], external[fields]], ignore_index=True)
        self.ids = combined.video_id.astype(str).to_numpy()
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('Prior bank contains duplicate video IDs')
        self.channels = combined.channel_id.astype(str).to_numpy()
        self.targets = np.r_[targets, external.prior_target.to_numpy(float)]
        self.is_external = np.arange(len(combined)) >= len(fitting)
        self.source_weights = np.where(self.is_external, .35, 1.)
        self.bank_titles = combined.title.map(normalized_title).to_numpy()
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=16000,
            sublinear_tf=True, stop_words='english', dtype=np.float32)
        self.bank = self.vectorizer.fit_transform(self.bank_titles).tocsr()
        self.ridge = self._ridge(np.arange(len(combined)))
        folds = GroupKFold(n_splits=min(5,len(set(self.bank_titles))))
        self.fold_assignments = np.empty(len(combined), int)
        self.oof = np.empty((len(fitting),7), np.float32)
        for fold, (train, valid) in enumerate(folds.split(self.bank, groups=self.bank_titles)):
            self.fold_assignments[valid] = fold
            original = valid[valid < len(fitting)]
            if len(original):
                self.oof[original] = self.transform(fitting.iloc[original], train, self._ridge(train))
        self.positions = {str(vid):i for i,vid in enumerate(fitting.video_id)}
        return self

    def _ridge(self, indices):
        return Ridge(alpha=20.,solver='lsqr').fit(self.bank[indices], self.targets[indices],
                                                sample_weight=self.source_weights[indices])

    def transform(self, frame, indices=None, model=None):
        indices = np.arange(len(self.ids)) if indices is None else indices
        query = self.vectorizer.transform(frame.title.map(normalized_title))
        similarities = (query @ self.bank[indices].T).toarray()
        raw = (self.ridge if model is None else model).predict(query)
        values = []
        for pos, (_, row) in enumerate(frame.iterrows()):
            sims = similarities[pos]
            if pd.notna(row.get('video_id')):
                sims[self.ids[indices] == str(row.video_id)] = 0
            local = np.argsort(-sims, kind='stable')[:32]
            selected = indices[local]
            weights = np.where(sims[local] >= .15, sims[local]**2, 0)*self.source_weights[selected]
            same = self.channels[selected] == str(row.get('channel_id'))
            outside = self.is_external[selected]
            def prior(w):
                return np.dot(w, self.targets[selected])/(1+w.sum())
            values.append([raw[pos],prior(weights),prior(weights*same),prior(weights*outside),
                           weights.sum(),(weights*outside).sum(),(weights*same).sum()])
        return np.asarray(values,np.float32)

    def training_matrix(self, frame):
        return self.oof[[self.positions[str(vid)] for vid in frame.video_id]]


class V5Features:
    def __init__(self, base, groups=('public_priors','emotions')):
        self.base = base
        self.groups = tuple(groups)

    @property
    def history(self):
        return self.base.history

    def history_context(self, row):
        return self.base.history_context(row)

    def fit(self, fitting, targets, external):
        self.emotion_rules = load_emotion_rules()
        if 'public_priors' in self.groups:
            self.expanded = ExpandedTitlePriors().fit(fitting,targets,external)
        return self

    def emotion_scores(self, title):
        return emotion_scores(title,self.emotion_rules)

    def _combine(self, frame, training):
        blocks = [self.base.training_matrix(frame) if training else self.base.transform(frame)]
        if 'public_priors' in self.groups:
            blocks.append(self.expanded.training_matrix(frame) if training else self.expanded.transform(frame))
        if 'emotions' in self.groups:
            blocks.append(emotion_matrix(frame.title,self.emotion_rules))
        return np.nan_to_num(np.hstack(blocks)).astype(np.float32)

    def training_matrix(self, frame):
        return self._combine(frame,True)

    def transform(self, frame):
        return self._combine(frame,False)


class LinearRankCorrection:
    """Frozen v4 + tiny regularized RankNet correction over new scalar features."""
    def __init__(self, base, base_width, l2=.1):
        self.base = base
        self.base_width = base_width
        self.l2 = l2

    def fit(self, X, y, channels):
        from scipy.optimize import minimize
        from scipy.special import expit
        from sklearn.preprocessing import StandardScaler
        from src.ranking import sample_pairs
        self.scaler = StandardScaler().fit(X[:,self.base_width:])
        extra = self.scaler.transform(X[:,self.base_width:]).astype(float)
        baseline = self.base.predict(X[:,:self.base_width])
        i,j = sample_pairs(y,channels,seed=142)
        differences = extra[i]-extra[j]
        signs = np.sign(y[i]-y[j])
        offsets = baseline[i]-baseline[j]
        def objective(weights):
            margin = signs*(offsets+differences@weights)
            loss = np.logaddexp(0,-margin).mean()+.5*self.l2*np.dot(weights,weights)
            gradient = -(differences.T@(signs*expit(-margin)))/len(i)+self.l2*weights
            return loss,gradient
        self.initial_loss = float(objective(np.zeros(extra.shape[1]))[0])
        result = minimize(objective,np.zeros(extra.shape[1]),jac=True,method='L-BFGS-B',
                          options={'maxiter':150,'ftol':1e-10,'gtol':1e-7})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError('Pairwise linear correction failed to converge')
        self.weights = result.x
        self.final_loss = float(result.fun)
        self.optimizer_iterations = int(result.nit)
        return self

    def predict(self, X):
        extra = self.scaler.transform(X[:,self.base_width:])
        return self.base.predict(X[:,:self.base_width])+extra@self.weights
