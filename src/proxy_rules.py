"""Optional explicit score adjustments. Never fits or modifies a prediction model."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.emotional_triggers import emotion_scores, load_emotion_rules

THRESHOLDS = dict(emotion_high=.7, emotion_low=.3, prior_high=70., prior_low=30.,
    prior_min_history=5, profile_min_history=20, recent_min_history=5, recent_limit=20,
    curiosity_high=.7, outrage_high=.7, outrage_style_fraction=.2,
    similarity_high=.8, thumbnail_brightness_low=.15, thumbnail_brightness_high=.85,
    thumbnail_contrast_low=.10, thumbnail_edges_low=.02, thumbnail_bad_checks=2,
    clip_low=-3., clip_high=3.)
WEIGHTS = dict(high_emotion_high_prior=.08, low_emotion_low_prior=-.06,
    curiosity_sweet_spot=.05, outrage_outside_usual_style=-.04,
    many_triggers_poor_thumbnail=-.03, no_triggers_similar_title=-.04)


def adjust_relative_score(score, emotions, *, channel_prior_percentile=None,
        in_sweet_spot=None, high_outrage_style=None, thumbnail_poor=None,
        very_similar_recent=None):
    """Apply the six independent rules, sum once, then clip. Unknown gates skip."""
    score = float(score)
    values = np.asarray(list(emotions.values()), float)
    if not np.isfinite(score) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Relative score must be finite and emotional scores must be in [0, 1]')
    if channel_prior_percentile is not None and (not np.isfinite(channel_prior_percentile) or
                                                not 0 <= channel_prior_percentile <= 100):
        raise ValueError('Channel prior percentile must be in [0, 100] or unknown')
    strength = float(values.max(initial=0))
    fired_count = int((values > 0).sum())
    fired = []
    if channel_prior_percentile is not None:
        if strength >= .7 and channel_prior_percentile >= 70:
            fired.append('high_emotion_high_prior')
        if strength <= .3 and channel_prior_percentile <= 30:
            fired.append('low_emotion_low_prior')
    if emotions.get('curiosity_gap', 0) >= .7 and in_sweet_spot is True:
        fired.append('curiosity_sweet_spot')
    if emotions.get('controversy_outrage', 0) >= .7 and high_outrage_style is False:
        fired.append('outrage_outside_usual_style')
    if fired_count >= 3 and thumbnail_poor is True:
        fired.append('many_triggers_poor_thumbnail')
    if fired_count == 0 and very_similar_recent is True:
        fired.append('no_triggers_similar_title')
    raw_delta = float(sum(WEIGHTS[name] for name in fired))
    unbounded = score+raw_delta
    adjusted = float(np.clip(unbounded, THRESHOLDS['clip_low'], THRESHOLDS['clip_high']))
    return dict(adjusted_score=adjusted, rule_delta=adjusted-score,
                unclipped_rule_delta=raw_delta, rules_fired=fired,
                rule_score_clipped=bool(adjusted != unbounded))


def midpoint_percentile(value, reference):
    """A tied constant prior is neutral (50), not an artificial 100th percentile."""
    values = np.sort(np.asarray(reference, float))
    values = values[np.isfinite(values)]
    if len(values) < THRESHOLDS['prior_min_history'] or not np.isfinite(value):
        return None
    left = np.searchsorted(values,value,side='left')
    right = np.searchsorted(values,value,side='right')
    return float(50*(left+right)/len(values))


def poor_thumbnail(row):
    keys = ['thumb_brightness','thumb_contrast','thumb_edge_density']
    values = [row.get(k, np.nan) for k in keys]
    if not all(pd.notna(v) and np.isfinite(float(v)) for v in values):
        return None  # Missing/imputed thumbnails are not observations of poor images.
    brightness,contrast,edges = map(float,values)
    checks = [brightness < .15 or brightness > .85, contrast < .10, edges < .02]
    return bool(sum(checks) >= 2)


def _peer_features(bundle):
    obj = bundle['features']
    for _ in range(8):
        if (hasattr(obj,'_lookup') and hasattr(obj,'oof') and 'peers' in getattr(obj,'groups',())):
            return obj
        obj = getattr(obj,'base',None)
        if obj is None:break
    raise ValueError('Proxy rules require an existing v4/v5 same-channel title prior')


class ProxyRuleContext:
    """Read-only contextual statistics from existing fitting history and priors.

    Extra history is accepted only for draft inference (e.g. TestRun's existing
    local history). The evaluation harness never supplies held-out extra history.
    """
    def __init__(self, bundle):
        if bundle.get('version') not in ('v4','v5'):
            raise ValueError('--apply-rules supports v4/v5 only')
        self.peer = _peer_features(bundle)
        self.age_a,self.age_b = bundle.get('age_a',0.),bundle.get('age_b',0.)
        self.emotion_rules = getattr(bundle['features'],'emotion_rules',None) or load_emotion_rules()
        self.outrage_rules = dict(self.emotion_rules,
            rules={'controversy_outrage':self.emotion_rules['rules']['controversy_outrage']})
        self.histories = {str(c):g.copy() for c,g in self.peer.history.groupby('channel_id',sort=False)}
        self.prior_references = {}
        for c in np.unique(self.peer.channels):
            self.prior_references[str(c)] = self.peer.oof[self.peer.channels == c,2]
        self.targets = dict(zip(self.peer.ids.astype(str),self.peer.targets))
        self.bank_positions = {str(vid):i for i,vid in enumerate(self.peer.ids)}
        self.profiles = {}

    def _profile(self, row, extra_history=None):
        channel = str(row.get('channel_id'))
        history = self.histories.get(channel)
        extra_key = None
        if history is None and extra_history is not None:
            history = extra_history.loc[extra_history.channel_id.astype(str) == channel].copy()
            extra_key = tuple(zip(history.video_id.astype(str),history.views.astype(float)))
        cutoff = row.get('published_ts')
        cutoff = float(cutoff) if pd.notna(cutoff) and np.isfinite(float(cutoff)) else None
        exclude = str(row.get('video_id',''))
        key = (channel,cutoff,exclude,extra_key)
        if key in self.profiles:return self.profiles[key]
        profile = dict(history_count=0, sweet_spot=None, high_outrage_style=None,
                       centroid=None, recent_count=0, recent_dates_approximate=None)
        if history is None or history.empty:
            self.profiles[key] = profile;return profile
        history = history.loc[history.video_id.astype(str) != exclude].copy()
        history['published_ts'] = pd.to_numeric(history.published_ts,errors='coerce')
        # Do not use a known future publish, or claim that snapped dates are exact.
        if cutoff is not None:history = history.loc[history.published_ts < cutoff]
        history = history.sort_values(['published_ts','video_id'],na_position='first')
        profile['history_count'] = len(history)
        recent = history.tail(20)
        profile['recent_count'] = len(recent)
        profile['recent_dates_approximate'] = bool(recent.hour_is_approximate.fillna(True).any()) if len(recent) else None
        if len(history) >= 20:
            outcomes = []
            for _,h in history.iterrows():
                target = self.targets.get(str(h.video_id))
                if target is None:
                    views = h.get('views',np.nan)
                    age = h.get('age_days',np.nan)
                    target = np.log1p(max(float(views),0)) if pd.notna(views) else np.nan
                    if pd.notna(age):target -= self.age_a+self.age_b*np.log1p(max(float(age),0))
                outcomes.append(target)
            outcomes = np.asarray(outcomes,float)
            known = np.isfinite(outcomes)
            if known.sum() >= 20 and np.ptp(outcomes[known]) > 0:
                top = known & (outcomes >= np.quantile(outcomes[known],.75))
                if top.sum() >= 5:
                    profile['sweet_spot'] = np.quantile(history.loc[top,'title'].str.len(),[.25,.75]).tolist()
            prevalence = np.mean([emotion_scores(t,self.outrage_rules)['controversy_outrage'] > 0 for t in recent.title])
            profile['high_outrage_style'] = bool(prevalence >= .2)
        if len(recent) >= 5:
            if all(str(v) in self.bank_positions for v in recent.video_id):
                matrix = self.peer.bank[[self.bank_positions[str(v)] for v in recent.video_id]]
            else:
                from src.research_features import normalized_title
                matrix = self.peer.vectorizer.transform(recent.title.map(normalized_title))
            centroid = np.asarray(matrix.sum(axis=0)).ravel()
            norm = np.sqrt(np.dot(centroid,centroid))
            if norm > 0:profile['centroid'] = centroid/norm
        self.profiles[key] = profile
        return profile

    def contexts(self, frame, extra_history=None):
        from src.research_features import normalized_title
        # Existing prior lookup; no model prediction, training, or image extraction.
        priors = self.peer._lookup(frame)[:,2]
        query = self.peer.vectorizer.transform(frame.title.map(normalized_title))
        results = []
        for pos,(_,row) in enumerate(frame.iterrows()):
            profile = self._profile(row,extra_history)
            channel = str(row.get('channel_id'))
            reference = self.prior_references.get(channel,[])
            sweet = profile['sweet_spot']
            similarity = (float(np.clip(query[pos].dot(profile['centroid'])[0],0,1))
                          if profile['centroid'] is not None else None)
            results.append(dict(channel_prior_percentile=midpoint_percentile(priors[pos],reference),
                title_length_sweet_spot=sweet,
                in_sweet_spot=bool(sweet[0] <= len(str(row.title)) <= sweet[1]) if sweet is not None else None,
                high_outrage_style=profile['high_outrage_style'],thumbnail_poor=poor_thumbnail(row),
                very_similar_recent=bool(similarity >= .8) if similarity is not None else None,
                recent_title_similarity=similarity,history_count=profile['history_count'],
                recent_count=profile['recent_count'],recent_dates_approximate=profile['recent_dates_approximate']))
        return results

    def apply(self, frame, predictions, score_column='score', extra_history=None):
        """Add fields without changing original scores, rows, order, or rank."""
        if predictions.empty:return predictions.copy()
        contexts = self.contexts(frame,extra_history)
        fields = ['video_id'] if 'video_id' in predictions and 'video_id' in frame else ['title','thumbnail']
        positions = defaultdict(deque)
        for pos,(_,row) in enumerate(frame.iterrows()):
            positions[tuple(str(row.get(k,'')) for k in fields)].append(pos)
        additions,details = [],[]
        for _,row in predictions.iterrows():
            pos = positions[tuple(str(row.get(k,'')) for k in fields)].popleft()
            context = contexts[pos]
            emotions = row.get('emotional_triggers')
            if not isinstance(emotions,dict):emotions = emotion_scores(str(row.title),self.emotion_rules)
            strength = max(emotions.values(),default=0.)
            detail = dict(context,emotional_strength=float(strength),
                          emotional_triggers_fired=int(sum(v > 0 for v in emotions.values())))
            additions.append(adjust_relative_score(row[score_column],emotions,
                **{k:context[k] for k in ['channel_prior_percentile','in_sweet_spot','high_outrage_style',
                                         'thumbnail_poor','very_similar_recent']}))
            details.append(detail)
        out = predictions.copy()
        for key in additions[0]:out[key] = [value[key] for value in additions]
        out['proxy_rule_context'] = details
        out['adjusted_rank'] = out.adjusted_score.rank(ascending=False,method='first').astype(int)
        return out
