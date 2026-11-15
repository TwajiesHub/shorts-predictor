"""Transparent title-cue scores; these are heuristics, not human psychometrics."""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

RESOURCE = Path(__file__).resolve().parents[1]/'data/lexicons/emotional_triggers_v5.json'


@lru_cache(maxsize=1)
def load_emotion_rules():
    return json.loads(RESOURCE.read_text())


@lru_cache(maxsize=512)
def _pattern(pattern):
    return re.compile(pattern, re.I)


def emotion_scores(title, rules=None):
    rules = load_emotion_rules() if rules is None else rules
    title = str(title).replace('’', "'")
    cap = float(rules['score_cap'])
    return {name: min(cap, sum(float(rule['weight'])*min(len(_pattern(rule['pattern']).findall(title)),
                     rules['per_pattern_match_cap']) for rule in patterns))/cap
            for name, patterns in rules['rules'].items()}


def emotion_matrix(titles, rules=None):
    rules = load_emotion_rules() if rules is None else rules
    scores = np.array([list(emotion_scores(t, rules).values()) for t in titles], dtype=np.float32)
    return np.column_stack([scores, scores.mean(axis=1), scores.max(axis=1), (scores > 0).mean(axis=1)])
