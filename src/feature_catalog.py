"""Analysis-only manifest of the existing v4/v5 transformed feature order.

Production feature builders are unchanged. Dimension and perturbation tests guard
this manifest against drifting away from their implementations.
"""
from collections import OrderedDict

from src.features import NUMERIC_FEATURES, title_manual_features
from src.ranking import THUMB_COLS

DESCRIPTIONS = OrderedDict([
    ('title_structure','Title length, punctuation, numbers, emoji, topic flags, openings and linguistic/readability patterns'),
    ('title_ngrams','64-dimensional SVD of word/bigram and character 3–5-gram hashing'),
    ('sentiment_lexicon','VADER valence summaries, negation and hand-counted power words; not v5 emotional triggers'),
    ('channel_context_priors','Channel median/support, relative title length, and cross-fitted title-performance priors/support'),
    ('thumbnail_cheap_stats','Brightness, contrast, faces, colorfulness, edges, text/layout and derived cheap image traits'),
    ('posting_time','Known/missing/coarse date indicators, hour and cyclical hour/weekday features'),
    ('subscriber_size','Log subscribers, subscriber tier, tier×power-words and tier×title-length interactions'),
    ('duration','Duration, log duration, duration relative to historical channel median'),
    ('emotional_triggers_v5','Ten explicit v5 trigger scores and three summaries; inactive in the current head'),
    ('external_public_priors_v5','Additional public-source prior scalars; inactive in the current head'),
    ('thumbnail_embeddings','CLIP embeddings/PCA/prompt similarities used by v2, absent from v4/v5'),
    ('novelty','V4 channel/topic familiarity candidates; not selected in the current head'),
])

LANGUAGE_NAMES = [
    'positive_valence_density','negative_valence_density','positive_word_fraction','negative_word_fraction',
    'maximum_absolute_valence','opening_negative_fraction','negation_count',
    'information_gap_pronoun_fraction','contradiction_fraction','second_person_fraction',
    'first_person_fraction','opening_question_word_fraction','starts_with_list_number',
    'numeric_unit_count','comparison_format','quantified_promise_format','colon_count',
    'bracket_count','mean_word_length','long_word_fraction','unique_word_fraction',
    'opening_uppercase_fraction','trailing_character_fraction',
]


def feature_catalog(bundle):
    top=bundle['features']
    wrapper=None
    if type(top).__name__=='V5Features':
        wrapper=top;research=top.base
    else:
        research=top
    if type(research).__name__!='ResearchFeatures' or type(research.base).__name__!='RankFeatures':
        raise ValueError('Unsupported feature pipeline: update the analysis manifest before proceeding')
    base=research.base
    rows=[]
    def add(name,group,dependencies=None):
        rows.append(dict(index=len(rows),name=name,group=group,
                         dependencies=dependencies or [group]))
    manual=set(title_manual_features(''))
    for name in NUMERIC_FEATURES:
        group=('title_structure' if name in manual else 'duration' if name in ('duration','log_duration')
               else 'subscriber_size' if name=='log_subscribers' else 'posting_time')
        add(name,group)
    for name in ['question_mark_count','number_group_count','emoji_codepoint_count','power_word_count',
                 'title_length_under_35','title_length_35_to_64','title_length_65_plus','uppercase_fraction']:
        add(name,'sentiment_lexicon' if name=='power_word_count' else 'title_structure')
    add('hour_sin','posting_time');add('hour_cos','posting_time')
    if base.context:
        for name,group,deps in [
            ('log_channel_baseline','channel_context_priors',None),
            ('channel_history_count','channel_context_priors',None),
            ('duration_vs_channel_median','duration',['duration','channel_context_priors']),
            ('title_length_vs_channel_median','channel_context_priors',['title_structure','channel_context_priors']),
            ('subscriber_tier','subscriber_size',None),
            ('subscriber_tier_x_power_words','subscriber_size',['subscriber_size','sentiment_lexicon']),
            ('subscriber_tier_x_title_length','subscriber_size',['subscriber_size','title_structure']),
        ]:add(name,group,deps)
    if base.thumbnail:
        for name in THUMB_COLS+['thumb_exposure_adjusted_contrast','thumb_mean_face_area',
            'thumb_edge_color_interaction','thumb_usable_exposure','thumb_observed']:
            add(name,'thumbnail_cheap_stats')
    if base.text:
        for i in range(base.svd.components_.shape[0]):add(f'title_svd_{i:02d}','title_ngrams')
    if 'language' in research.groups:
        for i,name in enumerate(LANGUAGE_NAMES):
            add('language_'+name,'sentiment_lexicon' if i<=6 else 'title_structure')
    if 'peers' in research.groups:
        for name in ['title_prior_ridge','title_peer_residual','same_channel_peer_residual',
                     'title_peer_support','same_channel_peer_support']:
            add(name,'channel_context_priors',['title_structure','title_ngrams','channel_context_priors'])
    if 'novelty' in research.groups:
        for name in ['max_title_similarity','max_channel_title_similarity','mean_channel_top5_similarity',
                     'no_vocabulary_match','channel_present_in_bank']:
            add(name,'novelty')
    if wrapper is not None:
        if 'public_priors' in wrapper.groups:
            for name in ['expanded_ridge_prior','expanded_neighbor_prior','expanded_channel_prior',
                         'external_only_prior','expanded_support','external_support','expanded_channel_support']:
                add(name,'external_public_priors_v5',['title_ngrams','channel_context_priors'])
        if 'emotions' in wrapper.groups:
            for name in list(wrapper.emotion_rules['rules'])+['mean','maximum','nonzero_fraction']:
                add('trigger_'+name,'emotional_triggers_v5')
    groups=OrderedDict()
    for key,description in DESCRIPTIONS.items():
        selected=[r for r in rows if r['group']==key]
        groups[key]=dict(description=description,columns=[r['index'] for r in selected],
                         feature_names=[r['name'] for r in selected],active=bool(selected))
    all_columns=[col for group in groups.values() for col in group['columns']]
    if sorted(all_columns)!=list(range(len(rows))) or len(all_columns)!=len(set(all_columns)):
        raise ValueError('Feature groups must partition every transformed column exactly once')
    return rows,groups,research
