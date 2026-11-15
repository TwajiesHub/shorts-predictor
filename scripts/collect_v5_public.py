#!/usr/bin/env python3
"""Collect only public title/performance metadata; preserve the original dataset.

Default: use configured YouTube API authentication, otherwise a pinned public
Shorts snapshot. No descriptions, tags, affiliate URLs, or video bodies are saved.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, parse_qs
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train import latin_letter_share
from src.config import TESTRUN_CHANNEL_ID
from src.research_features import normalized_title

REVISION = '60d8f38921e43ec1eac227e2b89e2d0c12ec4d15'
SOURCE = f'https://raw.githubusercontent.com/JamilAL01/YouTube-Shorts-Dataset/{REVISION}/shorts_metadata.json'
SOURCE_SHA = 'f767db96ddb148a6905a55543fb28db8fe53531cd0637691290e55fe0ead7f15'
CACHE = ROOT.parent/'.cache/shorts-v5/source_metadata.json'
TECH = re.compile(r'\b(?:iphone|android|samsung|pixel|smartphone|phone|ipad|ios|apple|computer|laptop|windows|linux|python|javascript|coding|programming|code|excel|wifi|bluetooth|usb|battery|charger|storage|ssd|keyboard|privacy|password|tech|gadget|science|physics|space|robot|ai)\b', re.I)


def video_id_from_url(url):
    p = urlparse(str(url))
    host = (p.hostname or '').lower()
    if host not in ('youtube.com', 'www.youtube.com', 'm.youtube.com', 'youtu.be'):
        return None
    if host == 'youtu.be':
        value = p.path.strip('/')
    elif p.path.startswith('/shorts/'):
        value = p.path.split('/')[2]
    else:
        value = parse_qs(p.query).get('v', [''])[0]
    return value if re.fullmatch(r'[A-Za-z0-9_-]{11}', value) else None


def iso_duration(value):
    m = re.fullmatch(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?', value or '')
    if not m:
        return None
    h, minute, sec = [float(v or 0) for v in m.groups()]
    return 3600*h+60*minute+sec


def fetch_api(channel_ids, key, max_pages=1):
    """Bounded uploads discovery: no search.list calls; 50 IDs per statistics call.

    API keys stay in memory; exceptions/logs never include request URLs or bodies.
    <=60s uploads are short-form candidates, not proof of Shorts-feed membership.
    """
    used = 0
    requests = Counter()
    now = datetime.now(timezone.utc)
    def request(method, params):
        nonlocal used
        params['key'] = key
        url = 'https://www.googleapis.com/youtube/v3/'+method+'?'+urlencode(params)
        try:
            with urlopen(Request(url, headers={'User-Agent':'shorts-predictor-research/0.5'}), timeout=25) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f'YouTube {method}: HTTP {exc.code}; check credentials/quota/network') from None
        except (URLError, TimeoutError):
            raise RuntimeError(f'YouTube {method}: transport unavailable; network policy may block access') from None
        used += 1  # These list endpoints cost one unit per successful request.
        requests[method] += 1
        time.sleep(.4)
        return result
    uploads = []
    for start in range(0, len(channel_ids), 50):
        data = request('channels', {'part':'contentDetails', 'id':','.join(channel_ids[start:start+50]),
                                   'fields':'items(contentDetails/relatedPlaylists/uploads)'})
        uploads.extend(item['contentDetails']['relatedPlaylists']['uploads'] for item in data.get('items', []))
    ids = []
    for playlist in uploads:
        token = None
        for _ in range(max_pages):
            params = {'part':'contentDetails','playlistId':playlist,'maxResults':50,
                      'fields':'nextPageToken,items(contentDetails/videoId)'}
            if token:params['pageToken'] = token
            data = request('playlistItems', params)
            ids.extend(item['contentDetails']['videoId'] for item in data.get('items', []))
            token = data.get('nextPageToken')
            if not token:break
    rows = []
    ids = list(dict.fromkeys(ids))
    for start in range(0, len(ids), 50):
        data = request('videos', {'part':'snippet,contentDetails,statistics','id':','.join(ids[start:start+50]),
            'fields':'items(id,snippet(title,channelId,publishedAt,categoryId),contentDetails/duration,statistics/viewCount)'})
        for item in data.get('items', []):
            snippet = item['snippet']
            duration = iso_duration(item.get('contentDetails', {}).get('duration'))
            published = pd.to_datetime(snippet['publishedAt'], utc=True)
            age = (now-published.to_pydatetime()).total_seconds()/86400
            if duration is None or not 5 <= duration <= 60 or not 3 <= age <= 180:
                continue
            rows.append(dict(video_id=item['id'], title=snippet['title'], channel_id=snippet['channelId'],
                published_at=snippet['publishedAt'], duration=duration,
                views=item.get('statistics', {}).get('viewCount'), category_name='',
                source='youtube_data_api', observed_at=now.isoformat()))
    return rows, dict(source='YouTube Data API v3', successful_requests=dict(requests),
                      successful_request_quota_units=used, search_requests=0,
                      note='Failed calls may also consume quota; successful-call count is not account quota telemetry. <=60s uploads are candidates, not verified vertical Shorts.')


def fetch_snapshot():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists() or hashlib.sha256(CACHE.read_bytes()).hexdigest() != SOURCE_SHA:
        digest = hashlib.sha256()
        temp = CACHE.with_suffix('.download')
        size = 0
        with urlopen(SOURCE, timeout=60) as response, temp.open('wb') as out:
            while True:
                chunk = response.read(1024*1024)
                if not chunk:break
                size += len(chunk)
                if size > 100*1024*1024:
                    raise ValueError('Source exceeds bounded 100 MiB download')
                digest.update(chunk);out.write(chunk)
        if digest.hexdigest() != SOURCE_SHA:
            temp.unlink(missing_ok=True)
            raise ValueError('Public snapshot integrity check failed')
        temp.replace(CACHE)
    raw = json.loads(CACHE.read_text())
    rows = []
    for item in raw:
        rows.append(dict(video_id=video_id_from_url(item.get('short_url')),
            title=item.get('title'), channel_id=item.get('channel_id'),
            published_at=item.get('published_at'), duration=item.get('duration'),
            views=item.get('view_count'), category_name=item.get('category_name',''),
            source='jamil_shorts_snapshot', observed_at=None))
    return rows, dict(source=SOURCE, revision=REVISION, raw_sha256=SOURCE_SHA,
        raw_records=len(raw), raw_bytes=CACHE.stat().st_size,
        license='CC BY-NC 4.0', attribution='JamilAL01 / YouTube-Shorts-Dataset',
        observed_at=None, note='Original view-count observation times are not provided; retrieval time is not a fresh view-count observation.')


def prepare_rows(rows, original, max_rows=5000):
    counts = Counter(input_rows=len(rows))
    if not rows:
        raise ValueError('No public metadata rows retrieved')
    df = pd.DataFrame(rows)
    for col in ('duration','views'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['published_at'] = pd.to_datetime(df.published_at, utc=True, errors='coerce')
    valid = (df.video_id.notna() & df.channel_id.notna() & df.title.fillna('').str.len().gt(0)
             & df.duration.between(5,180) & df.views.ge(0) & df.published_at.notna())
    counts['invalid_or_outside_short_duration'] = int((~valid).sum());df = df.loc[valid].copy()
    counts['duplicate_video_ids'] = int(df.video_id.duplicated().sum());df = df.drop_duplicates('video_id')
    existing_ids = set(original.video_id.astype(str))
    existing_titles = set(original.title.map(normalized_title))
    overlap = df.video_id.isin(existing_ids) | df.title.map(normalized_title).isin(existing_titles) | df.channel_id.eq(TESTRUN_CHANNEL_ID)
    counts['excluded_original_ids_titles_or_testrun'] = int(overlap.sum());df = df.loc[~overlap].copy()
    cutoff = df.published_at.max()-pd.Timedelta(days=731)
    recent = df.published_at.ge(cutoff)
    counts['older_than_two_years_within_source'] = int((~recent).sum());df = df.loc[recent].copy()
    latin = df.title.map(latin_letter_share).ge(.9)
    counts['non_latin_script'] = int((~latin).sum());df = df.loc[latin].copy()
    # Baseline uses comparable publication quarters within a channel. This does
    # not invent ages or apply 2026 age corrections to unknown-time 2025 counts.
    df['publication_cohort'] = df.published_at.dt.strftime('%Y')+'Q'+df.published_at.dt.quarter.astype(str)
    group = df.groupby(['channel_id','publication_cohort']).views
    df['cohort_size'] = group.transform('size')
    df['cohort_median_views'] = group.transform('median')
    df['prior_target'] = np.clip(np.log1p(df.views)-np.log1p(df.cohort_median_views), -3, 3)
    supported = df.cohort_size.ge(5)
    counts['insufficient_channel_quarter_support'] = int((~supported).sum());df = df.loc[supported].copy()
    matching = df.channel_id.isin(set(original.channel_id)) | df.title.str.contains(TECH) | df.category_name.eq('Science & Technology')
    counts['outside_matching_niches'] = int((~matching).sum());df = df.loc[matching].copy()
    df = df.sort_values(['published_at','video_id'], ascending=[False,True])
    capped = df.groupby('channel_id', sort=False).head(150)
    counts['per_channel_cap'] = len(df)-len(capped)
    df = capped.head(max_rows).reset_index(drop=True)
    counts['global_cap'] = len(capped)-len(df)
    counts['retained'] = len(df)
    df['published_at'] = df.published_at.map(lambda t:t.isoformat())
    return df, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['auto','snapshot','api'], default='auto')
    parser.add_argument('--max-rows', type=int, default=5000)
    parser.add_argument('--max-pages', type=int, default=1)
    args = parser.parse_args()
    if args.max_rows < 1 or not 1 <= args.max_pages <= 5:
        parser.error('--max-rows must be positive and --max-pages must be between 1 and 5')
    original = pd.read_parquet(ROOT/'data/videos.parquet')
    key = next((os.environ[n] for n in ['YOUTUBE_API_KEY','YT_API_KEY','GOOGLE_API_KEY'] if os.environ.get(n)), None)
    fallback = None
    if args.source == 'api' or (args.source == 'auto' and key):
        if not key:raise SystemExit('No API key configured; set it securely or use --source snapshot')
        try:
            channels = sorted(set(original.channel_id)-{TESTRUN_CHANNEL_ID})
            rows, source = fetch_api(channels, key, args.max_pages)
            if not rows:
                raise RuntimeError('API returned no eligible recent short-form candidates')
        except RuntimeError as exc:
            if args.source == 'api':raise SystemExit(str(exc)) from None
            fallback = str(exc);rows,source = fetch_snapshot()
    else:
        fallback = 'No YouTube API key configured; this run used the public snapshot. API/RSS probes were proxy-blocked.'
        rows,source = fetch_snapshot()
    data, counts = prepare_rows(rows, original, args.max_rows)
    if data.empty:raise SystemExit('No eligible public examples after filtering')
    target = ROOT/'data/public_title_examples_v5.parquet'
    data.to_parquet(target,index=False)
    report = dict(retrieved_at=datetime.now(timezone.utc).isoformat(), source=source,
        fallback_reason=fallback, api_credential_present=bool(key),
        authenticated_api_requests_made=source.get('successful_requests', {}), counts=counts, channels=int(data.channel_id.nunique()),
        publication_range=[data.published_at.min(),data.published_at.max()],
        view_count_range=[int(data.views.min()),int(data.views.max())],
        median_views=float(data.views.median()), matched_original_channels=int(data.channel_id.isin(original.channel_id).sum()),
        short_duration_range=[float(data.duration.min()),float(data.duration.max())],
        normalized_target='clip(log1p(views)-log1p(channel/publication-quarter median views), -3, 3); cohort >=5',
        output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        fields=list(data.columns),
        limitations=['Snapshot is older, not a live/recent-2026 collection',
                     'Latin-script filtering is not verified language identification',
                     'Public source is observational/popularity-selected, not randomized title tests'])
    (ROOT/'reports/public_collection_v5.json').write_text(json.dumps(report,indent=2)+'\n')
    if source.get('license') == 'CC BY-NC 4.0':
        license_url=f'https://raw.githubusercontent.com/JamilAL01/YouTube-Shorts-Dataset/{REVISION}/LICENSE'
        with urlopen(license_url, timeout=25) as response:
            (ROOT/'data/lexicons/LICENSE.public_shorts_v5.txt').write_bytes(response.read())
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
