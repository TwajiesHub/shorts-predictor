#!/usr/bin/env python3
"""Discover comparable Shorts channels and collect public video metadata.

Uses only public yt-dlp / RSS / playlist data. Does not log in, and never
fetches URLs found in descriptions (Amazon affiliate links included).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    CAPS_BY_ROLE,
    CHANNELS_PATH,
    CHANNEL_CANDIDATES_PATH,
    DATA_DIR,
    DEFAULT_CAP,
    RAW_DIR,
    RESOLVE_NAMES,
    SEARCH_QUERIES,
    SEED_CHANNELS,
    SHORTS_ELIGIBLE_MAX_SECONDS,
    SHORT_MAX_SECONDS,
    TESTRUN_CHANNEL_ID,
    TESTRUN_HANDLE,
    VIDEOS_CSV_PATH,
    VIDEOS_PATH,
    NICHE_KEYWORDS,
)
from src.yt_utils import YtDlpError, dump_json, iter_json_lines, polite_sleep, unique  # noqa: E402

ATOM = "{http://www.w3.org/2005/Atom}"
MEDIA = "{http://search.yahoo.com/mrss/}"
YT = "{http://www.youtube.com/xml/schemas/2015}"

HANDLE_RE = re.compile(r"^@[\w.-]+$", re.I)
TAG_RE = re.compile(r"#(\w+)")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def uploads_playlist_url(channel_id: str) -> str:
    # Standard uploads playlist: UU + channel_id without the leading "UC".
    if channel_id.startswith("UC") and len(channel_id) == 24:
        return f"https://www.youtube.com/playlist?list=UU{channel_id[2:]}"
    return f"https://www.youtube.com/channel/{channel_id}/videos"


def rss_url(channel_id: str) -> str:
    return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"


def fetch_rss(channel_id: str) -> list[dict]:
    """Last ~15 public videos, including exact published timestamps.

    Description text is stored only so we can measure length and count hashtags.
    URLs inside the description are never requested.
    """
    req = Request(
        rss_url(channel_id),
        headers={"User-Agent": "Mozilla/5.0 (compatible; post-predictor/0.1; +research)"},
    )
    try:
        with urlopen(req, timeout=20) as resp:
            raw = resp.read()
    except (URLError, HTTPError, TimeoutError) as exc:
        print(f"  RSS failed for {channel_id}: {exc}")
        return []

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        print(f"  RSS parse failed for {channel_id}: {exc}")
        return []

    rows = []
    for entry in root.findall(f"{ATOM}entry"):
        vid = (entry.findtext(f"{YT}videoId") or "").strip()
        if not vid:
            continue
        published = entry.findtext(f"{ATOM}published")
        desc = entry.findtext(f"{MEDIA}group/{MEDIA}description") or ""
        stats = entry.find(f"{MEDIA}group/{MEDIA}community/{MEDIA}statistics")
        rating = entry.find(f"{MEDIA}group/{MEDIA}community/{MEDIA}starRating")
        views = None
        if stats is not None and stats.get("views") is not None:
            try:
                views = int(stats.get("views"))
            except ValueError:
                views = None
        rating_count = None
        if rating is not None and rating.get("count") is not None:
            try:
                rating_count = int(rating.get("count"))
            except ValueError:
                rating_count = None
        ts = None
        if published:
            try:
                ts = datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp()
            except ValueError:
                ts = None
        thumb = entry.find(f"{MEDIA}group/{MEDIA}thumbnail")
        rows.append(
            {
                "video_id": vid,
                "published": published,
                "published_ts": ts,
                "rss_views": views,
                "rss_rating_count": rating_count,
                "description": desc,
                "description_length": len(desc),
                "thumbnail_url": thumb.get("url") if thumb is not None else None,
            }
        )
    return rows


def search_videos(query: str, n: int = 40) -> list[dict]:
    args = [
        "--flat-playlist",
        "--dump-json",
        "--playlist-end",
        str(n),
        f"ytsearch{n}:{query}",
    ]
    try:
        rows = iter_json_lines(args, timeout=120)
    except YtDlpError as exc:
        print(f"  search failed '{query}': {exc}")
        return []
    return rows


def resolve_handle_or_name(query: str) -> dict | None:
    """Resolve a handle or channel name to id, handle, and subscriber count."""
    if HANDLE_RE.match(query.strip()):
        url = f"https://www.youtube.com/{query.strip()}"
    else:
        url = f"ytsearch5:{query}"
        try:
            hits = iter_json_lines(
                ["--flat-playlist", "--dump-json", "--playlist-end", "5", url],
                timeout=90,
            )
        except YtDlpError as exc:
            print(f"  resolve search failed '{query}': {exc}")
            return None
        if not hits:
            return None
        # Prefer a hit whose channel/uploader name contains the query.
        q = query.lower().strip()
        q_compact = q.replace(" ", "")
        tokens = [t for t in q.split() if len(t) >= 3]

        def name_matches(h):
            name = (h.get("channel") or h.get("uploader") or "").lower()
            handle = (h.get("uploader_id") or "").lower().lstrip("@")
            compact_name = name.replace(" ", "")
            if q_compact and (q_compact in compact_name or q_compact in handle):
                return True
            if tokens and all(t in name or t in handle for t in tokens):
                return True
            return False

        hit = next((h for h in hits if name_matches(h)), None)
        if hit is None:
            print(f"  no name match for '{query}' in search results")
            return None
        uploader = hit.get("uploader_id")
        channel_id = hit.get("channel_id")
        if uploader:
            url = f"https://www.youtube.com/{uploader}"
        elif channel_id:
            url = f"https://www.youtube.com/channel/{channel_id}"
        else:
            return None
        polite_sleep()

    # One full (no-download) extract of a recent upload gives subs + IDs.
    try:
        info = dump_json(
            [
                "--skip-download",
                "--dump-json",
                "--playlist-end",
                "1",
                "--extractor-args",
                "youtube:player_client=android",
                f"{url}/shorts" if url.startswith("http") else url,
            ],
            timeout=90,
        )
    except YtDlpError:
        try:
            info = dump_json(
                [
                    "--skip-download",
                    "--dump-json",
                    "--playlist-end",
                    "1",
                    "--extractor-args",
                    "youtube:player_client=android",
                    url,
                ],
                timeout=90,
            )
        except YtDlpError as exc:
            print(f"  resolve extract failed '{query}': {exc}")
            return None

    handle = info.get("uploader_id") or query
    if handle and not str(handle).startswith("@") and HANDLE_RE.match("@" + str(handle)):
        handle = "@" + str(handle)
    return {
        "handle": handle,
        "channel": info.get("channel") or info.get("uploader"),
        "channel_id": info.get("channel_id"),
        "channel_url": info.get("channel_url") or (f"https://www.youtube.com/{handle}" if handle else None),
        "subscribers": info.get("channel_follower_count"),
    }


def collect_uploads(channel_id: str, cap: int) -> list[dict]:
    url = uploads_playlist_url(channel_id)
    # Over-fetch a bit so that after filtering to Shorts we still hit the cap
    # on mixed long-form + Shorts channels.
    fetch_n = max(cap * 3, cap)
    fetch_n = min(fetch_n, 1200)
    args = [
        "--flat-playlist",
        "--dump-json",
        "--playlist-end",
        str(fetch_n),
        "--extractor-args",
        "youtubetab:approximate_date",
        url,
    ]
    try:
        rows = iter_json_lines(args, timeout=240)
    except YtDlpError as exc:
        print(f"  uploads failed {channel_id}: {exc}")
        return []
    return rows


def niche_score(titles: list[str]) -> float:
    if not titles:
        return 0.0
    hits = 0
    for title in titles:
        t = (title or "").lower()
        if any(k in t for k in NICHE_KEYWORDS):
            hits += 1
    return hits / len(titles)


def is_short_row(duration: float | None, title: str | None) -> bool:
    if duration is not None and duration <= SHORT_MAX_SECONDS:
        return True
    if title and "#short" in title.lower():
        return True
    return False


def discover() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, dict] = {}

    def add_candidate(handle: str | None, channel_id: str | None, **extra):
        key = (channel_id or "") or (handle or "")
        if not key:
            return
        rec = candidates.get(key, {"search_hits": 0, "short_hits": 0, "sources": []})
        # Never downgrade a seed role if the same channel also shows up in search.
        new_role = extra.get("role")
        old_role = rec.get("role")
        protected = {"target", "comparable", "ceiling"}
        rec.update({k: v for k, v in extra.items() if v is not None and k != "role"})
        if new_role:
            if old_role in protected and new_role not in protected:
                rec["role"] = old_role
            elif old_role not in protected or new_role in protected:
                rec["role"] = new_role
        if handle:
            rec["handle"] = handle
        if channel_id:
            rec["channel_id"] = channel_id
        # If this channel was previously keyed only by handle, merge into the id key.
        if channel_id and handle and handle in candidates and handle != key:
            prev = candidates.pop(handle)
            for k, v in prev.items():
                if k in ("search_hits", "short_hits") and isinstance(v, (int, float)):
                    rec[k] = rec.get(k, 0) + v
                    continue
                if rec.get(k) in (None, "", [], 0) and v not in (None, ""):
                    rec[k] = v
            if prev.get("role") in protected:
                rec["role"] = prev["role"]
        rec["search_hits"] = rec.get("search_hits", 0) + int(extra.get("hit", 0))
        rec["short_hits"] = rec.get("short_hits", 0) + int(extra.get("short_hit", 0))
        src = extra.get("source")
        if src:
            rec.setdefault("sources", [])
            if src not in rec["sources"]:
                rec["sources"].append(src)
        candidates[key] = rec

    print("== Resolving seed channels ==")
    for seed in SEED_CHANNELS:
        print(f"  seed {seed['handle']}")
        info = resolve_handle_or_name(seed["handle"])
        polite_sleep()
        if not info or not info.get("channel_id"):
            print(f"  WARNING: could not resolve {seed['handle']}")
            add_candidate(
                seed["handle"],
                TESTRUN_CHANNEL_ID if seed["handle"].lower() == TESTRUN_HANDLE.lower() else None,
                role=seed["role"],
                note=seed["note"],
                source="seed",
                resolved=False,
            )
            continue
        add_candidate(
            info["handle"],
            info["channel_id"],
            channel=info.get("channel"),
            subscribers=info.get("subscribers"),
            channel_url=info.get("channel_url"),
            role=seed["role"],
            note=seed["note"],
            source="seed",
            resolved=True,
            hit=0,
        )
        print(f"    -> {info['handle']} {info['channel_id']} subs={info.get('subscribers')}")

    print("== Resolving named channels from research ==")
    for name in RESOLVE_NAMES:
        print(f"  name {name}")
        info = resolve_handle_or_name(name)
        polite_sleep(1.0)
        if not info or not info.get("channel_id"):
            print(f"  WARNING: could not resolve {name}")
            continue
        add_candidate(
            info["handle"],
            info["channel_id"],
            channel=info.get("channel"),
            subscribers=info.get("subscribers"),
            channel_url=info.get("channel_url"),
            role="discovered",
            note=f"Resolved from research name: {name}",
            source="research_name",
            resolved=True,
        )
        print(f"    -> {info['handle']} {info['channel']} subs={info.get('subscribers')}")

    print("== Search discovery ==")
    raw_search_path = RAW_DIR / "search_hits.jsonl"
    with raw_search_path.open("w", encoding="utf-8") as fh:
        for query in SEARCH_QUERIES:
            print(f"  query: {query}")
            hits = search_videos(query, n=35)
            polite_sleep(1.0)
            for h in hits:
                fh.write(json.dumps(h, ensure_ascii=False) + "\n")
                handle = h.get("uploader_id")
                channel_id = h.get("channel_id")
                dur = h.get("duration")
                short_hit = 1 if (dur is not None and dur <= SHORTS_ELIGIBLE_MAX_SECONDS) else 0
                add_candidate(
                    handle,
                    channel_id,
                    channel=h.get("channel") or h.get("uploader"),
                    role="discovered",
                    source="search",
                    resolved=False,
                    hit=1,
                    short_hit=short_hit,
                )
            print(f"    {len(hits)} hits")

    rows = []
    for rec in candidates.values():
        rec = dict(rec)
        rec["sources"] = "|".join(rec.get("sources") or [])
        rows.append(rec)
    df = pd.DataFrame(rows)
    CHANNEL_CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CHANNEL_CANDIDATES_PATH, index=False)
    print(f"Saved {len(df)} candidates to {CHANNEL_CANDIDATES_PATH}")
    return df


def peek_channel(channel_id: str, handle: str | None) -> dict:
    """Look at a sample of uploads to score how comparable the channel is."""
    rows = collect_uploads(channel_id, cap=80)
    durs = [r.get("duration") for r in rows if r.get("duration") is not None]
    titles = [r.get("title") or "" for r in rows]
    n = len(rows)
    n_short = sum(1 for r in rows if is_short_row(r.get("duration"), r.get("title")))
    return {
        "peek_videos": n,
        "peek_shorts": n_short,
        "peek_short_share": (n_short / n) if n else 0.0,
        "peek_median_duration": float(pd.Series(durs).median()) if durs else None,
        "peek_niche_score": niche_score(titles),
        "peek_sample_titles": " || ".join(titles[:3]),
    }


def select_channels(candidates: pd.DataFrame) -> pd.DataFrame:
    """Keep seeds plus discovered channels that actually post niche Shorts."""
    selected = []
    # Always keep resolved seeds.
    seeds = candidates[candidates["role"].isin(["target", "comparable", "ceiling"])].copy()
    discovered = candidates[candidates["role"] == "discovered"].copy()

    # Resolve remaining discovered channels that have Shorts search hits, by
    # search volume, until we have ~25-35 total channels.
    if "short_hits" not in discovered.columns:
        discovered["short_hits"] = 0
    if "search_hits" not in discovered.columns:
        discovered["search_hits"] = 0
    discovered["short_hits"] = discovered["short_hits"].fillna(0)
    discovered["search_hits"] = discovered["search_hits"].fillna(0)

    # Prefer channels that already showed up as Shorts in search.
    discovered = discovered.sort_values(
        ["short_hits", "search_hits"], ascending=False
    )

    # Drop obvious official / off-niche names before peeking.
    skip_substrings = [
        "vevo",
        "music",
        "official audio",
        "applesupport",
        "google",
        "microsoft",
        "t-series",
        "mrbeast",
        "brightside",
        "bright side",
        "hbo",
        "netflix",
        "disney",
        "sony pictures",
        "warner",
        "paramount",
        "nbc",
        "cnn",
        "fox news",
    ]

    print("== Peeking / selecting channels ==")
    kept_ids = set()

    def keep_row(row, role, note=None, cap=None, peek=None):
        cid = row.get("channel_id")
        handle = row.get("handle")
        key = cid or handle
        if not key or key in kept_ids:
            return
        if not cid:
            print(f"  skip (no channel_id): {handle}")
            return
        kept_ids.add(key)
        rec = {
            "handle": handle,
            "channel": row.get("channel"),
            "channel_id": cid,
            "channel_url": row.get("channel_url") or (f"https://www.youtube.com/{handle}" if handle else None),
            "subscribers": row.get("subscribers"),
            "role": role,
            "note": note or row.get("note"),
            "sources": row.get("sources"),
            "search_hits": row.get("search_hits"),
            "short_hits": row.get("short_hits"),
            "cap": cap if cap is not None else CAPS_BY_ROLE.get(role, DEFAULT_CAP),
        }
        if peek:
            rec.update(peek)
        selected.append(rec)

    for _, row in seeds.iterrows():
        cid = row.get("channel_id")
        handle = row.get("handle")
        if not cid:
            # Try one more resolve.
            info = resolve_handle_or_name(str(handle))
            polite_sleep()
            if info:
                row = {**row.to_dict(), **info}
                cid = row.get("channel_id")
        if not cid:
            print(f"  SEED unresolved, skipping: {handle}")
            continue
        print(f"  peek seed {handle} {cid}")
        peek = peek_channel(cid, handle)
        polite_sleep(0.8)
        print(
            f"    shorts={peek['peek_shorts']}/{peek['peek_videos']} "
            f"niche={peek['peek_niche_score']:.2f} "
            f"med_dur={peek['peek_median_duration']}"
        )
        # If we still lack subs, try a one-video extract.
        if pd.isna(row.get("subscribers")) or not row.get("subscribers"):
            info = resolve_handle_or_name(handle or cid)
            polite_sleep()
            if info and info.get("subscribers"):
                row = {**row.to_dict(), **{k: v for k, v in info.items() if v is not None}}
        keep_row(row, row.get("role") or "comparable", row.get("note"), peek=peek)

    # Target ~20-30 discovered additions, plus seeds (~7).
    target_n = 32
    for _, row in discovered.iterrows():
        if len(selected) >= target_n:
            break
        handle = str(row.get("handle") or "")
        channel = str(row.get("channel") or "")
        blob = (handle + " " + channel).lower()
        if any(s in blob for s in skip_substrings):
            print(f"  skip off-niche: {handle or channel}")
            continue
        cid = row.get("channel_id")
        if not cid or not handle:
            # Need a full resolve for subs + confirmed id.
            q = handle or channel
            if not q or q == "nan":
                continue
            print(f"  resolve discovered {q}")
            info = resolve_handle_or_name(str(q))
            polite_sleep(1.0)
            if not info or not info.get("channel_id"):
                continue
            row = {**row.to_dict(), **info}
            cid = row.get("channel_id")
            handle = row.get("handle")
        # Avoid re-keeping seeds.
        if cid in kept_ids or (handle and handle in kept_ids):
            continue
        print(f"  peek discovered {handle} {cid}")
        peek = peek_channel(cid, handle)
        polite_sleep(0.8)
        print(
            f"    shorts={peek['peek_shorts']}/{peek['peek_videos']} "
            f"niche={peek['peek_niche_score']:.2f} "
            f"med_dur={peek['peek_median_duration']}"
        )
        # Require a meaningful Shorts presence and some niche overlap.
        if peek["peek_shorts"] < 5 and peek["peek_short_share"] < 0.15:
            print("    reject: too few Shorts")
            continue
        if peek["peek_niche_score"] < 0.15:
            print("    reject: low niche overlap")
            continue
        # Fill subs if missing.
        if not row.get("subscribers"):
            info = resolve_handle_or_name(handle or cid)
            polite_sleep()
            if info:
                row = {**row.to_dict(), **{k: v for k, v in info.items() if v is not None}}
        keep_row(row, "discovered", row.get("note") or "Search discovery", peek=peek)

    out = pd.DataFrame(selected)
    CHANNELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CHANNELS_PATH, index=False)
    print(f"Selected {len(out)} channels -> {CHANNELS_PATH}")
    return out


def normalize_video_row(raw: dict, channel_meta: dict, rss_map: dict[str, dict]) -> dict:
    vid = raw.get("id") or raw.get("video_id")
    title = raw.get("title") or ""
    duration = raw.get("duration")
    views = raw.get("view_count")
    ts = raw.get("timestamp")
    upload_date = raw.get("upload_date")
    rss = rss_map.get(vid) or {}

    # Prefer exact RSS published time when we have it.
    published_ts = rss.get("published_ts") or ts
    published_iso = None
    hour_is_approximate = True
    if rss.get("published"):
        published_iso = rss["published"]
        hour_is_approximate = False
        if views is None:
            views = rss.get("rss_views")
    elif published_ts:
        published_iso = datetime.fromtimestamp(published_ts, timezone.utc).isoformat()
        # Playlist approximate_date often falls back to midnight UTC.
        dt = datetime.fromtimestamp(published_ts, timezone.utc)
        hour_is_approximate = dt.hour == 0 and dt.minute == 0
    elif upload_date and re.fullmatch(r"\d{8}", str(upload_date)):
        published_iso = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}T00:00:00+00:00"
        hour_is_approximate = True

    desc_len = rss.get("description_length")
    tags = None
    if rss.get("description"):
        tags = ",".join(TAG_RE.findall(rss["description"]))

    is_short = is_short_row(duration, title)
    if duration is not None and duration <= SHORTS_ELIGIBLE_MAX_SECONDS and (
        (title and "#short" in title.lower()) or is_short
    ):
        is_short = True

    thumb = None
    thumbs = raw.get("thumbnails") or []
    if isinstance(thumbs, list) and thumbs:
        thumb = thumbs[-1].get("url") if isinstance(thumbs[-1], dict) else None
    thumb = rss.get("thumbnail_url") or thumb or (
        f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg" if vid else None
    )

    return {
        "video_id": vid,
        "title": title,
        "channel": channel_meta.get("channel"),
        "handle": channel_meta.get("handle"),
        "channel_id": channel_meta.get("channel_id"),
        "channel_subscribers": channel_meta.get("subscribers"),
        "channel_role": channel_meta.get("role"),
        "upload_date": upload_date,
        "published_iso": published_iso,
        "published_ts": published_ts,
        "hour_is_approximate": hour_is_approximate,
        "duration": duration,
        "is_short": bool(is_short),
        "views": views,
        "likes": rss.get("rss_rating_count"),  # only a proxy; full like_count filled later if enriched
        "comment_count": None,
        "tags": tags,
        "description_length": desc_len,
        "thumbnail_url": thumb,
        "webpage_url": f"https://www.youtube.com/watch?v={vid}" if vid else raw.get("webpage_url"),
        "collected_at": now_utc().isoformat(),
    }


def collect_all_videos(channels: pd.DataFrame) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    per_channel_counts = []

    for i, row in channels.iterrows():
        meta = row.to_dict()
        cid = meta.get("channel_id")
        handle = meta.get("handle")
        cap = int(meta.get("cap") or DEFAULT_CAP)
        print(f"== Collecting {handle} ({cid}) cap={cap} ==")
        polite_sleep(0.8)
        uploads = collect_uploads(cid, cap=cap)
        print(f"  raw uploads fetched: {len(uploads)}")
        polite_sleep(0.5)
        rss_rows = fetch_rss(cid)
        rss_map = {r["video_id"]: r for r in rss_rows}
        print(f"  RSS entries: {len(rss_rows)}")

        # Filter to short-form first, then cap.
        normalized = [normalize_video_row(r, meta, rss_map) for r in uploads]
        shorts = [r for r in normalized if r["is_short"] or (
            r["duration"] is not None and r["duration"] <= SHORT_MAX_SECONDS
        )]
        # If a "comparable" channel has almost no Shorts, keep nothing long-form;
        # the model is a Shorts predictor.
        if len(shorts) >= 3:
            chosen = shorts[:cap]
        else:
            # Keep the few Shorts we have; do not flood with long-form.
            chosen = shorts[:cap]
            print(f"  WARNING: only {len(shorts)} Shorts")

        out_path = RAW_DIR / f"{(handle or cid).lstrip('@')}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for r in chosen:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_rows.extend(chosen)
        per_channel_counts.append((handle, cid, len(chosen), meta.get("subscribers")))
        print(f"  kept {len(chosen)} Shorts")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("ERROR: no videos collected")
        return df

    # Deduplicate by video_id (some channels can appear twice if discovery overlapped).
    df = df.drop_duplicates(subset=["video_id"], keep="first")

    # Compute video age now.
    now_ts = now_utc().timestamp()
    df["published_ts"] = pd.to_numeric(df["published_ts"], errors="coerce")
    df["age_days"] = (now_ts - df["published_ts"]) / 86400.0
    df["views"] = pd.to_numeric(df["views"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["channel_subscribers"] = pd.to_numeric(df["channel_subscribers"], errors="coerce")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(VIDEOS_PATH, index=False)
    df.to_csv(VIDEOS_CSV_PATH, index=False)
    print(f"Saved {len(df)} videos from {df['channel_id'].nunique()} channels")
    print(f"  parquet: {VIDEOS_PATH}")
    print(f"  csv:     {VIDEOS_CSV_PATH}")
    print("Per-channel Shorts kept:")
    for handle, cid, n, subs in per_channel_counts:
        print(f"  {handle}  n={n}  subs={subs}")
    return df


def enrich_subset(df: pd.DataFrame, per_channel: int = 25, all_testrun: bool = True, save: bool = True) -> pd.DataFrame:
    """Fill likes, comment_count, tags, description_length for a subset.

    Descriptions are taken from yt-dlp metadata only; no description URLs are fetched.
    """
    if df.empty:
        return df
    picks = []
    for cid, grp in df.groupby("channel_id"):
        role = grp["channel_role"].iloc[0] if "channel_role" in grp else None
        n = len(grp) if (all_testrun and role == "target") else min(per_channel, len(grp))
        picks.append(grp.head(n))
    sample = pd.concat(picks, ignore_index=True)
    print(f"== Enriching {len(sample)} videos with full public metadata ==")

    updates = {}
    for i, row in sample.iterrows():
        vid = row["video_id"]
        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            info = dump_json(
                [
                    "--skip-download",
                    "--dump-json",
                    "--extractor-args",
                    "youtube:player_client=android",
                    url,
                ],
                timeout=60,
                retries=1,
            )
        except YtDlpError as exc:
            print(f"  enrich fail {vid}: {exc}")
            polite_sleep(0.4)
            continue
        desc = info.get("description") or ""
        tags = info.get("tags") or []
        if isinstance(tags, list):
            tag_str = ",".join(str(t) for t in tags)
        else:
            tag_str = str(tags)
        updates[vid] = {
            "likes": info.get("like_count"),
            "comment_count": info.get("comment_count"),
            "description_length": len(desc) if desc is not None else None,
            "tags": tag_str or row.get("tags"),
            "duration": info.get("duration") if info.get("duration") is not None else row.get("duration"),
            "views": info.get("view_count") if info.get("view_count") is not None else row.get("views"),
            "channel_subscribers": info.get("channel_follower_count")
            if info.get("channel_follower_count") is not None
            else row.get("channel_subscribers"),
            "upload_date": info.get("upload_date") or row.get("upload_date"),
            "title": info.get("title") or row.get("title"),
        }
        if (i + 1) % 25 == 0:
            print(f"  enriched {i+1}/{len(sample)}")
        polite_sleep(0.45)

    if not updates:
        print("  no enrichments applied")
        return df

    df = df.copy()
    for vid, upd in updates.items():
        mask = df["video_id"] == vid
        for k, v in upd.items():
            if v is None:
                continue
            df.loc[mask, k] = v
    if save:
        df.to_parquet(VIDEOS_PATH, index=False)
        df.to_csv(VIDEOS_CSV_PATH, index=False)
    print(f"  applied enrichments to {len(updates)} videos")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public YouTube Shorts metadata")
    parser.add_argument("--skip-discover", action="store_true")
    parser.add_argument("--skip-enrich", action="store_true")
    parser.add_argument("--channels-only", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_discover and CHANNELS_PATH.exists():
        channels = pd.read_csv(CHANNELS_PATH)
        print(f"Loaded {len(channels)} selected channels from {CHANNELS_PATH}")
    else:
        if args.skip_discover and CHANNEL_CANDIDATES_PATH.exists():
            candidates = pd.read_csv(CHANNEL_CANDIDATES_PATH)
            print(f"Loaded {len(candidates)} candidates")
        else:
            candidates = discover()
        channels = select_channels(candidates)

    if args.channels_only:
        return

    videos = collect_all_videos(channels)
    if videos.empty:
        sys.exit(1)
    if not args.skip_enrich:
        enrich_subset(videos, per_channel=20, all_testrun=True)
    print("Done.")


if __name__ == "__main__":
    main()
