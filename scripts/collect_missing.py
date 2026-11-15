#!/usr/bin/env python3
"""Collect channels the first pass missed, and backfill subscriber counts.

Public metadata only; description URLs are never fetched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    CHANNELS_PATH,
    DATA_DIR,
    RAW_DIR,
    VIDEOS_CSV_PATH,
    VIDEOS_PATH,
)
from src.yt_utils import YtDlpError, iter_json_lines, polite_sleep  # noqa: E402

spec = importlib.util.spec_from_file_location("collect_data", ROOT / "scripts" / "collect_data.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)


# Research-brief channels dropped when search overwrote their seed role.
MISSING = [
    {
        "handle": "@zackdfilms",
        "channel": "Zack D. Films",
        "channel_id": "UCvz84_Q0BbvZThy75mbd-Dg",
        "subscribers": 29_700_000,
        "role": "comparable",
        "note": "Faceless curiosity explainers (recovered seed)",
        "cap": 350,
    },
    {
        "handle": "@athome-y1o",
        "channel": "at home",
        "channel_id": "UCOpdzXajGvWqle8m_6Kw-YA",
        "subscribers": 36_700,
        "role": "comparable",
        "note": "Faceless iPhone screen-recording tips (recovered seed)",
        "cap": 350,
    },
    {
        "handle": "@SpeedMcQueen",
        "channel": "Speed McQueen",
        "channel_id": "UCNFT_eq_QCApMEwCACHto9Q",
        "subscribers": 17_400_000,
        "role": "discovered",
        "note": "Resolved from research name: Speed McQueen",
        "cap": 300,
    },
    {
        "handle": "@yusuftechtalks",
        "channel": "Yusuf Tech Talks",
        "channel_id": "UCY9RET1ez_fV1fPDFSx3jQA",
        "subscribers": 13_500,
        "role": "discovered",
        "note": "Resolved from research name: Yusuf Tech Talks",
        "cap": 300,
    },
    {
        "handle": "@Keebo100",
        "channel": "Keebo100",
        "channel_id": "UCWlUNiP9DneURw6Nimyko3A",
        "subscribers": 18_700,
        "role": "discovered",
        "note": "Resolved from research name: Keebo100",
        "cap": 300,
    },
]


def collect_shorts_tab(handle: str, cap: int) -> list[dict]:
    url = f"https://www.youtube.com/{handle}/shorts"
    try:
        rows = iter_json_lines(
            [
                "--flat-playlist",
                "--dump-json",
                "--playlist-end",
                str(cap),
                url,
            ],
            timeout=240,
        )
    except YtDlpError as exc:
        print(f"  shorts tab failed {handle}: {exc}")
        return []
    return rows


def collect_one(meta: dict) -> list[dict]:
    cid = meta["channel_id"]
    handle = meta["handle"]
    cap = int(meta.get("cap") or 300)
    print(f"== Collecting {handle} ({cid}) cap={cap} ==")
    polite_sleep(0.8)
    uploads = cd.collect_uploads(cid, cap=cap)
    print(f"  raw uploads fetched: {len(uploads)}")
    polite_sleep(0.5)
    rss_rows = cd.fetch_rss(cid)
    rss_map = {r["video_id"]: r for r in rss_rows}
    print(f"  RSS entries: {len(rss_rows)}")

    uploads_by_id = {r.get("id") or r.get("video_id"): r for r in uploads}
    normalized = [cd.normalize_video_row(r, meta, rss_map) for r in uploads]
    shorts = [
        r
        for r in normalized
        if r["is_short"]
        or (r["duration"] is not None and r["duration"] <= 60)
    ]

    # If the uploads playlist omitted duration (common on some high-volume
    # Shorts channels), fall back to the /shorts tab and mark those as Shorts.
    if len(shorts) < 5:
        print("  few Shorts from uploads playlist; trying /shorts tab")
        polite_sleep(0.8)
        tab = collect_shorts_tab(handle, cap=cap)
        print(f"  shorts tab rows: {len(tab)}")
        extra = []
        for r in tab:
            vid = r.get("id")
            if not vid:
                continue
            base = uploads_by_id.get(vid, r)
            # Prefer upload-playlist dates when present; force short flag via title.
            if not (base.get("title") or "").lower().find("#short") >= 0:
                title = base.get("title") or r.get("title") or ""
                base = {**base, "title": title if "#short" in title.lower() else (title + " #Shorts")}
            if r.get("view_count") is not None and base.get("view_count") is None:
                base = {**base, "view_count": r.get("view_count")}
            extra.append(cd.normalize_video_row(base, meta, rss_map))
        # Treat every shorts-tab video as a Short even if duration is unknown.
        for row in extra:
            row["is_short"] = True
        shorts = extra[:cap]

    chosen = shorts[:cap]
    out_path = RAW_DIR / f"{handle.lstrip('@')}.jsonl"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        import json

        for r in chosen:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  kept {len(chosen)} Shorts")
    return chosen


def fill_subscribers(channels: pd.DataFrame) -> pd.DataFrame:
    out = channels.copy()
    for i, row in out.iterrows():
        if pd.notna(row.get("subscribers")) and float(row.get("subscribers") or 0) > 0:
            continue
        handle = row.get("handle")
        cid = row.get("channel_id")
        q = handle or cid
        print(f"  resolve subs for {q}")
        info = cd.resolve_handle_or_name(str(q))
        polite_sleep(0.8)
        if not info:
            continue
        for k in ("subscribers", "channel", "handle", "channel_url"):
            if info.get(k) is not None and (pd.isna(row.get(k)) or not row.get(k)):
                out.at[i, k] = info[k]
        if info.get("subscribers") is not None:
            print(f"    -> {info.get('handle')} subs={info.get('subscribers')}")
    return out


def merge_videos(new_rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(new_rows)
    from datetime import datetime, timezone

    now_ts = datetime.now(timezone.utc).timestamp()
    if "published_ts" in new.columns:
        new["published_ts"] = pd.to_numeric(new["published_ts"], errors="coerce")
        new["age_days"] = (now_ts - new["published_ts"]) / 86400.0
    for col in ("views", "duration", "channel_subscribers"):
        if col in new.columns:
            new[col] = pd.to_numeric(new[col], errors="coerce")

    if VIDEOS_PATH.exists():
        old = pd.read_parquet(VIDEOS_PATH)
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df = df.drop_duplicates(subset=["video_id"], keep="last")
    # Recompute age for the whole frame so old + new are consistent.
    if "published_ts" in df.columns:
        df["published_ts"] = pd.to_numeric(df["published_ts"], errors="coerce")
        df["age_days"] = (now_ts - df["published_ts"]) / 86400.0
    df.to_parquet(VIDEOS_PATH, index=False)
    df.to_csv(VIDEOS_CSV_PATH, index=False)
    print(f"Merged dataset: {len(df)} videos, {df['channel_id'].nunique()} channels")
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing = pd.read_csv(CHANNELS_PATH) if CHANNELS_PATH.exists() else pd.DataFrame()
    have = set(existing["channel_id"].dropna().astype(str)) if len(existing) else set()

    # Recover missing research/seed channels.
    to_add = []
    all_new_videos: list[dict] = []
    for meta in MISSING:
        if str(meta["channel_id"]) in have:
            print(f"already present: {meta['handle']}")
            continue
        rows = collect_one(meta)
        all_new_videos.extend(rows)
        peek = {
            "peek_videos": len(rows),
            "peek_shorts": len(rows),
            "peek_short_share": 1.0 if rows else 0.0,
            "peek_niche_score": None,
            "peek_sample_titles": " || ".join(r["title"] for r in rows[:3]),
            "peek_median_duration": pd.Series([r["duration"] for r in rows if r.get("duration") is not None]).median()
            if rows
            else None,
        }
        to_add.append({**meta, **peek})

    # Re-collect ceiling channels that stored 0 Shorts because duration was missing.
    if len(existing):
        need_retry = existing[
            existing["handle"].astype(str).str.lower().isin(
                {"@mavigadgets", "@mavigadgetoriginals"}
            )
        ]
        # Only retry if we currently have few videos for them.
        have_counts = {}
        if VIDEOS_PATH.exists():
            vids = pd.read_parquet(VIDEOS_PATH)
            have_counts = vids.groupby("channel_id").size().to_dict()
        for _, row in need_retry.iterrows():
            cid = row["channel_id"]
            n = have_counts.get(cid, 0)
            if n >= 20:
                print(f"ceiling channel already has {n} videos: {row['handle']}")
                continue
            meta = row.to_dict()
            meta["cap"] = int(meta.get("cap") or 200)
            rows = collect_one(meta)
            all_new_videos.extend(rows)

    if to_add:
        channels = pd.concat([existing, pd.DataFrame(to_add)], ignore_index=True)
        channels = channels.drop_duplicates(subset=["channel_id"], keep="last")
    else:
        channels = existing

    print("== Filling missing subscriber counts ==")
    channels = fill_subscribers(channels)
    channels.to_csv(CHANNELS_PATH, index=False)
    print(f"Channels file now has {len(channels)} rows -> {CHANNELS_PATH}")

    if all_new_videos:
        df = merge_videos(all_new_videos)
        # Propagate refreshed subscriber counts onto videos.
        sub_map = channels.set_index("channel_id")["subscribers"].to_dict()
        df["channel_subscribers"] = df["channel_id"].map(sub_map).fillna(df["channel_subscribers"])
        df.to_parquet(VIDEOS_PATH, index=False)
        df.to_csv(VIDEOS_CSV_PATH, index=False)
        print("Updated subscriber counts on videos")

        # Light enrich: TestRun already done in the main pass; enrich new channels a bit.
        print("== Enriching recovered channels (15 each) ==")
        new_ids = set(r["video_id"] for r in all_new_videos)
        subset = df[df["video_id"].isin(new_ids)].copy()
        enriched = cd.enrich_subset(subset, per_channel=15, all_testrun=False, save=False)
        keep_cols = [
            "likes",
            "comment_count",
            "description_length",
            "tags",
            "duration",
            "views",
            "channel_subscribers",
            "upload_date",
            "title",
        ]
        enriched = enriched.set_index("video_id")
        df = df.set_index("video_id")
        for col in keep_cols:
            if col in enriched.columns:
                df.loc[enriched.index, col] = enriched[col]
        df = df.reset_index()
        df.to_parquet(VIDEOS_PATH, index=False)
        df.to_csv(VIDEOS_CSV_PATH, index=False)
        print(f"Final: {len(df)} videos, {df['channel_id'].nunique()} channels")
    else:
        print("No new videos collected")


if __name__ == "__main__":
    main()
