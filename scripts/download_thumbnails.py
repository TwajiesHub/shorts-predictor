#!/usr/bin/env python3
"""Download public YouTube thumbnails from i.ytimg.com.

Prefers the vertical Shorts preview (oar2.jpg) when it exists, then other
public i.ytimg.com sizes. Does not log in and never fetches description or
Amazon/affiliate URLs.

Images are intentionally not shipped in the git repo. After cloning:

    python scripts/download_thumbnails.py
    python scripts/extract_thumb_features.py   # only needed if you retrain v2

Optional: also fetch TestRun's hqdefault files (the letterboxed custom hook)
into data/testrun_thumbs/ for the oar2-vs-custom comparison:

    python scripts/download_thumbnails.py --testrun-hooks
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TESTRUN_CHANNEL_ID, THUMB_DIR, VIDEOS_PATH  # noqa: E402
from src.thumb_features import is_allowed_thumb_url, is_amazon_url  # noqa: E402

# Prefer a real Shorts-oriented vertical preview over letterboxed hqdefault.
CANDIDATE_NAMES = (
    "oar2.jpg",
    "oardefault.jpg",
    "maxresdefault.jpg",
    "hq720.jpg",
    "sddefault.jpg",
    "hqdefault.jpg",
)

UA = "shorts-predictor/0.2 (+https://github.com/TwajiesHub/shorts-predictor; public metadata research)"
SLEEP_S = 0.4


def fetch_url(url: str, timeout: float = 30) -> bytes | None:
    if is_amazon_url(url):
        raise RuntimeError(f"Refusing to fetch an Amazon URL: {url}")
    if not is_allowed_thumb_url(url):
        raise RuntimeError(f"Thumbnail URL is not allowed: {url}")
    req = Request(url, headers={"User-Agent": UA, "Accept": "image/*"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except HTTPError as exc:
        if exc.code in {404, 403}:
            return None
        raise
    except URLError:
        return None
    if len(data) < 800 or data[:1] == b"<":
        return None
    return data


def best_public_thumb(video_id: str) -> tuple[bytes, str] | None:
    for name in CANDIDATE_NAMES:
        url = f"https://i.ytimg.com/vi/{video_id}/{name}"
        data = fetch_url(url)
        if data:
            return data, name
    return None


def save_jpg(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sleep",
        type=float,
        default=SLEEP_S,
        help=f"Seconds to wait between requests (default {SLEEP_S})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of videos (0 = all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when data/thumbnails/<id>.jpg already exists",
    )
    parser.add_argument(
        "--testrun-hooks",
        action="store_true",
        help="Also save TestRun oar2 + hqdefault copies under data/testrun_thumbs/",
    )
    args = parser.parse_args()

    if not VIDEOS_PATH.exists():
        sys.exit(f"Missing {VIDEOS_PATH}. Collect or copy data/videos.parquet first.")

    df = pd.read_parquet(VIDEOS_PATH, columns=["video_id", "channel_id"])
    ids = df["video_id"].astype(str).tolist()
    if args.limit:
        ids = ids[: args.limit]
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    ok = skipped = fail = 0
    for i, vid in enumerate(ids, 1):
        dest = THUMB_DIR / f"{vid}.jpg"
        if dest.exists() and dest.stat().st_size > 800 and not args.force:
            skipped += 1
        else:
            got = best_public_thumb(vid)
            time.sleep(max(0.0, args.sleep))
            if not got:
                print(f"  fail {vid}")
                fail += 1
            else:
                data, name = got
                save_jpg(dest, data)
                ok += 1
                if ok <= 3 or ok % 250 == 0:
                    print(f"  saved {ok}  last={vid} ({name})")
        if i % 500 == 0:
            print(f"  progress {i}/{len(ids)}  saved={ok} skipped={skipped} fail={fail}")

    print(f"Thumbnails: saved={ok} skipped={skipped} fail={fail} dir={THUMB_DIR}")

    if args.testrun_hooks:
        hook_root = VIDEOS_PATH.parent / "testrun_thumbs"
        testrun_ids = (
            df.loc[df["channel_id"].astype(str) == TESTRUN_CHANNEL_ID, "video_id"]
            .astype(str)
            .tolist()
        )
        hook_ok = hook_fail = 0
        for vid in testrun_ids:
            # Copy the already-chosen public preview as oar2/best.
            src = THUMB_DIR / f"{vid}.jpg"
            if src.exists():
                save_jpg(hook_root / "oar2" / f"{vid}.jpg", src.read_bytes())
            hq = fetch_url(f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg")
            time.sleep(max(0.0, args.sleep))
            if hq:
                save_jpg(hook_root / "hqdefault" / f"{vid}.jpg", hq)
                hook_ok += 1
            else:
                hook_fail += 1
        print(
            f"TestRun hooks: hqdefault saved={hook_ok} fail={hook_fail} "
            f"dir={hook_root}"
        )


if __name__ == "__main__":
    main()
