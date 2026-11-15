"""Pre-posting features for the Shorts performance model."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config import NICHE_KEYWORDS, MIN_AGE_DAYS, SHORT_MAX_SECONDS

LAGOS = ZoneInfo("Africa/Lagos")
UTC = timezone.utc

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)
HASHTAG_RE = re.compile(r"#\w+")
NUMBER_RE = re.compile(r"\d")
QUESTION_START = re.compile(
    r"^\s*(why|how|what|is|are|do|does|can|should|who|when|where)\b",
    re.I,
)

# Topic buckets that are useful as explicit flags (not just bag-of-words).
TOPIC_FLAGS = {
    "topic_iphone": ("iphone", "ios"),
    "topic_android": ("android", "samsung", "pixel", "galaxy"),
    "topic_python": ("python", "cpython"),
    "topic_javascript": ("javascript", "java script", "js "),
    "topic_windows": ("windows", "excel", "folder con"),
    "topic_excel": ("excel",),
    "topic_privacy": ("incognito", "privacy", "private", "tracking", "scam"),
    "topic_storage": ("gb", "tb", "storage", "drive", "ssd"),
    "topic_network": ("wifi", "wi-fi", "mbps", "internet", "bluetooth"),
    "topic_charging": ("charg", "battery", "mah", "usb", "watt", "60w"),
    "topic_review": ("worth it", "review", "vs ", "versus"),
    "topic_hidden": ("hidden", "secret", "didn't know", "did not know", "trick", "hack"),
    "topic_why": ("why ", "why?"),
}


def parse_published(row: pd.Series) -> datetime | None:
    ts = row.get("published_ts")
    if pd.notna(ts):
        try:
            return datetime.fromtimestamp(float(ts), UTC)
        except (ValueError, OSError, TypeError):
            pass
    iso = row.get("published_iso")
    if isinstance(iso, str) and iso:
        try:
            return datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except ValueError:
            return None
    upload_date = row.get("upload_date")
    if isinstance(upload_date, str) and re.fullmatch(r"\d{8}", upload_date):
        try:
            return datetime(
                int(upload_date[:4]),
                int(upload_date[4:6]),
                int(upload_date[6:8]),
                tzinfo=UTC,
            )
        except ValueError:
            return None
    return None


def title_manual_features(title: str) -> dict:
    t = title or ""
    tl = t.lower()
    words = re.findall(r"\w+", t, flags=re.UNICODE)
    emojis = EMOJI_RE.findall(t)
    return {
        "title_len_chars": len(t),
        "title_len_words": len(words),
        "has_question": int("?" in t or bool(QUESTION_START.search(t))),
        "has_exclamation": int("!" in t),
        "has_number": int(bool(NUMBER_RE.search(t))),
        "emoji_count": len(emojis),
        "hashtag_count": len(HASHTAG_RE.findall(t)),
        "starts_with_why": int(tl.lstrip().startswith("why")),
        "starts_with_how": int(tl.lstrip().startswith("how")),
        "starts_with_your": int(tl.lstrip().startswith("your") or tl.lstrip().startswith("you ")),
        "has_shorts_tag": int("#short" in tl),
        **{name: int(any(k in tl for k in keys)) for name, keys in TOPIC_FLAGS.items()},
    }


def numeric_feature_names() -> list[str]:
    sample = title_manual_features("Why does this work? #Shorts")
    return (
        list(sample.keys())
        + [
            "duration",
            "log_duration",
            "log_subscribers",
            "upload_hour",
            "hour_missing",
            "date_is_coarse",
            "dow_sin",
            "dow_cos",
            "is_weekend",
        ]
    )


NUMERIC_FEATURES = numeric_feature_names()


def build_numeric_frame(
    df: pd.DataFrame,
    default_duration: float = 30.0,
    default_subscribers: float | None = None,
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        feats = title_manual_features(str(row.get("title") or ""))
        duration = row.get("duration")
        if pd.isna(duration) or duration is None:
            duration = default_duration
        duration = float(duration)
        feats["duration"] = duration
        feats["log_duration"] = math.log1p(max(duration, 0.0))

        subs = row.get("channel_subscribers")
        if pd.isna(subs) or subs is None:
            subs = default_subscribers if default_subscribers is not None else 0.0
        feats["log_subscribers"] = math.log1p(max(float(subs), 0.0))

        published = parse_published(row)
        hour_missing = 1
        date_is_coarse = 1
        hour = 12.0
        dow = 0
        if published is not None:
            local = published.astimezone(LAGOS)
            dow = local.weekday()
            approx = row.get("hour_is_approximate")
            utc_mid = (
                published.astimezone(UTC).hour == 0 and published.minute == 0
            )
            # yt-dlp approximate_date often snaps "N months/years ago" to today's
            # calendar day, so weekday and hour are not real for those rows.
            if approx is False and not utc_mid:
                hour_missing = 0
                date_is_coarse = 0
                hour = float(local.hour) + local.minute / 60.0
            elif approx is False and not (utc_mid and local.day in (1, 15) and local.month == 11):
                # Exact-enough date (e.g. RSS or a real midnight publish) — keep weekday, drop hour.
                hour_missing = 1
                date_is_coarse = 0
                hour = 12.0
            else:
                hour_missing = 1
                date_is_coarse = 1
                hour = 12.0
                dow = 0
        feats["upload_hour"] = hour
        feats["hour_missing"] = hour_missing
        feats["date_is_coarse"] = date_is_coarse
        feats["dow_sin"] = 0.0 if date_is_coarse else math.sin(2 * math.pi * dow / 7)
        feats["dow_cos"] = 0.0 if date_is_coarse else math.cos(2 * math.pi * dow / 7)
        feats["is_weekend"] = 0 if date_is_coarse else int(dow >= 5)
        rows.append(feats)
    out = pd.DataFrame(rows, index=df.index)
    return out[NUMERIC_FEATURES]


def eligible_training_mask(df: pd.DataFrame) -> pd.Series:
    views = pd.to_numeric(df["views"], errors="coerce")
    duration = pd.to_numeric(df.get("duration"), errors="coerce")
    age = pd.to_numeric(df.get("age_days"), errors="coerce")
    titles = df["title"].fillna("").astype(str)
    is_short = df.get("is_short")
    if is_short is None:
        short_ok = duration <= SHORT_MAX_SECONDS
    else:
        short_ok = is_short.fillna(False).astype(bool) | (duration <= SHORT_MAX_SECONDS)
    # YouTube Shorts may be up to 3 minutes; TestRun itself is ~25-30s.
    duration_ok = (duration >= 5) & (
        (duration <= SHORT_MAX_SECONDS + 5) | (short_ok & (duration <= 180))
    )
    return (
        short_ok
        & titles.str.len().gt(0)
        & views.notna()
        & (views >= 0)
        & age.notna()
        & (age >= MIN_AGE_DAYS)
        & duration.notna()
        & duration_ok
    )


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Channel-normalized, age-adjusted log-view residual.

    target = log1p(views) - log1p(channel_median_views) - age_effect(log1p(age))

    The age effect is fit later on the training split only (see train.py) so
    this function only computes the channel-normalized residual and helpers.
    """
    out = df.copy()
    views = pd.to_numeric(out["views"], errors="coerce").clip(lower=0)
    out["log_views"] = np.log1p(views)
    # Channel median is computed among eligible rows only, by the caller, if
    # channel_median_views is not already present.
    if "channel_median_views" not in out.columns:
        out["channel_median_views"] = np.nan
    med = pd.to_numeric(out["channel_median_views"], errors="coerce")
    out["log_channel_median"] = np.log1p(med.clip(lower=0))
    out["rel_log_views"] = out["log_views"] - out["log_channel_median"]
    return out


def fill_channel_medians(df: pd.DataFrame, eligible: pd.Series) -> pd.DataFrame:
    out = df.copy()
    medians = (
        out.loc[eligible]
        .groupby("channel_id")["views"]
        .median()
        .rename("channel_median_views")
    )
    # Fallback for channels with no eligible videos: use all of that channel's shorts.
    all_medians = out.groupby("channel_id")["views"].median().rename("all_med")
    out = out.drop(columns=["channel_median_views"], errors="ignore")
    out = out.merge(medians, on="channel_id", how="left")
    out = out.merge(all_medians, on="channel_id", how="left")
    out["channel_median_views"] = out["channel_median_views"].fillna(out["all_med"])
    out = out.drop(columns=["all_med"])
    return out
