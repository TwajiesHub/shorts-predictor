#!/usr/bin/env python3
"""CLI: score draft Shorts titles, and optionally one or more thumbnails."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    MODEL_PATH,
    MODEL_V1_PATH,
    MODEL_V2_PATH,
    TESTRUN_CHANNEL_ID,
    THUMB_EMBEDDINGS_PATH,
    THUMB_FEATURES_PATH,
    VIDEOS_PATH,
)
from src.features import build_numeric_frame  # noqa: E402
from src.thumb_features import (  # noqa: E402
    INTERPRETABLE_FEATURES,
    ClipEncoder,
    features_from_path,
    is_allowed_thumb_url,
    is_amazon_url,
    thumb_path_for_video,
)

LAGOS = ZoneInfo("Africa/Lagos")
UTC = ZoneInfo("UTC")

_ENCODER: ClipEncoder | None = None


def percentile_of(value: float, ref: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return float("nan")
    return float((ref <= value).mean() * 100.0)


def resolve_model_path(which: str | None) -> Path:
    if which in (None, "auto"):
        if MODEL_V2_PATH.exists():
            return MODEL_V2_PATH
        if MODEL_V1_PATH.exists():
            return MODEL_V1_PATH
        return MODEL_PATH
    if which == "v2":
        return MODEL_V2_PATH
    if which == "v1":
        return MODEL_V1_PATH if MODEL_V1_PATH.exists() else MODEL_PATH
    if which in ("v3", "v4", "v5"):
        return ROOT / f"models/model_{which}.joblib"
    path = Path(which)
    return path


def load_bundle(which: str | None = "auto"):
    path = resolve_model_path(which)
    if not path.exists():
        sys.exit(f"Model not found at {path}. Train v1 or v2 first.")
    bundle = joblib.load(path)
    bundle["_path"] = str(path)
    bundle.setdefault("version", "v2" if "v2" in path.name or bundle.get("version") == "v2" else "v1")
    return bundle


def parse_when(text: str | None) -> datetime:
    if not text:
        now = datetime.now(LAGOS)
        when = now.replace(hour=10, minute=0, second=0, microsecond=0)
        if when <= now:
            when = when + timedelta(days=1)
        return when
    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=LAGOS)
        except ValueError:
            continue
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LAGOS)
    return dt


def _get_encoder() -> ClipEncoder:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = ClipEncoder()
        _ENCODER.load()
    return _ENCODER


def fetch_thumbnail(spec: str, tmpdir: Path) -> Path:
    """Load a local image path, or download an allowed http(s) image URL."""
    spec = spec.strip()
    p = Path(spec).expanduser()
    if p.exists() and p.is_file():
        return p
    if spec.startswith("http://") or spec.startswith("https://"):
        if is_amazon_url(spec):
            sys.exit(f"Refusing to fetch an Amazon URL as a thumbnail: {spec}")
        if not is_allowed_thumb_url(spec):
            sys.exit(f"Thumbnail URL is not allowed: {spec}")
        req = Request(spec, headers={"User-Agent": "post-predictor/0.2", "Accept": "image/*"})
        with urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 800 or data[:1] == b"<":
            sys.exit(f"Thumbnail URL did not return an image: {spec}")
        suffix = Path(urlparse(spec).path).suffix or ".jpg"
        dest = Path(tmpdir)
        if dest.is_dir():
            dest = dest / f"thumb{suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest
    sys.exit(f"Thumbnail not found (not a local file or URL): {spec}")


def title_block(bundle, titles: list[str]) -> np.ndarray:
    text = hstack(
        [
            bundle["tfidf_word"].transform(pd.Series(titles, dtype=str)),
            bundle["tfidf_char"].transform(pd.Series(titles, dtype=str)),
        ]
    ).tocsr()
    return bundle["svd"].transform(text)


def thumb_block_from_paths(bundle, paths: list[Path | None]) -> np.ndarray:
    """Build [interpretable | clip-pca] rows. None path -> training medians / mean embedding."""
    cols = bundle.get("thumb_interpretable_features") or INTERPRETABLE_FEATURES
    med = bundle.get("median_thumb_features") or {}
    mean_emb = np.asarray(bundle.get("mean_clip_embedding"), dtype=np.float32)
    enc = None
    interp_rows = []
    emb_rows = []
    for p in paths:
        if p is None:
            interp_rows.append([float(med.get(c, 0.0)) for c in cols])
            if mean_emb.ndim == 1 and mean_emb.size:
                emb_rows.append(mean_emb)
            else:
                emb_rows.append(np.zeros(512, dtype=np.float32))
            continue
        if enc is None:
            enc = _get_encoder()
        got = features_from_path(p, encoder=enc)
        if got is None:
            interp_rows.append([float(med.get(c, 0.0)) for c in cols])
            emb_rows.append(mean_emb if mean_emb.size else np.zeros(512, dtype=np.float32))
            continue
        feats, emb = got
        interp_rows.append([float(feats.get(c, med.get(c, 0.0))) for c in cols])
        emb_rows.append(np.asarray(emb, dtype=np.float32))
    interp = np.asarray(interp_rows, dtype=float)
    embs = np.stack(emb_rows, axis=0)
    pca = bundle["thumb_pca"]
    return np.hstack([interp, pca.transform(embs)])


def stored_thumb_block(bundle, video_ids: list[str]) -> np.ndarray:
    """Use precomputed features/embeddings for existing dataset videos."""
    cols = bundle.get("thumb_interpretable_features") or INTERPRETABLE_FEATURES
    med = bundle.get("median_thumb_features") or {}
    mean_emb = np.asarray(bundle.get("mean_clip_embedding"), dtype=np.float32)
    feats = pd.read_parquet(THUMB_FEATURES_PATH).set_index("video_id")
    blob = np.load(THUMB_EMBEDDINGS_PATH, allow_pickle=True)
    id_to_row = {str(v): i for i, v in enumerate(blob["video_id"].astype(str))}
    # NPZ members decompress on every access; load each large member once.
    cached_embeddings = blob["embedding"]
    cached_has_embedding = blob["has_embedding"]
    interp_rows = []
    emb_rows = []
    for vid in video_ids:
        if vid in feats.index and bool(feats.loc[vid].get("has_thumbnail", False)):
            row = feats.loc[vid]
            interp_rows.append([float(row[c]) if c in row.index and pd.notna(row[c]) else float(med.get(c, 0.0)) for c in cols])
        else:
            interp_rows.append([float(med.get(c, 0.0)) for c in cols])
        j = id_to_row.get(str(vid))
        if j is not None and cached_has_embedding[j]:
            emb_rows.append(cached_embeddings[j].astype(np.float32))
        else:
            emb_rows.append(mean_emb if mean_emb.size else np.zeros(512, dtype=np.float32))
    interp = np.asarray(interp_rows, dtype=float)
    embs = np.stack(emb_rows, axis=0)
    return np.hstack([interp, bundle["thumb_pca"].transform(embs)])


def is_v2(bundle) -> bool:
    return bundle.get("version") == "v2" or "thumb_pca" in bundle


def _implied_views(bundle, scores: np.ndarray) -> np.ndarray:
    median = bundle.get("testrun_median_views")
    if median is None or not np.isfinite(median):
        median = 200.0
    age_a = bundle.get("age_a", 0.0)
    age_b = bundle.get("age_b", 0.0)
    implied = np.expm1(scores + np.log1p(median) + age_a + age_b * np.log1p(14.0))
    return np.clip(implied, 0, None)


def _rows_for_titles(titles, duration, when, subscribers, channel_id) -> pd.DataFrame:
    when_utc = when.astimezone(UTC)
    return pd.DataFrame(
        [
            {
                "title": title,
                "duration": duration,
                "channel_subscribers": subscribers,
                "channel_id": channel_id,
                "published_ts": when_utc.timestamp(),
                "published_iso": when_utc.isoformat(),
                "hour_is_approximate": False,
            }
            for title in titles
        ]
    )


def predict_scores(
    bundle,
    titles: list[str],
    duration: float,
    when: datetime,
    subscribers: float | None = None,
    channel_id: str | None = None,
    thumb_paths: list[Path | None] | None = None,
    used_imputed_thumb: bool | None = None,
    recent_median_views: float | None = None,
    apply_rules: bool = False,
    rules_context=None,
) -> pd.DataFrame:
    if apply_rules and bundle.get("version") not in ("v4", "v5"):
        raise ValueError("--apply-rules supports v4/v5 only")
    if subscribers is None:
        subscribers = bundle.get("testrun_subscribers") or 8.0
    channel_id = channel_id or bundle.get("testrun_channel_id") or TESTRUN_CHANNEL_ID
    if thumb_paths is not None and len(thumb_paths) not in (0, 1, len(titles)):
        # Broadcast is handled by the caller.
        raise ValueError("thumb_paths length must be 1 or match titles")
    if thumb_paths is not None and len(thumb_paths) == 1 and len(titles) > 1:
        thumb_paths = thumb_paths * len(titles)
    if thumb_paths is None:
        thumb_paths = [None] * len(titles)
        if used_imputed_thumb is None:
            used_imputed_thumb = is_v2(bundle)

    df = _rows_for_titles(titles, duration, when, subscribers, channel_id)
    if bundle.get("version") in ("v3", "v4", "v5"):
        from src.ranking import attach_thumbnails, draft_output
        extra_rule_history = None
        if recent_median_views is not None:
            df["recent_median_views"] = recent_median_views
        elif channel_id == TESTRUN_CHANNEL_ID:
            # Inference-only context, never fed into fitting or held-out evaluation.
            history = pd.read_parquet(VIDEOS_PATH)
            history = history[history.channel_id == channel_id].sort_values("published_ts").tail(20)
            extra_rule_history = history
            if len(history):
                df["recent_median_views"] = float(history.views.median())
                df["recent_duration"] = float(history.duration.median())
                df["recent_title_length"] = float(history.title.str.len().median())
                df["recent_count"] = len(history)
        df["thumbnail"] = ["imputed" if p is None else str(p) for p in thumb_paths]
        enriched = attach_thumbnails(df, paths=thumb_paths)
        out = draft_output(bundle, enriched)
        if bundle.get("version") == "v5":
            out["emotional_triggers"] = [bundle["features"].emotion_scores(t) for t in out.title]
            out["emotional_triggers_used_for_ranking"] = "emotions" in bundle.get("selected_groups", [])
        if apply_rules:
            from src.proxy_rules import ProxyRuleContext
            context = rules_context if rules_context is not None else ProxyRuleContext(bundle)
            out = context.apply(enriched,out,extra_history=extra_rule_history)
        return out
    X_num = build_numeric_frame(df, default_duration=duration, default_subscribers=subscribers)
    T = title_block(bundle, df["title"].astype(str).tolist())
    if is_v2(bundle):
        Th = thumb_block_from_paths(bundle, thumb_paths)
        X = np.hstack([X_num.to_numpy(dtype=float), T, Th])
    else:
        X = np.hstack([X_num.to_numpy(dtype=float), T])
    Xs = bundle["scaler"].transform(X)
    deployed = bundle.get("deployed", "mlp")
    model = bundle["mlp"] if deployed == "mlp" else bundle["ridge_full"]
    scores = model.predict(Xs)
    ref = np.asarray(bundle.get("reference_pred_scores"), dtype=float)
    out = df[["title"]].copy()
    out["score"] = scores
    out["niche_percentile"] = [percentile_of(s, ref) for s in scores]
    out["implied_views_at_14d"] = _implied_views(bundle, scores)
    out["channel_median_views"] = bundle.get("testrun_median_views")
    out["better_than_your_typical"] = scores > 0
    out["rank"] = out["score"].rank(ascending=False, method="first").astype(int)
    out["thumbnail"] = [("imputed" if p is None else str(p)) for p in thumb_paths]
    if used_imputed_thumb:
        out["thumbnail_imputed"] = [p is None for p in thumb_paths]
    return out.sort_values("score", ascending=False).reset_index(drop=True)


# Backwards-compatible name used by earlier report scripts.
def predict_titles(bundle, titles, duration, when, subscribers=None, channel_id=None, **kwargs):
    return predict_scores(bundle, titles, duration, when, subscribers, channel_id, **kwargs)


def score_existing_shorts(bundle, apply_rules=False, rules_context=None) -> pd.DataFrame:
    if apply_rules and bundle.get("version") not in ("v4", "v5"):
        raise ValueError("--apply-rules supports v4/v5 only")
    if not VIDEOS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(VIDEOS_PATH)
    mask = df["channel_id"].astype(str) == (bundle.get("testrun_channel_id") or TESTRUN_CHANNEL_ID)
    sub = df.loc[mask].copy()
    if sub.empty:
        return sub
    if bundle.get("version") in ("v3", "v4", "v5"):
        from src.ranking import attach_thumbnails, predict_frame
        cache = pd.read_parquet(THUMB_FEATURES_PATH) if THUMB_FEATURES_PATH.exists() else None
        enriched = attach_thumbnails(sub, cache=cache)
        _, scores, percentiles = predict_frame(bundle, enriched)
        sub["pred_score"] = scores
        sub["pred_percentile"] = percentiles
        if apply_rules:
            from src.proxy_rules import ProxyRuleContext
            context = rules_context if rules_context is not None else ProxyRuleContext(bundle)
            sub = context.apply(enriched,sub,score_column="pred_score")
        return sub.sort_values("pred_score", ascending=False)
    X_num = build_numeric_frame(sub)
    T = title_block(bundle, sub["title"].fillna("").astype(str).tolist())
    if is_v2(bundle) and THUMB_FEATURES_PATH.exists() and THUMB_EMBEDDINGS_PATH.exists():
        Th = stored_thumb_block(bundle, sub["video_id"].astype(str).tolist())
        X = np.hstack([X_num.to_numpy(dtype=float), T, Th])
    elif is_v2(bundle):
        Th = thumb_block_from_paths(bundle, [None] * len(sub))
        X = np.hstack([X_num.to_numpy(dtype=float), T, Th])
    else:
        X = np.hstack([X_num.to_numpy(dtype=float), T])
    Xs = bundle["scaler"].transform(X)
    deployed = bundle.get("deployed", "mlp")
    model = bundle["mlp"] if deployed == "mlp" else bundle["ridge_full"]
    scores = model.predict(Xs)
    ref = np.asarray(bundle.get("reference_pred_scores"), dtype=float)
    sub = sub.copy()
    sub["pred_score"] = scores
    sub["pred_percentile"] = [percentile_of(s, ref) for s in scores]
    if is_v2(bundle) and THUMB_FEATURES_PATH.exists():
        feats = pd.read_parquet(THUMB_FEATURES_PATH)
        keep = ["video_id"] + [c for c in INTERPRETABLE_FEATURES if c in feats.columns]
        sub = sub.merge(feats[keep], on="video_id", how="left")
    return sub.sort_values("pred_score", ascending=False)


def _print_rule_adjustment(rec):
    names = ", ".join(rec["rules_fired"]) or "none"
    print(f"      adjusted_score={rec['adjusted_score']:+.3f}; adjusted rank={rec['adjusted_rank']}; "
          f"delta={rec['rule_delta']:+.3f}; rules={names}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Shorts performance from a draft title and optional thumbnail(s)."
    )
    parser.add_argument("titles", nargs="*", help="One or more draft titles to compare")
    parser.add_argument("--duration", type=float, default=30.0, help="Planned length in seconds (default 30)")
    parser.add_argument(
        "--when",
        default=None,
        help="Planned post time, Africa/Lagos unless a timezone offset is included. "
        "Format: YYYY-MM-DD HH:MM. Default: next 10:00 Africa/Lagos.",
    )
    parser.add_argument("--subscribers", type=float, default=None, help="Override channel subscriber count")
    parser.add_argument("--score-existing", action="store_true", help="Also score TestRun's existing Shorts")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    parser.add_argument(
        "--model",
        default="auto",
        help="Which model to use: auto (v2 if present), v1, v2, v3, v4, v5, or a joblib path.",
    )
    parser.add_argument(
        "--thumbnail",
        action="append",
        default=[],
        dest="thumbnails",
        help="Thumbnail image path or URL. Repeat to compare several thumbnails "
        "for one title, or pass one per title. Amazon URLs are refused.",
    )
    parser.add_argument("--channel-id", default=None, help="Channel context for draft scoring (default: TestRun)")
    parser.add_argument("--recent-median-views", type=float, default=None,
                        help="Recent channel median views, known before posting (v3/v4/v5 only)")
    parser.add_argument("--apply-rules", action="store_true",
                        help="Add experimental adjusted_score/adjusted_rank fields for v4/v5; keep original scores")
    args = parser.parse_args()
    if args.recent_median_views is not None and (not np.isfinite(args.recent_median_views) or args.recent_median_views < 0):
        parser.error("--recent-median-views must be finite and nonnegative")

    bundle = load_bundle(args.model)
    rules_context = None
    if args.apply_rules:
        if bundle.get("version") not in ("v4", "v5"):
            parser.error("--apply-rules requires --model v4 or --model v5")
        from src.proxy_rules import ProxyRuleContext
        rules_context = ProxyRuleContext(bundle)
    when = parse_when(args.when)
    result = {
        "planned_post_time_africa_lagos": when.isoformat(),
        "duration": args.duration,
        "model_version": bundle.get("version"),
        "model_path": bundle.get("_path"),
    }

    thumb_specs = args.thumbnails or []
    tmpdir_ctx = tempfile.TemporaryDirectory(prefix="post-predictor-thumbs-")
    tmpdir = Path(tmpdir_ctx.name)
    try:
        resolved_thumbs: list[Path] | None = None
        if thumb_specs:
            resolved_thumbs = [fetch_thumbnail(s, tmpdir / f"t{i}") for i, s in enumerate(thumb_specs)]
            # If several thumbnails and one title, compare thumbs for that title.
            # If several titles and one thumbnail, reuse the thumbnail.
            # If several of both, they must match 1:1.
            if args.titles and len(resolved_thumbs) > 1 and len(args.titles) == 1:
                titles = list(args.titles) * len(resolved_thumbs)
                thumbs = resolved_thumbs
            elif args.titles and len(resolved_thumbs) == 1:
                titles = list(args.titles)
                thumbs = resolved_thumbs
            elif args.titles and len(resolved_thumbs) == len(args.titles):
                titles = list(args.titles)
                thumbs = resolved_thumbs
            elif not args.titles:
                sys.exit("Pass a title when comparing thumbnails.")
            else:
                sys.exit("Pass one title and several --thumbnail flags, or one thumbnail per title.")
            if not is_v2(bundle) and bundle.get("version") not in ("v3", "v4", "v5"):
                print("Warning: v1 ignores thumbnails. Use --model v2.", file=sys.stderr)
                thumbs = None
            table = predict_scores(
                bundle,
                titles,
                duration=args.duration,
                when=when,
                subscribers=args.subscribers,
                channel_id=args.channel_id,
                recent_median_views=args.recent_median_views,
                apply_rules=args.apply_rules,
                rules_context=rules_context,
                thumb_paths=thumbs,
            )
            # Keep the original thumbnail spec in the table for display.
            if thumbs is not None and len(table):
                # predict_scores sorts by score; map back by joining on row order after re-sort.
                # Easier: attach specs in the unsorted order then sort.
                pass
            result["drafts"] = table.astype(object).where(pd.notna(table), None).to_dict(orient="records")
            if not args.json:
                print(f"Planned post time: {when.strftime('%Y-%m-%d %H:%M')} Africa/Lagos")
                print(
                    f"Duration: {args.duration:.0f}s   Model: {bundle.get('version')} "
                    f"({bundle.get('deployed')})"
                )
                print()
                for rec in table.to_dict(orient="records"):
                    flag = "BEST" if rec["rank"] == 1 and len(table) > 1 else ""
                    thumb = rec.get("thumbnail") or ""
                    thumb_s = f"  thumb={thumb}" if thumb and thumb != "imputed" else ""
                    print(
                        f"  [{rec['rank']}] score={rec['score']:+.3f}  "
                        f"niche percentile={rec['niche_percentile']:.0f}  "
                        f"implied views ~{rec['implied_views_at_14d']:.0f} at 14d  {flag}"
                    )
                    print(f"      {rec['title']}{thumb_s}")
                    if args.apply_rules:
                        _print_rule_adjustment(rec)
                    if bundle.get("version") in ("v3", "v4", "v5"):
                        print(f"      ranking_score={rec['ranking_score']:+.3f}; confidence={rec['confidence']}: {rec['reliability_note']}")
                print()
                if bundle.get("version") not in ("v3", "v4", "v5"):
                    print(
                        "Score is predicted log-views vs this channel's typical Short, "
                        "after a simple age adjustment. Percentile is vs other niche Shorts "
                        "the model has scored. Implied views are a rough 14-day translation "
                        "using TestRun's own median, not a guarantee."
                    )
                if is_v2(bundle) and thumbs and any(t is None for t in thumbs):
                    print("Rows without a thumbnail used the training-set average thumbnail.")
        elif args.titles:
            table = predict_scores(
                bundle,
                list(args.titles),
                duration=args.duration,
                when=when,
                subscribers=args.subscribers,
                channel_id=args.channel_id,
                recent_median_views=args.recent_median_views,
                apply_rules=args.apply_rules,
                rules_context=rules_context,
                thumb_paths=None,
            )
            result["drafts"] = table.astype(object).where(pd.notna(table), None).to_dict(orient="records")
            if not args.json:
                print(f"Planned post time: {when.strftime('%Y-%m-%d %H:%M')} Africa/Lagos")
                print(
                    f"Duration: {args.duration:.0f}s   Model: {bundle.get('version')} "
                    f"({bundle.get('deployed')})"
                )
                if is_v2(bundle):
                    print("No --thumbnail given; v2 used the training-set average thumbnail.")
                print()
                for rec in table.to_dict(orient="records"):
                    flag = "BEST" if rec["rank"] == 1 and len(table) > 1 else ""
                    print(
                        f"  [{rec['rank']}] score={rec['score']:+.3f}  "
                        f"niche percentile={rec['niche_percentile']:.0f}  "
                        f"implied views ~{rec['implied_views_at_14d']:.0f} at 14d  {flag}"
                    )
                    print(f"      {rec['title']}")
                    if args.apply_rules:
                        _print_rule_adjustment(rec)
                    if bundle.get("version") in ("v3", "v4", "v5"):
                        reliability = rec['comparison_reliability']
                        pair_text = f"{reliability:.0%}" if np.isfinite(reliability) else "unavailable"
                        print(f"      ranking_score={rec['ranking_score']:+.3f}; confidence={rec['confidence']}; pair reliability={pair_text}; "
                              f"relative error (80% validation) ±{rec['relative_error_80pct']:.2f}")
                        print(f"      {rec['reliability_note']}")
                print()
                if bundle.get("version") not in ("v3", "v4", "v5"):
                    print(
                        "Score is predicted log-views vs this channel's typical Short, "
                        "after a simple age adjustment. Percentile is vs other niche Shorts "
                        "the model has scored. Implied views are a rough 14-day translation "
                        "using TestRun's own median, not a guarantee."
                    )

        if bundle.get("version") in ("v3", "v4", "v5") and args.titles and not args.json:
            print(f"{bundle.get('version')} orders by ranking_score; score is a validation-calibrated relative log-view estimate.")
            print("Percentiles use inner-validation predictions. Pair reliability is empirical, not a forecast guarantee.")
            if bundle.get("version") == "v5" and not bundle.get("selected_groups"):
                print("New public-prior and emotional ranking features were not retained; v5 uses v4 ranking.")
                print("JSON emotional_triggers are descriptive cue scores, not measured human reactions.")
            if not thumb_specs:
                print("Thumbnail missing: cheap image features were imputed; no CLIP model was loaded.")

        if args.score_existing or not (args.titles or thumb_specs):
            existing = score_existing_shorts(bundle, apply_rules=args.apply_rules, rules_context=rules_context)
            if existing.empty:
                print("No TestRun videos found in the dataset.")
            else:
                cols = [
                    c
                    for c in [
                        "title",
                        "video_id",
                        "views",
                        "duration",
                        "published_iso",
                        "age_days",
                        "pred_score",
                        "pred_percentile",
                        "thumb_brightness",
                        "thumb_contrast",
                        "thumb_colorfulness",
                        "thumb_letterbox_frac",
                        "thumb_text_present",
                        "thumb_text_area_ratio",
                        "thumb_face_detected",
                        "clip_text_overlay",
                        "clip_bright_bold",
                        "adjusted_score", "adjusted_rank", "rule_delta", "unclipped_rule_delta",
                        "rules_fired", "rule_score_clipped", "proxy_rule_context",
                    ]
                    if c in existing.columns
                ]
                show = existing[cols].copy()
                result["existing_shorts"] = show.to_dict(orient="records")
                if not args.json:
                    print(f"\nTestRun existing Shorts ({bundle.get('version')} score, not used in training):")
                    for rec in show.sort_values("pred_score", ascending=False).to_dict(orient="records"):
                        age = rec["age_days"]
                        age_s = f"{age:.1f}d" if pd.notna(age) else "?"
                        views = rec["views"]
                        views_s = f"{int(views):,}" if pd.notna(views) else "?"
                        extra = ""
                        if "thumb_letterbox_frac" in rec and pd.notna(rec.get("thumb_letterbox_frac")):
                            extra = (
                                f"  letterbox={rec['thumb_letterbox_frac']:.2f}"
                                f"  contrast={rec.get('thumb_contrast', float('nan')):.2f}"
                                f"  text={rec.get('thumb_text_present', float('nan')):.0f}"
                                f"  face={rec.get('thumb_face_detected', float('nan')):.0f}"
                            )
                        print(
                            f"  score={rec['pred_score']:+.3f}  pctl={rec['pred_percentile']:.0f}  "
                            f"actual views={views_s}  age={age_s}{extra}"
                        )
                        print(f"      {rec['title']}")
                        if args.apply_rules:
                            _print_rule_adjustment(rec)

        if args.apply_rules and not args.json:
            print("Rules add adjusted_score/adjusted_rank only; original ordering, percentiles and view estimates are unchanged.")
        if args.json:
            print(json.dumps(result, indent=2, default=str))
    finally:
        tmpdir_ctx.cleanup()


if __name__ == "__main__":
    main()
