#!/usr/bin/env python3
"""Fair pairwise recompute, TestRun oar2 vs custom-thumb comparison, feature profile."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    MODEL_V1_PATH,
    MODEL_V2_PATH,
    RANDOM_SEED,
    REPORTS_DIR,
    TESTRUN_CHANNEL_ID,
    THUMB_EMBEDDINGS_PATH,
    THUMB_FEATURES_PATH,
    VIDEOS_PATH,
)
from src.features import (  # noqa: E402
    add_targets,
    build_numeric_frame,
    eligible_training_mask,
    fill_channel_medians,
)
from src.thumb_features import ClipEncoder, features_from_path  # noqa: E402
from scripts.train import (  # noqa: E402
    HELD_OUT_FRACTION,
    apply_age_effect,
    channel_stratified_split,
    latin_letter_share,
    metrics_dict,
)
from scripts.predict import percentile_of, _implied_views  # noqa: E402
from scripts.train_v2 import load_thumb_tables  # noqa: E402

TESTRUN_THUMBS = ROOT / "data" / "testrun_thumbs"
NON_SERIES = {
    "Rt2pRmGQPNw",  # July 20, 2025
    "WEzENMO5NF0",  # coding
    "7Gjn-zDyrKo",  # Pascal DSA
}


def pairwise_exact(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    dt = y_true[:, None] - y_true[None, :]
    dp = y_pred[:, None] - y_pred[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), 1) & (dt != 0)
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(dt[mask]) == np.sign(dp[mask])).mean())


def crop_center_9x16(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    target_w = max(1, int(round(h * 9 / 16)))
    if w <= target_w + 2:
        return bgr
    x0 = (w - target_w) // 2
    return bgr[:, x0 : x0 + target_w]


def save_cropped_vertical(src: Path, dest: Path, out_w=1080, out_h=1920) -> bool:
    data = np.fromfile(str(src), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        return False
    cropped = crop_center_9x16(bgr)
    resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return False
    dest.write_bytes(buf.tobytes())
    return True


def score_with_thumbs(bundle, frame: pd.DataFrame, thumb_paths: list[Path], encoder) -> np.ndarray:
    from scripts.predict import title_block, thumb_block_from_paths

    X_num = build_numeric_frame(frame)
    T = title_block(bundle, frame["title"].fillna("").astype(str).tolist())
    # thumb_block_from_paths loads encoder itself if needed; pass paths
    Th = thumb_block_from_paths(bundle, thumb_paths)
    X = np.hstack([X_num.to_numpy(dtype=float), T, Th])
    Xs = bundle["scaler"].transform(X)
    model = bundle["mlp"] if bundle.get("deployed", "mlp") == "mlp" else bundle["ridge_full"]
    return model.predict(Xs)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    v1 = joblib.load(MODEL_V1_PATH)
    v2 = joblib.load(MODEL_V2_PATH)
    age_a, age_b = float(v1["age_a"]), float(v1["age_b"])

    df = pd.read_parquet(VIDEOS_PATH)
    thumb_feats, emb, has_emb = load_thumb_tables()
    df = df.merge(thumb_feats, on="video_id", how="left")
    emb_map = {vid: emb[i] for i, vid in enumerate(thumb_feats["video_id"].astype(str))}
    has_map = {vid: bool(has_emb[i]) for i, vid in enumerate(thumb_feats["video_id"].astype(str))}
    df["has_thumbnail"] = df["has_thumbnail"].fillna(False).astype(bool)
    df["has_embedding"] = df["video_id"].astype(str).map(has_map).fillna(False).astype(bool)

    eligible = eligible_training_mask(df)
    df = fill_channel_medians(df, eligible)
    df = add_targets(df)
    df["target"] = apply_age_effect(df["rel_log_views"].to_numpy(), df["age_days"].to_numpy(), age_a, age_b)

    is_testrun = df["channel_id"].astype(str) == TESTRUN_CHANNEL_ID
    latin_ok = df["title"].fillna("").map(latin_letter_share) >= 0.8
    train_pool = df.loc[eligible & ~is_testrun & latin_ok].copy()
    train_idx, test_idx = channel_stratified_split(train_pool, HELD_OUT_FRACTION, RANDOM_SEED)
    test_v2 = train_pool.loc[test_idx].copy()
    y_test = test_v2["target"].to_numpy()
    print(f"Held-out comparison set n={len(test_v2)}")

    # v1 features
    X_num = build_numeric_frame(test_v2)
    titles = test_v2["title"].fillna("").astype(str)
    text = hstack([v1["tfidf_word"].transform(titles), v1["tfidf_char"].transform(titles)]).tocsr()
    X_v1 = np.hstack([X_num.to_numpy(dtype=float), v1["svd"].transform(text)])
    pred_v1 = v1["mlp"].predict(v1["scaler"].transform(X_v1))

    # v2 / thumb-only using saved transformers
    N = build_numeric_frame(test_v2).to_numpy(dtype=float)
    T = v2["svd"].transform(
        hstack([v2["tfidf_word"].transform(titles), v2["tfidf_char"].transform(titles)]).tocsr()
    )
    cols = v2["thumb_interpretable_features"]
    X_th = test_v2[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(X_th, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = np.where(~np.isfinite(X_th))
    if len(bad[0]):
        X_th[bad] = med[bad[1]]
    E = np.stack([emb_map[str(v)] for v in test_v2["video_id"].astype(str)], axis=0)
    E_d = v2["thumb_pca"].transform(E)
    X_v2 = np.hstack([N, T, X_th, E_d])
    pred_v2 = v2["mlp"].predict(v2["scaler"].transform(X_v2))
    pred_v2_ridge = v2["ridge_full"].predict(v2["scaler"].transform(X_v2))
    Xth = np.hstack([X_th, E_d])
    pred_th_mlp = v2["thumb_only_mlp"].predict(v2["thumb_only_scaler"].transform(Xth))
    pred_th_ridge = v2["thumb_only_ridge"].predict(v2["thumb_only_scaler"].transform(Xth))

    preds = {
        "v1_mlp": pred_v1,
        "v2_mlp": pred_v2,
        "v2_ridge": pred_v2_ridge,
        "thumb_only_mlp": pred_th_mlp,
        "thumb_only_ridge": pred_th_ridge,
    }
    exact_rows = []
    for name, pred in preds.items():
        sp = spearmanr(y_test, pred)
        rec = {
            "model": name,
            "n": int(len(y_test)),
            "spearman": float(sp.correlation),
            "spearman_pvalue": float(sp.pvalue),
            "pairwise_exact_all_pairs": pairwise_exact(y_test, pred),
            "mae": float(np.mean(np.abs(y_test - pred))),
            "r2": float(1 - np.sum((y_test - pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)),
        }
        exact_rows.append(rec)
        print(
            f"{name:20s}  spearman={rec['spearman']:.12f}  "
            f"pairwise_exact={rec['pairwise_exact_all_pairs']:.12f}  "
            f"mae={rec['mae']:.6f}  r2={rec['r2']:.6f}"
        )

    # Confirm v1 predictions match stored spearman
    print("v1 spearman matches stored:", abs(exact_rows[0]["spearman"] - 0.14235487108832698) < 1e-12)

    exact_path = REPORTS_DIR / "metrics_v2_exact_pairwise.json"
    exact_path.write_text(json.dumps({"held_out_n": int(len(y_test)), "models": exact_rows}, indent=2))
    print("Wrote", exact_path)

    # --- TestRun feature profile vs training ---
    tr = df.loc[is_testrun].copy()
    series = tr[~tr["video_id"].astype(str).isin(NON_SERIES)].copy()
    profile = []
    for c in cols:
        if c not in train_pool.columns or c not in series.columns:
            continue
        tr_vals = pd.to_numeric(series[c], errors="coerce")
        pool_vals = pd.to_numeric(train_pool[c], errors="coerce").dropna()
        med_tr = float(tr_vals.median())
        med_pool = float(pool_vals.median())
        # percentile of TestRun median within the training pool
        pctl = float((pool_vals <= med_tr).mean() * 100.0) if len(pool_vals) else float("nan")
        profile.append(
            {
                "feature": c,
                "testrun_series_median": med_tr,
                "testrun_series_mean": float(tr_vals.mean()),
                "train_pool_median": med_pool,
                "train_pool_mean": float(pool_vals.mean()),
                "testrun_median_percentile_in_train": pctl,
            }
        )
    profile_df = pd.DataFrame(profile)
    profile_df.to_csv(REPORTS_DIR / "testrun_thumb_profile.csv", index=False)
    print("\nTestRun series thumb profile (median vs train pool):")
    show = profile_df.sort_values("testrun_median_percentile_in_train")
    # print extremes
    print(show.head(8).to_string(index=False))
    print("...")
    print(show.tail(8).to_string(index=False))

    # --- Prepare cropped custom thumbs ---
    cropped_dir = TESTRUN_THUMBS / "custom_cropped"
    hq_dir = TESTRUN_THUMBS / "hqdefault"
    oar_dir = TESTRUN_THUMBS / "oar2"
    n_crop = 0
    for p in sorted(hq_dir.glob("*.jpg")):
        if save_cropped_vertical(p, cropped_dir / p.name):
            n_crop += 1
    print(f"\nCropped+resized custom thumbs: {n_crop}")

    enc = ClipEncoder()
    enc.load()

    # Score three thumbnail versions with the same titles/metadata
    from scripts.predict import title_block, thumb_block_from_paths

    def pack(paths):
        X_num = build_numeric_frame(tr)
        T = title_block(v2, tr["title"].fillna("").astype(str).tolist())
        Th = thumb_block_from_paths(v2, paths)
        X = np.hstack([X_num.to_numpy(dtype=float), T, Th])
        model = v2["mlp"] if v2.get("deployed", "mlp") == "mlp" else v2["ridge_full"]
        return model.predict(v2["scaler"].transform(X))

    ids = tr["video_id"].astype(str).tolist()
    oar_paths = [oar_dir / f"{vid}.jpg" for vid in ids]
    hq_paths = [hq_dir / f"{vid}.jpg" for vid in ids]
    crop_paths = [cropped_dir / f"{vid}.jpg" for vid in ids]
    for p in oar_paths + hq_paths + crop_paths:
        if not p.exists():
            raise SystemExit(f"missing {p}")

    print("Scoring oar2 / hqdefault / custom_cropped ...")
    s_oar = pack(oar_paths)
    s_hq = pack(hq_paths)
    s_crop = pack(crop_paths)

    # v1 scores on the same rows for comparison (title+numeric only)
    Xn = build_numeric_frame(tr)
    text_tr = hstack(
        [
            v1["tfidf_word"].transform(tr["title"].fillna("").astype(str)),
            v1["tfidf_char"].transform(tr["title"].fillna("").astype(str)),
        ]
    ).tocsr()
    Xv1 = np.hstack([Xn.to_numpy(dtype=float), v1["svd"].transform(text_tr)])
    s_v1 = v1["mlp"].predict(v1["scaler"].transform(Xv1))

    ref_v2 = np.asarray(v2.get("reference_pred_scores"), dtype=float)
    ref_v1 = np.asarray(v1.get("reference_pred_scores"), dtype=float)

    out = tr[
        [
            "video_id",
            "title",
            "views",
            "age_days",
            "duration",
            "is_short",
        ]
    ].copy()
    out["is_current_series"] = ~out["video_id"].astype(str).isin(NON_SERIES)
    out["v1_score"] = s_v1
    out["v1_percentile"] = [percentile_of(s, ref_v1) for s in s_v1]
    out["v2_score_oar2"] = s_oar
    out["v2_pctl_oar2"] = [percentile_of(s, ref_v2) for s in s_oar]
    out["v2_score_hqdefault"] = s_hq
    out["v2_pctl_hqdefault"] = [percentile_of(s, ref_v2) for s in s_hq]
    out["v2_score_custom_cropped"] = s_crop
    out["v2_pctl_custom_cropped"] = [percentile_of(s, ref_v2) for s in s_crop]
    out["delta_crop_minus_oar2"] = out["v2_score_custom_cropped"] - out["v2_score_oar2"]
    out["implied_views_14d_oar2"] = _implied_views(v2, s_oar)
    out["implied_views_14d_custom_cropped"] = _implied_views(v2, s_crop)

    # attach a few interpretable features from stored (oar2) table
    feat_cols = [
        "thumb_brightness",
        "thumb_contrast",
        "thumb_colorfulness",
        "thumb_saturation_mean",
        "thumb_letterbox_frac",
        "thumb_text_area_ratio",
        "thumb_text_top_frac",
        "thumb_face_detected",
        "thumb_bright_frac",
        "thumb_dark_frac",
        "thumb_dom_frac",
        "thumb_height",
        "thumb_width",
        "clip_text_overlay",
        "clip_phone_screen",
        "clip_code_screen",
        "clip_bright_bold",
        "clip_product_shot",
    ]
    feats = thumb_feats.set_index("video_id")
    for c in feat_cols:
        if c in feats.columns:
            out[c] = out["video_id"].astype(str).map(feats[c])

    out = out.sort_values("v2_score_oar2", ascending=False)
    csv_path = REPORTS_DIR / "testrun_scores_v2.csv"
    out.to_csv(csv_path, index=False)
    print("Wrote", csv_path)

    series_out = out[out["is_current_series"]]
    print("\n=== Current-series TestRun, ranked by v2 oar2/feed preview score ===")
    for rec in series_out.to_dict(orient="records"):
        print(
            f"  oar2={rec['v2_score_oar2']:+.3f}  crop={rec['v2_score_custom_cropped']:+.3f}  "
            f"d={rec['delta_crop_minus_oar2']:+.3f}  v1={rec['v1_score']:+.3f}  "
            f"views={int(rec['views']):5d}  {rec['title']}"
        )

    print("\n=== Non-series uploads ===")
    for rec in out[~out["is_current_series"]].to_dict(orient="records"):
        print(
            f"  oar2={rec['v2_score_oar2']:+.3f}  crop={rec['v2_score_custom_cropped']:+.3f}  "
            f"views={int(rec['views']):5d}  {rec['title']}"
        )

    def summarize(label, scores, views):
        r = spearmanr(scores, views)
        print(f"  {label}: spearman vs raw views r={r.correlation} p={r.pvalue}")

    print("\nDiagnostic (not official metrics): v2 scores vs raw TestRun views, current series")
    summarize("oar2", series_out["v2_score_oar2"], series_out["views"])
    summarize("custom_cropped", series_out["v2_score_custom_cropped"], series_out["views"])
    summarize("v1", series_out["v1_score"], series_out["views"])
    print(
        f"  custom_cropped beats oar2 on {int((series_out['delta_crop_minus_oar2']>0).sum())}"
        f"/{len(series_out)} current-series videos; "
        f"median delta={series_out['delta_crop_minus_oar2'].median():+.3f}; "
        f"mean delta={series_out['delta_crop_minus_oar2'].mean():+.3f}"
    )

    summary = {
        "exact_pairwise": exact_rows,
        "testrun_current_series_n": int(len(series_out)),
        "custom_cropped_beats_oar2_n": int((series_out["delta_crop_minus_oar2"] > 0).sum()),
        "delta_crop_minus_oar2_median": float(series_out["delta_crop_minus_oar2"].median()),
        "delta_crop_minus_oar2_mean": float(series_out["delta_crop_minus_oar2"].mean()),
        "spearman_vs_views_series": {
            "v2_oar2": {
                "r": float(spearmanr(series_out["v2_score_oar2"], series_out["views"]).correlation),
                "p": float(spearmanr(series_out["v2_score_oar2"], series_out["views"]).pvalue),
            },
            "v2_custom_cropped": {
                "r": float(spearmanr(series_out["v2_score_custom_cropped"], series_out["views"]).correlation),
                "p": float(spearmanr(series_out["v2_score_custom_cropped"], series_out["views"]).pvalue),
            },
            "v1": {
                "r": float(spearmanr(series_out["v1_score"], series_out["views"]).correlation),
                "p": float(spearmanr(series_out["v1_score"], series_out["views"]).pvalue),
            },
        },
    }
    (REPORTS_DIR / "testrun_thumb_comparison_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
