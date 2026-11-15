#!/usr/bin/env python3
"""Train v2 (title + numeric + thumbnail) and compare it to intact v1 and a thumb-only model.

Uses the same per-channel held-out split and the same target definition as v1
(age effect coefficients taken from the saved v1 bundle) so the comparison is fair.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    METRICS_V2_PATH,
    MIN_AGE_DAYS,
    MODEL_V1_PATH,
    MODEL_V2_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TESTRUN_CHANNEL_ID,
    THUMB_EMBEDDINGS_PATH,
    THUMB_FEATURES_PATH,
    VIDEOS_PATH,
)
from src.features import (  # noqa: E402
    NUMERIC_FEATURES,
    add_targets,
    build_numeric_frame,
    eligible_training_mask,
    fill_channel_medians,
)
from src.thumb_features import INTERPRETABLE_FEATURES  # noqa: E402
from scripts.train import (  # noqa: E402
    HELD_OUT_FRACTION,
    SVD_DIM,
    apply_age_effect,
    channel_stratified_split,
    latin_letter_share,
    metrics_dict,
)

# Compress 512-d CLIP embeddings so the MLP stays small, same idea as title SVD.
THUMB_PCA_DIM = 32
MLP_KW = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-3,
    batch_size=64,
    learning_rate_init=1e-3,
    max_iter=400,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=RANDOM_SEED,
    verbose=False,
)


def load_thumb_tables() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    feats = pd.read_parquet(THUMB_FEATURES_PATH)
    blob = np.load(THUMB_EMBEDDINGS_PATH, allow_pickle=True)
    ids = blob["video_id"].astype(str)
    emb = blob["embedding"].astype(np.float32)
    has = blob["has_embedding"].astype(bool)
    # Align embeddings to the features frame by video_id.
    order = {vid: i for i, vid in enumerate(ids)}
    idx = feats["video_id"].astype(str).map(order)
    emb_aligned = np.zeros((len(feats), emb.shape[1]), dtype=np.float32)
    has_aligned = np.zeros(len(feats), dtype=bool)
    ok = idx.notna()
    emb_aligned[ok.to_numpy()] = emb[idx[ok].astype(int).to_numpy()]
    has_aligned[ok.to_numpy()] = has[idx[ok].astype(int).to_numpy()]
    feats = feats.copy()
    feats["has_embedding"] = has_aligned
    return feats, emb_aligned, has_aligned


def grouped_permutation(model, X, y, feature_blocks, rng, n_repeats=8):
    base = metrics_dict(y, model.predict(X), rng).get("spearman") or 0.0
    rows = []
    for name, cols in feature_blocks.items():
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, cols] = Xp[perm][:, cols]
            s = metrics_dict(y, model.predict(Xp), rng).get("spearman") or 0.0
            drops.append(base - s)
        rows.append(
            {
                "feature": name,
                "spearman_drop_mean": float(np.mean(drops)),
                "spearman_drop_std": float(np.std(drops)),
            }
        )
    rows.sort(key=lambda r: r["spearman_drop_mean"], reverse=True)
    return base, rows


def fit_mlp(X, y) -> MLPRegressor:
    m = MLPRegressor(**MLP_KW)
    m.fit(X, y)
    return m


def main() -> None:
    if not MODEL_V1_PATH.exists():
        sys.exit(f"Missing v1 model at {MODEL_V1_PATH}")
    if not THUMB_FEATURES_PATH.exists() or not THUMB_EMBEDDINGS_PATH.exists():
        sys.exit("Missing thumbnail features. Run scripts/extract_thumb_features.py first.")

    v1 = joblib.load(MODEL_V1_PATH)
    age_a = float(v1["age_a"])
    age_b = float(v1["age_b"])
    print(f"Loaded v1 bundle; reusing age effect a={age_a:.4f} b={age_b:.4f}")

    df = pd.read_parquet(VIDEOS_PATH)
    thumb_feats, emb, has_emb = load_thumb_tables()
    df = df.merge(thumb_feats, on="video_id", how="left", suffixes=("", "_thumb"))
    # After merge, embeddings still follow thumb_feats row order, not df order.
    emb_map = {vid: emb[i] for i, vid in enumerate(thumb_feats["video_id"].astype(str))}
    has_map = {vid: bool(has_emb[i]) for i, vid in enumerate(thumb_feats["video_id"].astype(str))}
    df["has_thumbnail"] = df["has_thumbnail"].fillna(False).astype(bool)
    df["has_embedding"] = df["video_id"].astype(str).map(has_map).fillna(False).astype(bool)

    eligible = eligible_training_mask(df)
    df = fill_channel_medians(df, eligible)
    df = add_targets(df)
    df["target"] = apply_age_effect(
        df["rel_log_views"].to_numpy(), df["age_days"].to_numpy(), age_a, age_b
    )

    is_testrun = df["channel_id"].astype(str) == TESTRUN_CHANNEL_ID
    latin_ok = df["title"].fillna("").map(latin_letter_share) >= 0.8
    train_pool = df.loc[eligible & ~is_testrun & latin_ok].copy()
    print(
        f"Eligible pool (ex-TestRun, English-script): {len(train_pool)}  "
        f"with thumb+emb: {int((train_pool.has_thumbnail & train_pool.has_embedding).sum())}"
    )

    # Exact same split as v1 (same function, seed, and pool membership).
    train_idx, test_idx = channel_stratified_split(train_pool, HELD_OUT_FRACTION, RANDOM_SEED)
    train_df = train_pool.loc[train_idx].copy()
    test_df = train_pool.loc[test_idx].copy()
    print(f"Full split: train={len(train_df)} held-out={len(test_df)}")

    thumb_ok_train = train_df["has_thumbnail"] & train_df["has_embedding"]
    thumb_ok_test = test_df["has_thumbnail"] & test_df["has_embedding"]
    # Fair comparison set: held-out rows that have a usable thumbnail.
    # v1 is evaluated on this same set so the numbers are comparable.
    if int(thumb_ok_test.sum()) < 50:
        sys.exit(f"Too few held-out thumbnails ({int(thumb_ok_test.sum())})")
    dropped_train = int((~thumb_ok_train).sum())
    dropped_test = int((~thumb_ok_test).sum())
    print(f"Thumbnail coverage: drop {dropped_train} train / {dropped_test} held-out rows without thumbs")

    train_v2 = train_df.loc[thumb_ok_train].copy()
    test_v2 = test_df.loc[thumb_ok_test].copy()
    y_train = train_v2["target"].to_numpy()
    y_test = test_v2["target"].to_numpy()
    print(f"v2 comparison set: train={len(train_v2)} held-out={len(test_v2)}")

    # --- v1 features on the comparison set (reuse v1 vectorizers) ---
    def v1_matrix(frame: pd.DataFrame, fit: bool = False):
        X_num = build_numeric_frame(frame)
        titles = frame["title"].fillna("").astype(str)
        if fit:
            # Should not refit; use the saved v1 transformers.
            raise RuntimeError("v1 transformers are already fitted")
        text = hstack(
            [
                v1["tfidf_word"].transform(titles),
                v1["tfidf_char"].transform(titles),
            ]
        ).tocsr()
        X = np.hstack([X_num.to_numpy(dtype=float), v1["svd"].transform(text)])
        return X, X_num

    X_test_v1, _ = v1_matrix(test_v2)
    X_test_v1_s = v1["scaler"].transform(X_test_v1)
    v1_model = v1["mlp"] if v1.get("deployed", "mlp") == "mlp" else v1["ridge_full"]
    rng = np.random.default_rng(RANDOM_SEED)
    results = {
        "v1_mlp": {"held_out": metrics_dict(y_test, v1_model.predict(X_test_v1_s), rng)},
    }

    # Also score v1 dummy / ridge if present, for completeness.
    if "dummy" in v1:
        results["v1_dummy_mean"] = {"held_out": metrics_dict(y_test, v1["dummy"].predict(X_test_v1_s), rng)}
    if "ridge_full" in v1:
        results["v1_ridge_text"] = {"held_out": metrics_dict(y_test, v1["ridge_full"].predict(X_test_v1_s), rng)}
    if "ridge_numeric" in v1:
        X_num_test = build_numeric_frame(test_v2).to_numpy(dtype=float)
        results["v1_ridge_numeric"] = {
            "held_out": metrics_dict(y_test, v1["ridge_numeric"].predict(X_num_test), rng)
        }

    # --- Thumbnail matrices ---
    def thumb_interp_matrix(frame: pd.DataFrame) -> np.ndarray:
        cols = [c for c in INTERPRETABLE_FEATURES if c in frame.columns]
        X = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        # Column-wise median imputation (should be rare inside the ok subset).
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        inds = np.where(~np.isfinite(X))
        if len(inds[0]):
            X[inds] = med[inds[1]]
        return X, cols

    def gather_embeddings(frame: pd.DataFrame) -> np.ndarray:
        return np.stack([emb_map[str(v)] for v in frame["video_id"].astype(str)], axis=0)

    X_th_tr, thumb_cols = thumb_interp_matrix(train_v2)
    X_th_te, _ = thumb_interp_matrix(test_v2)
    E_tr = gather_embeddings(train_v2)
    E_te = gather_embeddings(test_v2)

    pca_dim = min(THUMB_PCA_DIM, max(4, E_tr.shape[0] - 1), E_tr.shape[1])
    thumb_pca = PCA(n_components=pca_dim, random_state=RANDOM_SEED)
    E_tr_d = thumb_pca.fit_transform(E_tr)
    E_te_d = thumb_pca.transform(E_te)
    print(
        f"CLIP PCA dim={pca_dim}  explained={thumb_pca.explained_variance_ratio_.sum():.3f}"
    )

    # Title + numeric (refit on the v2 train set so v2 is a self-contained bundle).
    # Refitting TF-IDF on almost the same rows as v1 should be nearly identical.
    titles_tr = train_v2["title"].fillna("").astype(str)
    titles_te = test_v2["title"].fillna("").astype(str)
    tfidf = TfidfVectorizer(
        lowercase=True, min_df=3, max_df=0.9, ngram_range=(1, 2),
        max_features=4000, strip_accents="unicode",
    )
    char_tfidf = TfidfVectorizer(
        lowercase=True, min_df=3, max_df=0.95, analyzer="char_wb",
        ngram_range=(3, 5), max_features=2500, strip_accents="unicode",
    )
    word_tr, word_te = tfidf.fit_transform(titles_tr), tfidf.transform(titles_te)
    char_tr, char_te = char_tfidf.fit_transform(titles_tr), char_tfidf.transform(titles_te)
    text_tr = hstack([word_tr, char_tr]).tocsr()
    text_te = hstack([word_te, char_te]).tocsr()
    svd_dim = min(SVD_DIM, max(8, text_tr.shape[1] - 1), max(8, text_tr.shape[0] - 1))
    svd = TruncatedSVD(n_components=svd_dim, random_state=RANDOM_SEED)
    T_tr = svd.fit_transform(text_tr)
    T_te = svd.transform(text_te)
    N_tr = build_numeric_frame(train_v2).to_numpy(dtype=float)
    N_te = build_numeric_frame(test_v2).to_numpy(dtype=float)

    # Feature blocks: [numeric | title_svd | thumb_interp | thumb_pca]
    X_tr = np.hstack([N_tr, T_tr, X_th_tr, E_tr_d])
    X_te = np.hstack([N_te, T_te, X_th_te, E_te_d])
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

    n_num, n_title, n_th, n_pca = N_tr.shape[1], T_tr.shape[1], X_th_tr.shape[1], E_tr_d.shape[1]
    blocks = {
        "numeric_title_meta": list(range(0, n_num)),
        "title_text_svd": list(range(n_num, n_num + n_title)),
        "thumb_interpretable": list(range(n_num + n_title, n_num + n_title + n_th)),
        "thumb_clip_pca": list(range(n_num + n_title + n_th, n_num + n_title + n_th + n_pca)),
    }
    # Also permute each interpretable thumb feature.
    for i, name in enumerate(thumb_cols):
        blocks[name] = [n_num + n_title + i]

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_tr_s, y_train)
    results["v2_dummy_mean"] = {
        "train": metrics_dict(y_train, dummy.predict(X_tr_s), rng),
        "held_out": metrics_dict(y_test, dummy.predict(X_te_s), rng),
    }

    ridge_v2 = Ridge(alpha=3.0, random_state=RANDOM_SEED)
    ridge_v2.fit(X_tr_s, y_train)
    results["v2_ridge"] = {
        "train": metrics_dict(y_train, ridge_v2.predict(X_tr_s), rng),
        "held_out": metrics_dict(y_test, ridge_v2.predict(X_te_s), rng),
    }

    mlp_v2 = fit_mlp(X_tr_s, y_train)
    results["v2_mlp"] = {
        "train": metrics_dict(y_train, mlp_v2.predict(X_tr_s), rng),
        "held_out": metrics_dict(y_test, mlp_v2.predict(X_te_s), rng),
        "n_iter": int(mlp_v2.n_iter_),
    }

    # Thumbnail-only model: interpretable + CLIP PCA, no title/numeric.
    Xth_tr = np.hstack([X_th_tr, E_tr_d])
    Xth_te = np.hstack([X_th_te, E_te_d])
    th_scaler = StandardScaler()
    Xth_tr_s, Xth_te_s = th_scaler.fit_transform(Xth_tr), th_scaler.transform(Xth_te)
    ridge_th = Ridge(alpha=3.0, random_state=RANDOM_SEED)
    ridge_th.fit(Xth_tr_s, y_train)
    mlp_th = fit_mlp(Xth_tr_s, y_train)
    results["thumb_only_ridge"] = {
        "train": metrics_dict(y_train, ridge_th.predict(Xth_tr_s), rng),
        "held_out": metrics_dict(y_test, ridge_th.predict(Xth_te_s), rng),
    }
    results["thumb_only_mlp"] = {
        "train": metrics_dict(y_train, mlp_th.predict(Xth_tr_s), rng),
        "held_out": metrics_dict(y_test, mlp_th.predict(Xth_te_s), rng),
        "n_iter": int(mlp_th.n_iter_),
    }

    print("\n=== Held-out metrics (same split and target as v1; rows with thumbnails) ===")
    order = [
        "v1_dummy_mean",
        "v1_ridge_numeric",
        "v1_ridge_text",
        "v1_mlp",
        "thumb_only_ridge",
        "thumb_only_mlp",
        "v2_dummy_mean",
        "v2_ridge",
        "v2_mlp",
    ]
    for name in order:
        if name not in results:
            continue
        ho = results[name]["held_out"]
        print(
            f"{name:20s}  n={ho.get('n')}  spearman={ho.get('spearman')}  "
            f"pairwise={ho.get('pairwise_accuracy')}  mae={ho.get('mae')}  r2={ho.get('r2')}"
        )

    base_s, importances = grouped_permutation(mlp_v2, X_te_s, y_test, blocks, rng)
    print("\n=== v2 permutation importance (held-out Spearman drop) ===")
    for rec in importances[:20]:
        print(f"  {rec['feature']:24s}  drop={rec['spearman_drop_mean']:.4f} ± {rec['spearman_drop_std']:.4f}")

    # Linear direction for interpretable thumb traits (standardized ridge on thumb-only interp).
    th_only_scaler = StandardScaler()
    Z_tr = th_only_scaler.fit_transform(X_th_tr)
    Z_te = th_only_scaler.transform(X_th_te)
    ridge_interp = Ridge(alpha=2.0, random_state=RANDOM_SEED)
    ridge_interp.fit(Z_tr, y_train)
    interp_coefs = [
        {"feature": k, "coef": float(c)}
        for k, c in sorted(zip(thumb_cols, ridge_interp.coef_), key=lambda kv: abs(kv[1]), reverse=True)
    ]
    print("\n=== Thumbnail-only ridge coefficients (standardized, direction of residual) ===")
    for rec in interp_coefs[:15]:
        print(f"  {rec['feature']:24s}  {rec['coef']:+.4f}")

    # Univariate Spearman of each interpretable thumb feature vs target on held-out.
    from scipy.stats import spearmanr

    uni = []
    for i, name in enumerate(thumb_cols):
        r = spearmanr(X_th_te[:, i], y_test)
        uni.append(
            {
                "feature": name,
                "spearman": float(r.correlation) if r.correlation == r.correlation else None,
                "pvalue": float(r.pvalue) if r.pvalue == r.pvalue else None,
            }
        )
    uni.sort(key=lambda r: abs(r["spearman"] or 0), reverse=True)
    print("\n=== Held-out univariate Spearman, interpretable thumbnail features ===")
    for rec in uni[:15]:
        print(f"  {rec['feature']:24s}  r={rec['spearman']}  p={rec['pvalue']}")

    deployed = "mlp"
    if (results["v2_ridge"]["held_out"].get("spearman") or -1) > (
        results["v2_mlp"]["held_out"].get("spearman") or -1
    ):
        deployed = "ridge"

    # Reference scores for percentiles: v2 model on train+held-in pool with thumbs.
    pool_ok = train_pool["has_thumbnail"] & train_pool["has_embedding"]
    pool = train_pool.loc[pool_ok].copy()
    # Build pool matrix with the v2 transformers.
    N_pool = build_numeric_frame(pool).to_numpy(dtype=float)
    titles_pool = pool["title"].fillna("").astype(str)
    T_pool = svd.transform(
        hstack([tfidf.transform(titles_pool), char_tfidf.transform(titles_pool)]).tocsr()
    )
    Xth_pool, _ = thumb_interp_matrix(pool)
    E_pool_d = thumb_pca.transform(gather_embeddings(pool))
    X_pool = np.hstack([N_pool, T_pool, Xth_pool, E_pool_d])
    X_pool_s = scaler.transform(X_pool)
    model_for_ref = mlp_v2 if deployed == "mlp" else ridge_v2
    ref_scores = model_for_ref.predict(X_pool_s)

    # Median interpretable features (used when the CLI is called without a thumbnail).
    median_thumb = {c: float(train_v2[c].median()) for c in thumb_cols if c in train_v2.columns}
    mean_emb = E_tr.mean(axis=0)

    bundle = {
        "version": "v2",
        "tfidf_word": tfidf,
        "tfidf_char": char_tfidf,
        "svd": svd,
        "scaler": scaler,
        "mlp": mlp_v2,
        "ridge_full": ridge_v2,
        "dummy": dummy,
        "deployed": deployed,
        "numeric_features": NUMERIC_FEATURES,
        "thumb_interpretable_features": thumb_cols,
        "thumb_pca": thumb_pca,
        "thumb_pca_dim": pca_dim,
        "thumb_only_scaler": th_scaler,
        "thumb_only_mlp": mlp_th,
        "thumb_only_ridge": ridge_th,
        "median_thumb_features": median_thumb,
        "mean_clip_embedding": mean_emb,
        "age_a": age_a,
        "age_b": age_b,
        "testrun_channel_id": TESTRUN_CHANNEL_ID,
        "testrun_median_views": float(v1.get("testrun_median_views") or float("nan")),
        "testrun_subscribers": float(v1.get("testrun_subscribers") or 8.0),
        "reference_pred_scores": ref_scores,
        "random_seed": RANDOM_SEED,
        "svd_dim": svd_dim,
        "clip_model": "open_clip ViT-B-32 openai",
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_V2_PATH)
    (MODELS_DIR / "v2").mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / "v2" / "model.joblib")

    def ho(name):
        rec = results.get(name, {}).get("held_out", {})
        return {
            "n": rec.get("n"),
            "spearman": rec.get("spearman"),
            "pairwise_accuracy": rec.get("pairwise_accuracy"),
            "mae": rec.get("mae"),
            "r2": rec.get("r2"),
        }

    summary = {
        "version": "v2",
        "comparison_set": {
            "train": int(len(train_v2)),
            "held_out": int(len(test_v2)),
            "dropped_train_without_thumb": dropped_train,
            "dropped_held_out_without_thumb": dropped_test,
            "min_age_days": MIN_AGE_DAYS,
            "same_split_as_v1": True,
            "same_age_effect_as_v1": True,
        },
        "metrics": results,
        "headline": {
            "v1_mlp": ho("v1_mlp"),
            "v2_mlp": ho("v2_mlp"),
            "thumb_only_mlp": ho("thumb_only_mlp"),
            "thumb_only_ridge": ho("thumb_only_ridge"),
            "v2_ridge": ho("v2_ridge"),
        },
        "deployed_model": deployed,
        "permutation_importance": importances,
        "thumb_ridge_coefs": interp_coefs,
        "thumb_univariate_spearman": uni,
        "clip_pca_explained": float(thumb_pca.explained_variance_ratio_.sum()),
    }
    METRICS_V2_PATH.write_text(json.dumps(summary, indent=2, default=str))
    (REPORTS_DIR / "train_summary_v2.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved v2 model to {MODEL_V2_PATH}")
    print(f"Saved v2 metrics to {METRICS_V2_PATH}")
    print(f"v1 model at {MODEL_V1_PATH} was not modified.")


if __name__ == "__main__":
    main()
