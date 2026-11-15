#!/usr/bin/env python3
"""Train a small MLP Shorts predictor and compare it to simple baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    METRICS_PATH,
    MIN_AGE_DAYS,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TESTRUN_CHANNEL_ID,
    TRAIN_SUMMARY_PATH,
    VIDEOS_PATH,
)
from src.features import (  # noqa: E402
    NUMERIC_FEATURES,
    add_targets,
    build_numeric_frame,
    eligible_training_mask,
    fill_channel_medians,
)

HELD_OUT_FRACTION = 0.2
SVD_DIM = 64
PAIR_SAMPLES = 8000


def pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_pairs: int, rng: np.random.Generator) -> float:
    n = len(y_true)
    if n < 2:
        return float("nan")
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    true_sign = np.sign(y_true[i] - y_true[j])
    pred_sign = np.sign(y_pred[i] - y_pred[j])
    comparable = true_sign != 0
    if comparable.sum() == 0:
        return float("nan")
    return float((true_sign[comparable] == pred_sign[comparable]).mean())


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, rng: np.random.Generator) -> dict:
    if len(y_true) < 3:
        return {"n": int(len(y_true)), "note": "too few rows"}
    spear = spearmanr(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "spearman": float(spear.correlation) if spear.correlation == spear.correlation else None,
        "spearman_pvalue": float(spear.pvalue) if spear.pvalue == spear.pvalue else None,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pairwise_accuracy": pairwise_accuracy(y_true, y_pred, PAIR_SAMPLES, rng),
    }


def time_split(df: pd.DataFrame, frac: float) -> tuple[pd.Index, pd.Index]:
    ordered = df.sort_values("published_ts")
    cut = int(len(ordered) * (1 - frac))
    cut = min(max(cut, 1), len(ordered) - 1)
    train_idx = ordered.index[:cut]
    test_idx = ordered.index[cut:]
    return train_idx, test_idx


def latin_letter_share(title: str) -> float:
    letters = [ch for ch in str(title) if ch.isalpha()]
    if not letters:
        return 1.0
    return sum(ch.isascii() for ch in letters) / len(letters)


def channel_stratified_split(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.Index, pd.Index]:
    """Hold out ~frac of each channel so no single channel dominates the test set."""
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for _, grp in df.groupby("channel_id"):
        idx = np.array(grp.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        if len(idx) == 1:
            # A single video can't be both; put it in train so medians stay defined.
            train_idx.extend(idx)
            continue
        n_test = max(1, int(round(len(idx) * frac)))
        n_test = min(n_test, len(idx) - 1)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return pd.Index(train_idx), pd.Index(test_idx)


def fit_age_effect(rel: np.ndarray, age_days: np.ndarray) -> tuple[float, float]:
    """Ordinary least squares: rel ≈ a + b * log1p(age)."""
    x = np.log1p(np.clip(age_days, 0, None))
    A = np.column_stack([np.ones(len(x)), x])
    try:
        coef, *_ = np.linalg.lstsq(A, rel, rcond=None)
        return float(coef[0]), float(coef[1])
    except np.linalg.LinAlgError:
        return 0.0, 0.0


def apply_age_effect(rel: np.ndarray, age_days: np.ndarray, a: float, b: float) -> np.ndarray:
    return rel - (a + b * np.log1p(np.clip(age_days, 0, None)))


def main() -> None:
    if not VIDEOS_PATH.exists():
        print(f"Missing {VIDEOS_PATH}. Run scripts/collect_data.py first.")
        sys.exit(1)

    df = pd.read_parquet(VIDEOS_PATH)
    print(f"Loaded {len(df)} videos, {df['channel_id'].nunique()} channels")

    eligible = eligible_training_mask(df)
    df = fill_channel_medians(df, eligible)
    df = add_targets(df)

    # Hold TestRun out of training / the main held-out set so later scoring is honest.
    is_testrun = df["channel_id"].astype(str) == TESTRUN_CHANNEL_ID
    # Drop titles that are mostly non-Latin (e.g. Hindi-only) — TestRun is English.
    latin_ok = df["title"].fillna("").map(latin_letter_share) >= 0.8
    train_pool = df.loc[eligible & ~is_testrun & latin_ok].copy()
    testrun_eligible = df.loc[eligible & is_testrun].copy()
    print(
        f"Eligible for training (ex-TestRun, English-script titles): {len(train_pool)} | "
        f"TestRun eligible: {len(testrun_eligible)} | "
        f"excluded (age < {MIN_AGE_DAYS}d, not a Short, missing fields, or non-Latin): "
        f"{int((~ (eligible & ~is_testrun & latin_ok)).sum())}"
    )
    if len(train_pool) < 80:
        print(f"ERROR: not enough training rows ({len(train_pool)})")
        sys.exit(1)

    # Primary split: per-channel random hold-out. A pure time split is unreliable
    # because most public upload dates from the playlist API are "N months ago"
    # approximations snapped to today's calendar day.
    train_idx, test_idx = channel_stratified_split(train_pool, HELD_OUT_FRACTION, RANDOM_SEED)
    train_df = train_pool.loc[train_idx].copy()
    test_df = train_pool.loc[test_idx].copy()
    time_train_idx, time_test_idx = time_split(train_pool, HELD_OUT_FRACTION)
    print(
        f"Channel-stratified split: train={len(train_df)}  held-out={len(test_df)}  "
        f"channels in held-out={test_df['channel_id'].nunique()}"
    )
    print(f"Time-split sizes (secondary, dates approximate): train={len(time_train_idx)} held-out={len(time_test_idx)}")

    # Age effect estimated on TRAIN only, then applied everywhere.
    age_a, age_b = fit_age_effect(
        train_df["rel_log_views"].to_numpy(),
        train_df["age_days"].to_numpy(),
    )
    print(f"Age effect on channel-relative log views: a={age_a:.4f}  b={age_b:.4f}")

    for frame in (train_df, test_df, train_pool, testrun_eligible, df):
        frame["target"] = apply_age_effect(
            frame["rel_log_views"].to_numpy(),
            frame["age_days"].to_numpy(),
            age_a,
            age_b,
        )

    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()

    X_num_train = build_numeric_frame(train_df)
    X_num_test = build_numeric_frame(test_df)

    titles_train = train_df["title"].fillna("").astype(str)
    titles_test = test_df["title"].fillna("").astype(str)

    tfidf = TfidfVectorizer(
        lowercase=True,
        min_df=3,
        max_df=0.9,
        ngram_range=(1, 2),
        max_features=4000,
        strip_accents="unicode",
    )
    char_tfidf = TfidfVectorizer(
        lowercase=True,
        min_df=3,
        max_df=0.95,
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=2500,
        strip_accents="unicode",
    )
    word_train = tfidf.fit_transform(titles_train)
    word_test = tfidf.transform(titles_test)
    char_train = char_tfidf.fit_transform(titles_train)
    char_test = char_tfidf.transform(titles_test)

    # Stack word + char TF-IDF, then compress so the MLP stays small.
    from scipy.sparse import hstack

    text_train = hstack([word_train, char_train]).tocsr()
    text_test = hstack([word_test, char_test]).tocsr()

    svd_dim = min(SVD_DIM, max(8, text_train.shape[1] - 1), max(8, text_train.shape[0] - 1))
    svd = TruncatedSVD(n_components=svd_dim, random_state=RANDOM_SEED)
    text_train_d = svd.fit_transform(text_train)
    text_test_d = svd.transform(text_test)
    print(f"TF-IDF shape={text_train.shape}  SVD dim={svd_dim}  explained={svd.explained_variance_ratio_.sum():.3f}")

    X_train = np.hstack([X_num_train.to_numpy(dtype=float), text_train_d])
    X_test = np.hstack([X_num_test.to_numpy(dtype=float), text_test_d])

    num_scaler = StandardScaler()
    # Scale only the numeric + SVD block together for the MLP / ridge.
    X_train_s = num_scaler.fit_transform(X_train)
    X_test_s = num_scaler.transform(X_test)

    rng = np.random.default_rng(RANDOM_SEED)
    results = {}

    # Baseline 1: always predict the mean residual (channel-median, age-adjusted = 0-ish).
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train_s, y_train)
    results["dummy_mean"] = {
        "train": metrics_dict(y_train, dummy.predict(X_train_s), rng),
        "held_out": metrics_dict(y_test, dummy.predict(X_test_s), rng),
    }

    # Baseline 2: Ridge on numeric features only (no title text).
    ridge_num = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=2.0, random_state=RANDOM_SEED)),
        ]
    )
    ridge_num.fit(X_num_train.to_numpy(dtype=float), y_train)
    results["ridge_numeric"] = {
        "train": metrics_dict(y_train, ridge_num.predict(X_num_train.to_numpy(dtype=float)), rng),
        "held_out": metrics_dict(y_test, ridge_num.predict(X_num_test.to_numpy(dtype=float)), rng),
    }

    # Baseline 3: Ridge on numeric + title SVD (linear text model).
    ridge_full = Ridge(alpha=3.0, random_state=RANDOM_SEED)
    ridge_full.fit(X_train_s, y_train)
    results["ridge_text"] = {
        "train": metrics_dict(y_train, ridge_full.predict(X_train_s), rng),
        "held_out": metrics_dict(y_test, ridge_full.predict(X_test_s), rng),
    }

    # Small neural net.
    mlp = MLPRegressor(
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
    mlp.fit(X_train_s, y_train)
    results["mlp"] = {
        "train": metrics_dict(y_train, mlp.predict(X_train_s), rng),
        "held_out": metrics_dict(y_test, mlp.predict(X_test_s), rng),
        "n_iter": int(mlp.n_iter_),
    }

    print("\n=== Held-out metrics (channel-normalized, age-adjusted log views) ===")
    for name, rec in results.items():
        ho = rec["held_out"]
        print(
            f"{name:16s}  n={ho.get('n')}  spearman={ho.get('spearman')}  "
            f"pairwise={ho.get('pairwise_accuracy')}  mae={ho.get('mae')}  r2={ho.get('r2')}"
        )

    # Permutation importance on grouped feature blocks (held-out).
    # We permute columns of the already-scaled matrix in meaningful groups.
    feature_blocks = {
        "title_text_svd": list(range(len(NUMERIC_FEATURES), X_test_s.shape[1])),
    }
    for i, name in enumerate(NUMERIC_FEATURES):
        feature_blocks[name] = [i]

    def grouped_permutation(model, X, y, n_repeats=8):
        base = metrics_dict(y, model.predict(X), rng)["spearman"] or 0.0
        importances = []
        for name, cols in feature_blocks.items():
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                # Shuffle those columns together by row permutation of the block.
                perm = rng.permutation(len(X))
                Xp[:, cols] = Xp[perm][:, cols]
                s = metrics_dict(y, model.predict(Xp), rng)["spearman"] or 0.0
                drops.append(base - s)
            importances.append(
                {
                    "feature": name,
                    "spearman_drop_mean": float(np.mean(drops)),
                    "spearman_drop_std": float(np.std(drops)),
                }
            )
        importances.sort(key=lambda r: r["spearman_drop_mean"], reverse=True)
        return base, importances

    base_s, importances = grouped_permutation(mlp, X_test_s, y_test)
    print("\n=== Permutation importance (held-out Spearman drop) ===")
    for rec in importances[:12]:
        print(f"  {rec['feature']:20s}  drop={rec['spearman_drop_mean']:.4f} ± {rec['spearman_drop_std']:.4f}")

    # Also surface top Ridge text n-grams for plain-language drivers.
    # Fit a Ridge directly on TF-IDF (not SVD) for interpretability.
    ridge_words = Ridge(alpha=3.0, random_state=RANDOM_SEED)
    ridge_words.fit(word_train, y_train)
    vocab = np.array(tfidf.get_feature_names_out())
    coefs = ridge_words.coef_
    top_pos = np.argsort(coefs)[-15:][::-1]
    top_neg = np.argsort(coefs)[:15]
    word_drivers = {
        "positive": [{"term": str(vocab[i]), "coef": float(coefs[i])} for i in top_pos],
        "negative": [{"term": str(vocab[i]), "coef": float(coefs[i])} for i in top_neg],
    }
    print("\n=== Title terms associated with better / worse than channel-typical views ===")
    print("  up:", ", ".join(f"{r['term']} ({r['coef']:.3f})" for r in word_drivers["positive"][:8]))
    print("  down:", ", ".join(f"{r['term']} ({r['coef']:.3f})" for r in word_drivers["negative"][:8]))

    # Numeric ridge coefficients (standardized).
    ridge_coef = dict(zip(NUMERIC_FEATURES, [float(c) for c in ridge_num.named_steps["ridge"].coef_]))
    top_numeric = sorted(ridge_coef.items(), key=lambda kv: abs(kv[1]), reverse=True)

    # Choose the deployed model: MLP if it beats ridge_text and dummy on held-out Spearman,
    # otherwise fall back to the better linear model and say so honestly.
    def ho_spear(name: str) -> float:
        v = results[name]["held_out"].get("spearman")
        return v if v is not None and v == v else -1.0

    deployed = "mlp"
    if ho_spear("mlp") < ho_spear("ridge_text") and ho_spear("ridge_text") >= ho_spear("dummy_mean"):
        deployed = "ridge_text"
    if ho_spear(deployed) < ho_spear("dummy_mean"):
        print("WARNING: no model beat the mean baseline on Spearman; still saving the MLP for the CLI.")
        deployed = "mlp"

    bundle = {
        "tfidf_word": tfidf,
        "tfidf_char": char_tfidf,
        "svd": svd,
        "scaler": num_scaler,
        "mlp": mlp,
        "ridge_full": ridge_full,
        "ridge_numeric": ridge_num,
        "dummy": dummy,
        "deployed": deployed,
        "numeric_features": NUMERIC_FEATURES,
        "age_a": age_a,
        "age_b": age_b,
        "channel_medians": (
            df.drop_duplicates("channel_id")
            .set_index("channel_id")["channel_median_views"]
            .to_dict()
        ),
        "channel_subs": (
            df.drop_duplicates("channel_id")
            .set_index("channel_id")["channel_subscribers"]
            .to_dict()
        ),
        "testrun_channel_id": TESTRUN_CHANNEL_ID,
        "testrun_median_views": float(
            df.loc[is_testrun & eligible, "views"].median()
            if (is_testrun & eligible).any()
            else df.loc[is_testrun, "views"].median()
            if is_testrun.any()
            else float("nan")
        ),
        "testrun_subscribers": float(
            df.loc[is_testrun, "channel_subscribers"].dropna().iloc[0]
            if is_testrun.any() and df.loc[is_testrun, "channel_subscribers"].notna().any()
            else 8.0
        ),
        # Percentile lookup: predicted scores on the training pool (for CLI percentiles).
        "train_target_scores": train_pool["target"].to_numpy(),
        "random_seed": RANDOM_SEED,
        "svd_dim": svd_dim,
    }

    # Store prediction reference distribution from the model on train+held-in pool.
    X_pool_num = build_numeric_frame(train_pool)
    text_pool = hstack(
        [
            tfidf.transform(train_pool["title"].fillna("").astype(str)),
            char_tfidf.transform(train_pool["title"].fillna("").astype(str)),
        ]
    ).tocsr()
    X_pool = np.hstack([X_pool_num.to_numpy(dtype=float), svd.transform(text_pool)])
    X_pool_s = num_scaler.transform(X_pool)
    model_for_ref = mlp if deployed == "mlp" else ridge_full
    bundle["reference_pred_scores"] = model_for_ref.predict(X_pool_s)

    # Secondary robustness check: time split. Public dates are coarse, so
    # this is reported but is not the official number.
    def eval_time_split():
        tr = train_pool.loc[time_train_idx]
        te = train_pool.loc[time_test_idx]
        a, b = fit_age_effect(tr["rel_log_views"].to_numpy(), tr["age_days"].to_numpy())
        ytr = apply_age_effect(tr["rel_log_views"].to_numpy(), tr["age_days"].to_numpy(), a, b)
        yte = apply_age_effect(te["rel_log_views"].to_numpy(), te["age_days"].to_numpy(), a, b)
        Xtr_n = build_numeric_frame(tr)
        Xte_n = build_numeric_frame(te)
        tf = TfidfVectorizer(lowercase=True, min_df=3, max_df=0.9, ngram_range=(1, 2), max_features=4000, strip_accents="unicode")
        cf = TfidfVectorizer(lowercase=True, min_df=3, max_df=0.95, analyzer="char_wb", ngram_range=(3, 5), max_features=2500, strip_accents="unicode")
        wtr, wte = tf.fit_transform(tr["title"].fillna("").astype(str)), tf.transform(te["title"].fillna("").astype(str))
        ctr, cte = cf.fit_transform(tr["title"].fillna("").astype(str)), cf.transform(te["title"].fillna("").astype(str))
        from scipy.sparse import hstack as sp_hstack
        ttr = sp_hstack([wtr, ctr]).tocsr()
        tte = sp_hstack([wte, cte]).tocsr()
        dim = min(SVD_DIM, max(8, ttr.shape[1] - 1), max(8, ttr.shape[0] - 1))
        sv = TruncatedSVD(n_components=dim, random_state=RANDOM_SEED)
        Xtr = np.hstack([Xtr_n.to_numpy(dtype=float), sv.fit_transform(ttr)])
        Xte = np.hstack([Xte_n.to_numpy(dtype=float), sv.transform(tte)])
        sc = StandardScaler()
        Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)
        out = {}
        dmy = DummyRegressor(strategy="mean"); dmy.fit(Xtr_s, ytr)
        out["dummy_mean"] = metrics_dict(yte, dmy.predict(Xte_s), rng)
        rg = Ridge(alpha=3.0, random_state=RANDOM_SEED); rg.fit(Xtr_s, ytr)
        out["ridge_text"] = metrics_dict(yte, rg.predict(Xte_s), rng)
        nn = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=1e-3,
                          batch_size=64, learning_rate_init=1e-3, max_iter=400, early_stopping=True,
                          validation_fraction=0.15, n_iter_no_change=20, random_state=RANDOM_SEED)
        nn.fit(Xtr_s, ytr)
        out["mlp"] = metrics_dict(yte, nn.predict(Xte_s), rng)
        return out

    time_split_metrics = eval_time_split()
    print("\n=== Time-split held-out (secondary; upload dates are approximate) ===")
    for name, ho in time_split_metrics.items():
        print(
            f"{name:16s}  n={ho.get('n')}  spearman={ho.get('spearman')}  "
            f"pairwise={ho.get('pairwise_accuracy')}  mae={ho.get('mae')}  r2={ho.get('r2')}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    summary = {
        "dataset": {
            "videos_total": int(len(df)),
            "channels_total": int(df["channel_id"].nunique()),
            "eligible_ex_testrun": int(len(train_pool)),
            "train": int(len(train_df)),
            "held_out": int(len(test_df)),
            "testrun_eligible": int(len(testrun_eligible)),
            "testrun_total": int(is_testrun.sum()),
            "min_age_days": MIN_AGE_DAYS,
            "split": "channel_stratified",
            "held_out_channels": int(test_df["channel_id"].nunique()),
        },
        "age_effect": {"intercept": age_a, "log1p_age_coef": age_b},
        "metrics": results,
        "time_split_metrics": time_split_metrics,
        "deployed_model": deployed,
        "permutation_importance": importances,
        "title_term_drivers": word_drivers,
        "numeric_ridge_coefs": [{"feature": k, "coef": v} for k, v in top_numeric],
        "held_out_notes": (
            "Official metrics use a per-channel random hold-out. "
            "Most public upload dates from the playlist API are coarse "
            "(often snapped to today's calendar day), so a time split is a weaker check."
        ),
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2, default=str))
    TRAIN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Deployed model: {deployed}")


if __name__ == "__main__":
    main()
