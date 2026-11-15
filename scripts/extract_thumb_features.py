#!/usr/bin/env python3
"""Compute interpretable + CLIP features for every downloaded thumbnail."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    THUMB_DIR,
    THUMB_EMBEDDINGS_PATH,
    THUMB_FEATURES_PATH,
    VIDEOS_PATH,
)
from src.thumb_features import (  # noqa: E402
    CLIP_EMBED_DIM,
    CLIP_PROMPTS,
    INTERPRETABLE_FEATURES,
    ClipEncoder,
    interpretable_features,
    thumb_path_for_video,
)


def main() -> None:
    df = pd.read_parquet(VIDEOS_PATH, columns=["video_id", "title", "channel_id", "handle"])
    ids = df["video_id"].astype(str).tolist()
    paths = []
    missing = 0
    for vid in ids:
        p = thumb_path_for_video(vid)
        if p is None:
            missing += 1
            paths.append(None)
        else:
            paths.append(p)
    print(f"Videos: {len(ids)}  with thumbnail file: {len(ids) - missing}  missing: {missing}")

    # Interpretable features first (CPU OpenCV, can be serial; it's fast).
    rows = []
    ok = 0
    fail = 0
    present = [(i, p) for i, p in enumerate(paths) if p is not None]
    for n, (i, p) in enumerate(present, 1):
        feats = interpretable_features(p)
        if feats is None:
            fail += 1
            rows.append((i, None))
        else:
            ok += 1
            rows.append((i, feats))
        if n % 500 == 0 or n == len(present):
            print(f"  interpretable {n}/{len(present)}  ok={ok} fail={fail}", flush=True)

    print("Encoding CLIP ViT-B/32 embeddings on CPU...")
    encoder = ClipEncoder()
    encoder.load()
    valid_paths = [p for _i, p in present if p is not None]
    # We need alignment with `present` (same order). Skip decode failures individually.
    emb = np.zeros((len(ids), CLIP_EMBED_DIM), dtype=np.float32)
    sim = np.zeros((len(ids), len(CLIP_PROMPTS)), dtype=np.float32)
    has_emb = np.zeros(len(ids), dtype=bool)
    # Batch only the rows whose interpretable features succeeded.
    encode_items = [(i, p) for (i, p), (_j, feats) in zip(present, rows) if feats is not None]
    batch_size = 32
    for start in range(0, len(encode_items), batch_size):
        chunk = encode_items[start : start + batch_size]
        chunk_paths = [p for _i, p in chunk]
        e, s = encoder.encode_paths(chunk_paths, batch_size=len(chunk_paths))
        for k, (idx, _p) in enumerate(chunk):
            if np.linalg.norm(e[k]) > 0:
                emb[idx] = e[k]
                sim[idx] = s[k]
                has_emb[idx] = True
        done = min(start + batch_size, len(encode_items))
        if done % 256 == 0 or done == len(encode_items):
            print(f"  clip {done}/{len(encode_items)}", flush=True)

    prompt_names = encoder.prompt_names
    records = []
    for i, vid in enumerate(ids):
        rec = {"video_id": vid, "has_thumbnail": False, "has_embedding": bool(has_emb[i])}
        for name in INTERPRETABLE_FEATURES:
            rec[name] = np.nan
        # find feats
        # rows is only for present thumbs; build a map
        records.append(rec)

    feat_map = {i: feats for (i, _p), (_j, feats) in zip(present, rows) if feats is not None}
    for i, rec in enumerate(records):
        feats = feat_map.get(i)
        if not feats:
            continue
        rec["has_thumbnail"] = True
        for k, v in feats.items():
            rec[k] = v
        if has_emb[i]:
            for j, name in enumerate(prompt_names):
                rec[name] = float(sim[i, j])

    out = pd.DataFrame.from_records(records)
    THUMB_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(THUMB_FEATURES_PATH, index=False)
    np.savez_compressed(
        THUMB_EMBEDDINGS_PATH,
        video_id=np.array(ids),
        embedding=emb,
        has_embedding=has_emb,
        prompt_names=np.array(prompt_names),
        prompt_sim=sim,
    )
    print(f"Saved features: {THUMB_FEATURES_PATH}  rows={len(out)}  with_thumb={int(out.has_thumbnail.sum())}")
    print(f"Saved embeddings: {THUMB_EMBEDDINGS_PATH}  with_emb={int(has_emb.sum())}")


if __name__ == "__main__":
    main()
