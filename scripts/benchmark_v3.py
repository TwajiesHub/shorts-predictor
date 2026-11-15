#!/usr/bin/env python3
"""Measure local v3 loading and three-draft inference, including large thumbnails."""
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from scripts.predict import load_bundle, predict_scores, parse_when


def main(version="v3"):
    bundle = load_bundle(version)
    loaded = time.perf_counter()-START
    titles = ['Why Does My 128GB iPhone Only Show 113GB Free?',
              'Your 128GB iPhone Is Lying About Its Storage',
              "Why 128GB on iPhone Isn't Really 128GB"]
    with tempfile.TemporaryDirectory() as tmp:
        images = [Path(tmp) / f'thumbnail_{i}.jpg' for i in range(3)]
        # Reproducible textured full-resolution frame, not a trivially blank image.
        pixels = np.random.default_rng(3).integers(0, 256, (1920, 1080, 3), dtype=np.uint8)
        for image in images:
            Image.fromarray(pixels).save(image)
        times = {}
        for name, paths in [('titles_only', None), ('shared_thumbnail', [images[0]]*3), ('three_thumbnails', images)]:
            durations = []
            for _ in range(5):
                start = time.perf_counter()
                out = predict_scores(bundle, titles, 30, parse_when('2026-11-16 10:00'), thumb_paths=paths)
                assert len(out) == 3 and np.isfinite(out.ranking_score).all()
                durations.append(time.perf_counter()-start)
            times[name] = dict(first_seconds=durations[0], median_seconds=float(np.median(durations)))
        assert 'torch' not in sys.modules and 'open_clip' not in sys.modules
    print(json.dumps(dict(load_and_import_seconds=loaded, timings=times,
                         peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                         model_mib=(ROOT/f"models/model_{version}.joblib").stat().st_size/1024**2,
                         torch_or_clip_loaded=False), indent=2))


if __name__ == '__main__':
    main()
