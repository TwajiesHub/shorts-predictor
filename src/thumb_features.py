"""Thumbnail features for the v2 Shorts predictor.

Uses public image files already saved under data/thumbnails (i.ytimg.com only).
Never fetches description or affiliate URLs.

Interpretable features are computed with OpenCV. Image embeddings come from
OpenCLIP ViT-B/32 (OpenAI weights) on CPU. Text amount is estimated with a
morphological text-region detector because a system Tesseract package was not
available; CLIP text-overlay similarity is a second, independent signal.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

from src.config import HAAR_FACE_PATH, ROOT, THUMB_DIR, YUNET_FACE_PATH

CLIP_PROMPTS = {
    "clip_text_overlay": "a YouTube thumbnail with large bold text captions",
    "clip_face_closeup": "a close-up photo of a person's face looking at the camera",
    "clip_phone_screen": "a screenshot of a smartphone settings screen",
    "clip_code_screen": "computer code on a monitor",
    "clip_product_shot": "a product photo of a gadget or electronic device",
    "clip_dark_blurry": "a dark blurry low-quality video frame",
    "clip_bright_bold": "a bright high-contrast colorful thumbnail",
}

INTERPRETABLE_FEATURES = [
    "thumb_brightness",
    "thumb_contrast",
    "thumb_colorfulness",
    "thumb_saturation_mean",
    "thumb_saturation_std",
    "thumb_letterbox_frac",
    "thumb_is_letterboxed",
    "thumb_content_aspect",
    "thumb_edge_density",
    "thumb_text_present",
    "thumb_text_area_ratio",
    "thumb_text_region_count",
    "thumb_text_top_frac",
    "thumb_face_detected",
    "thumb_face_count",
    "thumb_face_area_ratio",
    "thumb_dark_frac",
    "thumb_bright_frac",
    "thumb_warm_frac",
    "thumb_dom_h",
    "thumb_dom_s",
    "thumb_dom_v",
    "thumb_dom_frac",
    "thumb_width",
    "thumb_height",
    *CLIP_PROMPTS.keys(),
]

CLIP_EMBED_DIM = 512

AMAZON_HOST_PARTS = (
    "amazon.",
    "amzn.",
    "amzn.to",
    "amazon.co",
    "smile.amazon",
)


def is_amazon_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(part in host for part in AMAZON_HOST_PARTS)


def is_allowed_thumb_url(url: str) -> bool:
    """Allow local-style http(s) image URLs except Amazon/affiliate hosts."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    if is_amazon_url(url):
        return False
    return True


def _load_bgr(path: str | Path) -> np.ndarray | None:
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        try:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return img


def _letterbox_masks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Detect near-black, low-variance bars from the edges inward."""
    h, w = gray.shape
    row_mean, row_std = gray.mean(axis=1), gray.std(axis=1)
    col_mean, col_std = gray.mean(axis=0), gray.std(axis=0)
    row_bar = (row_mean < 22) & (row_std < 10)
    col_bar = (col_mean < 22) & (col_std < 10)

    def edge_run(mask: np.ndarray) -> np.ndarray:
        out = np.zeros_like(mask)
        n = len(mask)
        i = 0
        while i < n and mask[i]:
            out[i] = True
            i += 1
        i = n - 1
        while i >= 0 and mask[i]:
            out[i] = True
            i -= 1
        return out

    row_lb = edge_run(row_bar)
    col_lb = edge_run(col_bar)
    frac = float((row_lb.sum() * w + col_lb.sum() * h - row_lb.sum() * col_lb.sum()) / max(h * w, 1))
    return row_lb, col_lb, frac


def _colorfulness(bgr: np.ndarray) -> float:
    """Hasler and Süsstrunk (2003) colorfulness metric, scaled roughly 0-100+."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _text_regions(gray: np.ndarray) -> tuple[int, float, float]:
    """Morphological text-like region detector (not full OCR).

    Returns (region_count, area_ratio, fraction_of_text_area_in_top_third).
    """
    h, w = gray.shape
    # Horizontal-ish kernels catch caption lines; a slightly taller one catches stacked type.
    results = []
    for kw, kh in ((13, 3), (9, 5), (17, 3)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        enh = cv2.add(blackhat, tophat)
        results.append(enh)
    enh = np.maximum.reduce(results)
    enh = cv2.GaussianBlur(enh, (3, 3), 0)
    # Adaptive-ish: keep strong high-pass responses.
    thresh_val = max(18.0, float(np.percentile(enh, 88)))
    _, th = cv2.threshold(enh, thresh_val, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2)))
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(th.astype(np.uint8))
    count = 0
    area = 0
    top_area = 0
    top_cut = h / 3.0
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        if a < 25 or bh < 6 or bw < 8:
            continue
        ar = bw / max(bh, 1)
        if bh > 0.5 * h or bw > 0.98 * w:
            continue
        if not (1.15 <= ar <= 20 and 6 <= bh <= 0.4 * h):
            continue
        count += 1
        area += int(a)
        # Count overlap with the top third.
        y2 = y + bh
        overlap = max(0.0, min(y2, top_cut) - y)
        if overlap > 0:
            top_area += int(a * (overlap / max(bh, 1)))
    area_ratio = float(area / max(h * w, 1))
    top_frac = float(top_area / area) if area else 0.0
    return count, area_ratio, top_frac


def _faces(bgr: np.ndarray, detector: "cv2.FaceDetectorYN | None") -> tuple[int, float]:
    if detector is None:
        return 0, 0.0
    h, w = bgr.shape[:2]
    try:
        detector.setInputSize((w, h))
        _retval, faces = detector.detect(bgr)
    except Exception:
        return 0, 0.0
    if faces is None or len(faces) == 0:
        return 0, 0.0
    # YuNet rows: x, y, w, h, landmarks..., score
    area = 0.0
    n = 0
    for row in faces:
        fw, fh = float(row[2]), float(row[3])
        if fw <= 0 or fh <= 0:
            continue
        n += 1
        area += fw * fh
    return n, float(area / max(h * w, 1))


def _dominant_hsv(bgr: np.ndarray, k: int = 3) -> tuple[float, float, float, float]:
    small = cv2.resize(bgr, (48, 48), interpolation=cv2.INTER_AREA)
    z = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _ret, labels, centers = cv2.kmeans(z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    main = int(np.argmax(counts))
    bgr_c = centers[main].reshape(1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(bgr_c, cv2.COLOR_BGR2HSV)[0, 0]
    frac = float(counts[main] / max(counts.sum(), 1))
    return float(hsv[0] / 179.0), float(hsv[1] / 255.0), float(hsv[2] / 255.0), frac


_YUNET: cv2.FaceDetectorYN | None | bool = None


def get_face_detector() -> "cv2.FaceDetectorYN | None":
    global _YUNET
    if _YUNET is False:
        return None
    if _YUNET is not None:
        return _YUNET
    path = YUNET_FACE_PATH
    if not path.exists():
        _YUNET = False
        return None
    try:
        det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.6, 0.3, 5000)
    except Exception:
        _YUNET = False
        return None
    _YUNET = det
    return det


def interpretable_features(path: str | Path) -> dict[str, float] | None:
    bgr = _load_bgr(path)
    if bgr is None or bgr.size == 0:
        return None
    h0, w0 = bgr.shape[:2]
    if h0 < 8 or w0 < 8:
        return None
    # Downscale very large vertical previews so CV features stay fast; ratios are scale-invariant.
    max_side = 854
    scale = min(1.0, max_side / max(h0, w0))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (max(8, int(w0 * scale)), max(8, int(h0 * scale))), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Crop letterbox before some stats so contrast isn't crushed by black bars.
    row_lb, col_lb, lb_frac = _letterbox_masks(gray)
    ys = np.where(~row_lb)[0]
    xs = np.where(~col_lb)[0]
    if len(ys) >= 8 and len(xs) >= 8:
        content = bgr[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]
        content_gray = gray[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]
        content_hsv = hsv[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]
        ch, cw = content_gray.shape
        content_aspect = float(ch / max(cw, 1))
    else:
        content, content_gray, content_hsv = bgr, gray, hsv
        ch, cw = h, w
        content_aspect = float(h / max(w, 1))

    brightness = float(content_gray.mean() / 255.0)
    contrast = float(content_gray.std() / 255.0)
    sat = content_hsv[:, :, 1] / 255.0
    val = content_hsv[:, :, 2] / 255.0
    hue = content_hsv[:, :, 0]  # 0-179
    # Warm = reds/oranges/yellows (H near 0-30 or 160-179).
    warm = ((hue <= 30) | (hue >= 160)) & (content_hsv[:, :, 1] > 40) & (content_hsv[:, :, 2] > 40)
    edges = cv2.Canny(content_gray, 80, 160)
    text_count, text_area, text_top = _text_regions(content_gray)
    face_count, face_area = _faces(content, get_face_detector())
    dom_h, dom_s, dom_v, dom_frac = _dominant_hsv(content)

    feats = {
        "thumb_brightness": brightness,
        "thumb_contrast": contrast,
        "thumb_colorfulness": _colorfulness(content),
        "thumb_saturation_mean": float(sat.mean()),
        "thumb_saturation_std": float(sat.std()),
        "thumb_letterbox_frac": float(lb_frac),
        "thumb_is_letterboxed": float(lb_frac >= 0.12),
        "thumb_content_aspect": content_aspect,
        "thumb_edge_density": float((edges > 0).mean()),
        "thumb_text_present": float(text_count >= 2 or text_area >= 0.02),
        "thumb_text_area_ratio": float(text_area),
        "thumb_text_region_count": float(text_count),
        "thumb_text_top_frac": float(text_top),
        "thumb_face_detected": float(face_count > 0),
        "thumb_face_count": float(face_count),
        "thumb_face_area_ratio": float(face_area),
        "thumb_dark_frac": float((val < 0.15).mean()),
        "thumb_bright_frac": float((val > 0.85).mean()),
        "thumb_warm_frac": float(warm.mean()),
        "thumb_dom_h": float(dom_h),
        "thumb_dom_s": float(dom_s),
        "thumb_dom_v": float(dom_v),
        "thumb_dom_frac": float(dom_frac),
        "thumb_width": float(w0),
        "thumb_height": float(h0),
    }
    return feats


class ClipEncoder:
    """Lazy CPU CLIP ViT-B/32 encoder (image embeddings + prompt similarities)."""

    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.prompt_features = None
        self.prompt_names = list(CLIP_PROMPTS.keys())

    def load(self) -> None:
        if self.model is not None:
            return
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()
        model.to("cpu")
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            tokens = self.tokenizer([CLIP_PROMPTS[n] for n in self.prompt_names])
            text_feat = model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            self.prompt_features = text_feat

    def encode_paths(self, paths: list[Path | str], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """Return (N, 512) L2-normalized embeddings and (N, P) prompt cosine sims."""
        import torch

        self.load()
        embs = []
        sims = []
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            tensors = []
            keep_idx = []
            for j, p in enumerate(chunk):
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self.preprocess(img))
                    keep_idx.append(j)
                except Exception:
                    tensors.append(None)
            # Replace failures with zeros after the batch, but still run valid ones.
            valid = [(k, t) for k, t in zip(range(len(chunk)), tensors) if t is not None]
            out_emb = np.zeros((len(chunk), CLIP_EMBED_DIM), dtype=np.float32)
            out_sim = np.zeros((len(chunk), len(self.prompt_names)), dtype=np.float32)
            if valid:
                batch = torch.stack([t for _k, t in valid], dim=0)
                with torch.no_grad():
                    feat = self.model.encode_image(batch)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    sim = feat @ self.prompt_features.T
                    feat_np = feat.cpu().numpy().astype(np.float32)
                    sim_np = sim.cpu().numpy().astype(np.float32)
                for row, (k, _t) in enumerate(valid):
                    out_emb[k] = feat_np[row]
                    out_sim[k] = sim_np[row]
            embs.append(out_emb)
            sims.append(out_sim)
        return np.concatenate(embs, axis=0), np.concatenate(sims, axis=0)


def features_from_path(path: str | Path, encoder: ClipEncoder | None = None) -> tuple[dict[str, float], np.ndarray] | None:
    """Full feature vector for one local thumbnail image."""
    interp = interpretable_features(path)
    if interp is None:
        return None
    enc = encoder or ClipEncoder()
    emb, sim = enc.encode_paths([path], batch_size=1)
    for i, name in enumerate(enc.prompt_names):
        interp[name] = float(sim[0, i])
    return interp, emb[0]


def thumb_path_for_video(video_id: str, thumb_dir: Path | None = None) -> Path | None:
    d = Path(thumb_dir) if thumb_dir else THUMB_DIR
    p = d / f"{video_id}.jpg"
    if p.exists() and p.stat().st_size > 800:
        return p
    # occasional png/webp saves
    for ext in (".png", ".webp", ".jpeg"):
        q = d / f"{video_id}{ext}"
        if q.exists() and q.stat().st_size > 800:
            return q
    return None


def median_interpretable(frame) -> dict[str, float]:
    out = {}
    for col in INTERPRETABLE_FEATURES:
        if col in frame.columns and frame[col].notna().any():
            out[col] = float(frame[col].median())
        else:
            out[col] = 0.0
    return out
