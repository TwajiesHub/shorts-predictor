"""Paths and collection defaults for the post predictor."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

CHANNEL_CANDIDATES_PATH = DATA_DIR / "channel_candidates.csv"
CHANNELS_PATH = DATA_DIR / "channels.csv"
VIDEOS_PATH = DATA_DIR / "videos.parquet"
VIDEOS_CSV_PATH = DATA_DIR / "videos.csv"
MODEL_PATH = MODELS_DIR / "model_v1.joblib"  # v1 (alias of MODEL_V1_PATH)
METRICS_PATH = MODELS_DIR / "metrics.json"
TRAIN_SUMMARY_PATH = REPORTS_DIR / "train_summary.json"

MODEL_V1_PATH = MODELS_DIR / "model_v1.joblib"
METRICS_V1_PATH = MODELS_DIR / "metrics_v1.json"
MODEL_V2_PATH = MODELS_DIR / "model_v2.joblib"
METRICS_V2_PATH = MODELS_DIR / "metrics_v2.json"

THUMB_DIR = DATA_DIR / "thumbnails"
THUMB_FEATURES_PATH = DATA_DIR / "thumb_features.parquet"
THUMB_EMBEDDINGS_PATH = DATA_DIR / "thumb_embeddings.npz"
THUMB_MANIFEST_PATH = DATA_DIR / "thumbnails_manifest.csv"
HAAR_FACE_PATH = MODELS_DIR / "cv" / "haarcascade_frontalface_default.xml"
YUNET_FACE_PATH = MODELS_DIR / "cv" / "face_detection_yunet_2023mar.onnx"

TESTRUN_HANDLE = "@testru7"
TESTRUN_CHANNEL_ID = "UCExmWKJ2B0tAC0FT3c0YHeg"
TESTRUN_NAME = "TestRun"

# Seed comparable channels. mavigadgets is a reach ceiling only.
SEED_CHANNELS = [
    {"handle": "@testru7", "role": "target", "note": "User channel (TestRun)"},
    {"handle": "@zackdfilms", "role": "comparable", "note": "Faceless curiosity explainers"},
    {"handle": "@athome-y1o", "role": "comparable", "note": "Faceless iPhone screen-recording tips"},
    {"handle": "@FixWindowsNow", "role": "comparable", "note": "Faceless tech tutorials"},
    {"handle": "@phonerepairguru", "role": "comparable", "note": "Phone repair / Apple curiosity Shorts"},
    {"handle": "@mavigadgets", "role": "ceiling", "note": "Reach ceiling only; high-volume gadget demos"},
    {"handle": "@mavigadgetoriginals", "role": "comparable", "note": "Smaller hands-only gadget channel"},
]

# Channel names that may still need a public handle resolution.
RESOLVE_NAMES = [
    "Arya Vale",
    "Speed McQueen",
    "Yusuf Tech Talks",
    "Keebo100",
    "DATARK",
]

SEARCH_QUERIES = [
    "iphone back tap #shorts",
    "hidden iphone feature #shorts",
    "iphone secret button #shorts",
    "iphone keyboard trackpad #shorts",
    "android hidden feature #shorts",
    "samsung secret setting #shorts",
    "why 1TB shows 931GB #shorts",
    "why 0.1 + 0.2 != 0.3 #shorts",
    "python coding fact #shorts",
    "javascript weird #shorts",
    "windows tip CON folder #shorts",
    "excel leap year 1900 #shorts",
    "excel zip code leading zero #shorts",
    "computer fact you didn't know #shorts",
    "why is bluetooth named #shorts",
    "incognito mode not private #shorts",
    "phone charger 60W #shorts",
    "usb-c hub review #shorts",
    "coding secrets python #shorts",
    "tech numbers that lie #shorts",
    "your computer can't add #shorts",
    "fireship python shorts",
    "yusuf tech talks phone OS",
    "arya vale split keyboard",
    "zack d films how it works",
]

# Title/topic keywords used both to rank comparable channels and as model features.
NICHE_KEYWORDS = [
    "iphone", "android", "samsung", "ios", "pixel", "phone", "smartphone",
    "python", "javascript", "java", "coding", "code", "programmer", "programming",
    "windows", "excel", "chrome", "google", "apple", "mac", "linux",
    "computer", "laptop", "pc", "wifi", "wi-fi", "bluetooth", "usb", "charger",
    "battery", "storage", "gb", "tb", "4k", "incognito", "privacy", "password",
    "hidden", "secret", "feature", "setting", "tip", "trick", "why", "how",
    "tech", "gadget", "shortcut", "hack",
]

# Caps so mega-channels do not drown the dataset or the request budget.
DEFAULT_CAP = 350
CAPS_BY_ROLE = {
    "target": 10_000,
    "comparable": 350,
    "ceiling": 200,
    "discovered": 300,
}

MIN_AGE_DAYS = 3.0
SHORT_MAX_SECONDS = 60
SHORTS_ELIGIBLE_MAX_SECONDS = 180
RANDOM_SEED = 42
