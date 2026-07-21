"""
config.py - Application-wide configuration constants
"""
import os

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
HISTORY_DIR = os.path.join(BASE_DIR, "history")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ─────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────
FAKE_ACCOUNT_CSV = os.path.join(DATA_DIR, "fake_accounts.csv")
BOT_DATASET_CSV = os.path.join(DATA_DIR, "bot_dataset.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

# ─────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────
FAKE_ACCOUNT_MODEL = os.path.join(MODEL_DIR, "fake_account_model.pkl")
BOT_DETECTOR_MODEL = os.path.join(MODEL_DIR, "bot_detector.pkl")
IMAGE_DETECTOR_MODEL = os.path.join(MODEL_DIR, "image_detector.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
BOT_SCALER_PATH = os.path.join(MODEL_DIR, "bot_scaler.pkl")

# ─────────────────────────────────────────────
# HISTORY / REPORT PATHS
# ─────────────────────────────────────────────
HISTORY_CSV = os.path.join(HISTORY_DIR, "prediction_history.csv")
REPORT_PDF = os.path.join(REPORTS_DIR, "report.pdf")
REPORT_CSV = os.path.join(REPORTS_DIR, "report.csv")

# ─────────────────────────────────────────────
# MODEL THRESHOLDS
# ─────────────────────────────────────────────
FAKE_THRESHOLD = 0.50
BOT_THRESHOLD = 0.50
IMAGE_THRESHOLD = 0.50

# ─────────────────────────────────────────────
# APP SETTINGS
# ─────────────────────────────────────────────
APP_TITLE = "Instagram AI Detector"
APP_ICON = "🔍"
VERSION = "1.0.1"

# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
USERS = {
    "admin": "admin123",
    "demo": "demo123",
    "analyst": "analyst@123",
}

# ─────────────────────────────────────────────
# IMAGE SETTINGS
# ─────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
IMAGE_CLASSES = ["Real", "AI Generated", "Edited"]
MAX_UPLOAD_MB = 10

# ─────────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────────
FAKE_ACCOUNT_FEATURES = [
    "username_length", "full_name_length", "followers", "following",
    "posts", "bio_length", "has_external_url", "has_profile_pic",
    "is_private", "is_verified", "engagement_rate",
    "follower_following_ratio", "posts_per_follower"
]

BOT_FEATURES = [
    "posting_frequency", "avg_likes", "avg_comments", "followers",
    "following", "avg_engagement", "duplicate_comments_ratio",
    "active_hours_count", "account_age_days", "like_comment_ratio",
    "following_follower_ratio"
]
