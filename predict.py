"""
predict.py
==========
Unified prediction API used by all Streamlit pages.
Handles lazy model loading with caching.
"""

import os, sys, joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    FAKE_ACCOUNT_MODEL, BOT_DETECTOR_MODEL, IMAGE_DETECTOR_MODEL,
    SCALER_PATH, BOT_SCALER_PATH, IMAGE_CLASSES, IMAGE_SIZE,
    FAKE_ACCOUNT_FEATURES, BOT_FEATURES
)
from utils.preprocessing import engineer_fake_account_features, engineer_bot_features


# ─────────────────────────────────────────────────────────────────────────────
# LAZY MODEL CACHE
# ─────────────────────────────────────────────────────────────────────────────
_cache: dict = {}


def _load(key: str, path: str, loader):
    if key not in _cache:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Run the corresponding training script first."
            )
        _cache[key] = loader(path)
    return _cache[key]


def _get_fake_model():
    return _load("fake_model", FAKE_ACCOUNT_MODEL, joblib.load)

def _get_fake_scaler():
    return _load("fake_scaler", SCALER_PATH, joblib.load)

def _get_bot_model():
    return _load("bot_model", BOT_DETECTOR_MODEL, joblib.load)

def _get_bot_scaler():
    return _load("bot_scaler", BOT_SCALER_PATH, joblib.load)

def _get_image_model():
    import tensorflow as tf
    return _load("image_model", IMAGE_DETECTOR_MODEL, tf.keras.models.load_model)


# ─────────────────────────────────────────────────────────────────────────────
# FAKE ACCOUNT PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_fake_account(raw: dict) -> dict:
    """
    Predict whether an Instagram account is Real or Fake.

    Parameters
    ----------
    raw : dict with keys matching FAKE_ACCOUNT_FEATURES (pre-derived)
          plus followers, following, posts for ratio computation.

    Returns
    -------
    dict: {label, confidence, prob_fake, prob_real, features}
    """
    model  = _get_fake_model()
    scaler = _get_fake_scaler()

    feat_df = engineer_fake_account_features(raw)
    X_scaled = scaler.transform(feat_df)

    proba = model.predict_proba(X_scaled)[0]
    prob_real, prob_fake = float(proba[0]), float(proba[1])
    label = "Fake" if prob_fake >= 0.50 else "Real"
    confidence = prob_fake if label == "Fake" else prob_real

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "prob_fake": round(prob_fake, 4),
        "prob_real": round(prob_real, 4),
        "features": feat_df.to_dict(orient="records")[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# BOT DETECTION PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_bot(raw: dict) -> dict:
    """
    Predict whether an account is a Bot or Human.

    Parameters
    ----------
    raw : dict with keys matching BOT_FEATURES (pre-derived).

    Returns
    -------
    dict: {label, confidence, prob_bot, prob_human, features}
    """
    model  = _get_bot_model()
    scaler = _get_bot_scaler()

    feat_df = engineer_bot_features(raw)
    X_scaled = scaler.transform(feat_df)

    proba = model.predict_proba(X_scaled)[0]
    prob_human, prob_bot = float(proba[0]), float(proba[1])
    label = "Bot" if prob_bot >= 0.50 else "Human"
    confidence = prob_bot if label == "Bot" else prob_human

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "prob_bot": round(prob_bot, 4),
        "prob_human": round(prob_human, 4),
        "features": feat_df.to_dict(orient="records")[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_image(img_array: np.ndarray) -> dict:
    """
    Predict whether a profile image is Real, AI-Generated, or Edited.

    Parameters
    ----------
    img_array : np.ndarray  RGB uint8 (H, W, 3)

    Returns
    -------
    dict: {label, confidence, class_probs}
    """
    import cv2
    model = _get_image_model()

    # Preprocess
    img = cv2.resize(img_array, IMAGE_SIZE).astype(np.float32) / 255.0
    X   = np.expand_dims(img, axis=0)

    proba = model.predict(X, verbose=0)[0]
    class_idx = int(np.argmax(proba))
    label = IMAGE_CLASSES[class_idx]
    confidence = float(proba[class_idx])

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "class_probs": {cls: round(float(p), 4) for cls, p in zip(IMAGE_CLASSES, proba)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL STATUS CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_models_exist() -> dict:
    """Return dict of model availability."""
    return {
        "fake_account": os.path.exists(FAKE_ACCOUNT_MODEL),
        "bot_detector": os.path.exists(BOT_DETECTOR_MODEL),
        "image_detector": os.path.exists(IMAGE_DETECTOR_MODEL),
    }
