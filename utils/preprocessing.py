"""
utils/preprocessing.py - Feature engineering and data scaling utilities
"""
import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SCALER_PATH, BOT_SCALER_PATH, FAKE_ACCOUNT_FEATURES, BOT_FEATURES


def engineer_fake_account_features(raw: dict) -> pd.DataFrame:
    """
    Derive computed features from raw user inputs for the fake-account model.
    raw keys: username_length, full_name_length, followers, following, posts,
              bio_length, has_external_url, has_profile_pic, is_private,
              is_verified, engagement_rate
    """
    followers = max(raw.get("followers", 0), 0)
    following = max(raw.get("following", 0), 0)
    posts = max(raw.get("posts", 0), 0)

    raw["follower_following_ratio"] = followers / (following + 1)
    raw["posts_per_follower"] = posts / (followers + 1)

    df = pd.DataFrame([raw])[FAKE_ACCOUNT_FEATURES]
    return df


def scale_fake_features(df: pd.DataFrame) -> np.ndarray:
    """Load saved scaler and transform fake-account feature dataframe."""
    scaler = joblib.load(SCALER_PATH)
    return scaler.transform(df)


def engineer_bot_features(raw: dict) -> pd.DataFrame:
    """
    Derive computed features for the bot-detection model.
    raw keys: posting_frequency, avg_likes, avg_comments, followers,
              following, avg_engagement, duplicate_comments_ratio,
              active_hours_count, account_age_days
    """
    raw["like_comment_ratio"] = raw.get("avg_likes", 0) / (raw.get("avg_comments", 0) + 1)
    raw["following_follower_ratio"] = raw.get("following", 0) / (raw.get("followers", 0) + 1)

    df = pd.DataFrame([raw])[BOT_FEATURES]
    return df


def scale_bot_features(df: pd.DataFrame) -> np.ndarray:
    """Load saved bot scaler and transform bot feature dataframe."""
    scaler = joblib.load(BOT_SCALER_PATH)
    return scaler.transform(df)
