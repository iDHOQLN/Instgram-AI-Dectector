"""
generate_datasets.py - Generate synthetic realistic datasets for training
Run this script once before training.
"""
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)
N = 2000  # samples per class

# ─────────────────────────────────────────────────────────────────────────────
# FAKE ACCOUNT DATASET
# ─────────────────────────────────────────────────────────────────────────────
def generate_fake_account_data():
    # --- REAL accounts ---
    real = pd.DataFrame({
        "username_length":        np.random.randint(5, 22, N),
        "full_name_length":       np.random.randint(5, 35, N),
        "followers":              np.random.randint(50, 50000, N),
        "following":              np.random.randint(50, 2000, N),
        "posts":                  np.random.randint(5, 500, N),
        "bio_length":             np.random.randint(10, 150, N),
        "has_external_url":       np.random.choice([0, 1], N, p=[0.55, 0.45]),
        "has_profile_pic":        np.random.choice([0, 1], N, p=[0.05, 0.95]),
        "is_private":             np.random.choice([0, 1], N, p=[0.45, 0.55]),
        "is_verified":            np.random.choice([0, 1], N, p=[0.97, 0.03]),
        "engagement_rate":        np.random.uniform(0.01, 0.12, N),
        "label":                  0   # REAL
    })
    real["follower_following_ratio"] = real["followers"] / (real["following"] + 1)
    real["posts_per_follower"] = real["posts"] / (real["followers"] + 1)

    # --- FAKE accounts ---
    fake = pd.DataFrame({
        "username_length":        np.random.randint(15, 30, N),   # random long names
        "full_name_length":       np.random.randint(0, 10, N),    # short/empty
        "followers":              np.random.randint(0, 300, N),   # very few
        "following":              np.random.randint(500, 7500, N),# follow many
        "posts":                  np.random.randint(0, 20, N),    # few posts
        "bio_length":             np.random.randint(0, 30, N),    # empty/short bio
        "has_external_url":       np.random.choice([0, 1], N, p=[0.7, 0.3]),
        "has_profile_pic":        np.random.choice([0, 1], N, p=[0.60, 0.40]),
        "is_private":             np.random.choice([0, 1], N, p=[0.8, 0.2]),
        "is_verified":            np.zeros(N, dtype=int),
        "engagement_rate":        np.random.uniform(0.0, 0.02, N),
        "label":                  1   # FAKE
    })
    fake["follower_following_ratio"] = fake["followers"] / (fake["following"] + 1)
    fake["posts_per_follower"] = fake["posts"] / (fake["followers"] + 1)

    df = pd.concat([real, fake], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BOT DATASET
# ─────────────────────────────────────────────────────────────────────────────
def generate_bot_data():
    # --- HUMAN accounts ---
    human = pd.DataFrame({
        "posting_frequency":         np.random.uniform(0.1, 3.0, N),    # posts/day
        "avg_likes":                 np.random.randint(10, 2000, N),
        "avg_comments":              np.random.randint(1, 200, N),
        "followers":                 np.random.randint(50, 30000, N),
        "following":                 np.random.randint(50, 2000, N),
        "avg_engagement":            np.random.uniform(0.01, 0.15, N),
        "duplicate_comments_ratio":  np.random.uniform(0.0, 0.15, N),
        "active_hours_count":        np.random.randint(4, 18, N),
        "account_age_days":          np.random.randint(180, 3650, N),
        "label":                     0  # HUMAN
    })
    human["like_comment_ratio"] = human["avg_likes"] / (human["avg_comments"] + 1)
    human["following_follower_ratio"] = human["following"] / (human["followers"] + 1)

    # --- BOT accounts ---
    bot = pd.DataFrame({
        "posting_frequency":         np.random.uniform(10, 80, N),       # very high
        "avg_likes":                 np.random.randint(0, 50, N),        # very low
        "avg_comments":              np.random.randint(0, 10, N),
        "followers":                 np.random.randint(0, 500, N),
        "following":                 np.random.randint(1000, 8000, N),   # follows tons
        "avg_engagement":            np.random.uniform(0.0, 0.01, N),
        "duplicate_comments_ratio":  np.random.uniform(0.4, 1.0, N),    # high duplicates
        "active_hours_count":        np.random.randint(20, 24, N),       # always on
        "account_age_days":          np.random.randint(1, 180, N),       # new accounts
        "label":                     1  # BOT
    })
    bot["like_comment_ratio"] = bot["avg_likes"] / (bot["avg_comments"] + 1)
    bot["following_follower_ratio"] = bot["following"] / (bot["followers"] + 1)

    df = pd.concat([human, bot], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    print("Generating fake_accounts.csv ...")
    fa = generate_fake_account_data()
    fa.to_csv(os.path.join(data_dir, "fake_accounts.csv"), index=False)
    print(f"  ✔  {len(fa)} rows  |  Real: {(fa.label==0).sum()}  Fake: {(fa.label==1).sum()}")

    print("Generating bot_dataset.csv ...")
    bot = generate_bot_data()
    bot.to_csv(os.path.join(data_dir, "bot_dataset.csv"), index=False)
    print(f"  ✔  {len(bot)} rows  |  Human: {(bot.label==0).sum()}  Bot: {(bot.label==1).sum()}")

    print("\nDatasets generated successfully!")
