"""
train_bot_model.py
==================
Train a Bot Detection classifier using:
  • Random Forest (primary)
  • Isolation Forest (anomaly detection baseline)

Saves: model/bot_detector.pkl
       model/bot_scaler.pkl
"""

import os, sys, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, f1_score)

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    BOT_DATASET_CSV, BOT_DETECTOR_MODEL, BOT_SCALER_PATH,
    MODEL_DIR, BOT_FEATURES
)
from generate_datasets import generate_bot_data


def load_or_generate_data():
    if os.path.exists(BOT_DATASET_CSV):
        print(f"Loading data from {BOT_DATASET_CSV} ...")
        df = pd.read_csv(BOT_DATASET_CSV)
    else:
        print("CSV not found – generating synthetic data ...")
        df = generate_bot_data()
        os.makedirs(os.path.dirname(BOT_DATASET_CSV), exist_ok=True)
        df.to_csv(BOT_DATASET_CSV, index=False)
    return df


def train():
    print("=" * 60)
    print("  BOT DETECTION — MODEL TRAINING")
    print("=" * 60)

    df = load_or_generate_data()
    print(f"\nDataset: {len(df)} rows | Human: {(df.label==0).sum()} | Bot: {(df.label==1).sum()}")

    X = df[BOT_FEATURES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Scaling ───────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Supervised Models ─────────────────────────────────────────────────────
    models = {
        "Random Forest":     RandomForestClassifier(n_estimators=200, max_depth=15,
                                                     min_samples_leaf=2, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                         max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, random_state=42),
    }

    best_model, best_auc, best_name = None, 0, ""

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1  = f1_score(y_test, y_pred)
        cv  = cross_val_score(model, X_train_s, y_train, cv=5, scoring="roc_auc").mean()

        print(f"  Accuracy: {acc:.4f}  |  ROC-AUC: {auc:.4f}  |  F1: {f1:.4f}  |  CV-AUC: {cv:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Human", "Bot"]))

        if auc > best_auc:
            best_auc, best_model, best_name = auc, model, name

    # ── Isolation Forest (anomaly approach) ────────────────────────────────────
    print("\nTraining Isolation Forest (anomaly detection) ...")
    iso = IsolationForest(n_estimators=200, contamination=0.5, random_state=42)
    iso.fit(X_train_s)
    iso_pred_raw = iso.predict(X_test_s)
    iso_pred = np.where(iso_pred_raw == -1, 1, 0)   # -1 = anomaly = bot
    iso_acc = accuracy_score(y_test, iso_pred)
    iso_f1  = f1_score(y_test, iso_pred)
    print(f"  Accuracy: {iso_acc:.4f}  |  F1: {iso_f1:.4f}")

    print(f"\n✔  Best supervised model: {best_name}  (ROC-AUC={best_auc:.4f})")

    # ── Feature Importance ────────────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({"feature": BOT_FEATURES, "importance": best_model.feature_importances_})
        print("\nTop Feature Importances:")
        print(fi.sort_values("importance", ascending=False).to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, BOT_DETECTOR_MODEL)
    joblib.dump(scaler, BOT_SCALER_PATH)
    print(f"\n✔  Model saved  → {BOT_DETECTOR_MODEL}")
    print(f"✔  Scaler saved → {BOT_SCALER_PATH}")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    y_pred_best = best_model.predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Human", "Bot"]); ax.set_yticklabels(["Human", "Bot"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "bot_cm.png")
    plt.savefig(cm_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"✔  Confusion matrix saved → {cm_path}")

    print("\n✅  Bot Detection model training complete!")


if __name__ == "__main__":
    train()
