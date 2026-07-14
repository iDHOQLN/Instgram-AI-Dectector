"""
train_fake_account.py
=====================
Train a Fake Account Detection classifier using:
  • Logistic Regression
  • Random Forest (primary)
  • XGBoost (optional)

Saves: model/fake_account_model.pkl
       model/scaler.pkl
"""

import os, sys, joblib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, f1_score)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import (
    FAKE_ACCOUNT_CSV, FAKE_ACCOUNT_MODEL, SCALER_PATH,
    MODEL_DIR, FAKE_ACCOUNT_FEATURES
)
from generate_datasets import generate_fake_account_data


def load_or_generate_data():
    if os.path.exists(FAKE_ACCOUNT_CSV):
        print(f"Loading data from {FAKE_ACCOUNT_CSV} ...")
        df = pd.read_csv(FAKE_ACCOUNT_CSV)
    else:
        print("CSV not found – generating synthetic data ...")
        df = generate_fake_account_data()
        os.makedirs(os.path.dirname(FAKE_ACCOUNT_CSV), exist_ok=True)
        df.to_csv(FAKE_ACCOUNT_CSV, index=False)
    return df


def train():
    print("=" * 60)
    print("  FAKE ACCOUNT DETECTION — MODEL TRAINING")
    print("=" * 60)

    df = load_or_generate_data()
    print(f"\nDataset: {len(df)} rows | Real: {(df.label==0).sum()} | Fake: {(df.label==1).sum()}")

    X = df[FAKE_ACCOUNT_FEATURES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Scaling ──────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Models ───────────────────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=12,
                                                       min_samples_leaf=2, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                           max_depth=5, random_state=42),
    }

    best_model, best_auc, best_name = None, 0, ""
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1  = f1_score(y_test, y_pred)
        cv  = cross_val_score(model, X_train_s, y_train, cv=StratifiedKFold(5), scoring="roc_auc").mean()

        results[name] = {"accuracy": acc, "auc": auc, "f1": f1, "cv_auc": cv}
        print(f"  Accuracy: {acc:.4f}  |  ROC-AUC: {auc:.4f}  |  F1: {f1:.4f}  |  CV-AUC: {cv:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

        if auc > best_auc:
            best_auc, best_model, best_name = auc, model, name

    print(f"\n✔  Best model: {best_name}  (ROC-AUC={best_auc:.4f})")

    # ── Feature Importance ───────────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        fi_df = pd.DataFrame({"feature": FAKE_ACCOUNT_FEATURES, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=False)
        print("\nTop Feature Importances:")
        print(fi_df.to_string(index=False))

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, FAKE_ACCOUNT_MODEL)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n✔  Model saved  → {FAKE_ACCOUNT_MODEL}")
    print(f"✔  Scaler saved → {SCALER_PATH}")

    # ── Confusion Matrix plot ────────────────────────────────────────────────
    y_pred_best = best_model.predict(X_test_s)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"]); ax.set_yticklabels(["Real", "Fake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "fake_account_cm.png")
    plt.savefig(cm_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"✔  Confusion matrix saved → {cm_path}")

    print("\n✅  Fake Account model training complete!")
    return results


if __name__ == "__main__":
    train()
