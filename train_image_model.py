"""
train_image_model.py
====================
Train a Fake Profile Image detector using Transfer Learning.
Architecture: MobileNetV2 (pretrained on ImageNet) + custom dense head.

Classes: 0=Real  1=AI-Generated  2=Edited

Since we don't have a labelled real-world image dataset, this script:
  1. Generates synthetic training data from solid-colour noise images
     (just enough to create a working .keras model file).
  2. In a real project, replace the data generator with actual labeled images.

Saves: model/image_detector.keras
"""

import os, sys, warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import IMAGE_SIZE, IMAGE_DETECTOR_MODEL, MODEL_DIR


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_images(n_per_class: int = 300, img_size: tuple = (224, 224)):
    """
    Generate synthetic training images with distinct visual signatures per class.
    Real      → natural-looking noise + warm tones
    AI-Gen    → uniform pastel gradients
    Edited    → sharp pattern with artefacts
    """
    np.random.seed(42)
    X, y = [], []
    H, W = img_size

    for _ in range(n_per_class):
        # Class 0 — Real: natural noise image
        img = np.random.randint(60, 200, (H, W, 3), dtype=np.uint8)
        img[:, :, 0] = np.clip(img[:, :, 0] + 30, 0, 255)   # warm tint
        X.append(img); y.append(0)

    for _ in range(n_per_class):
        # Class 1 — AI Generated: smooth gradient
        base = np.linspace(100, 220, W, dtype=np.float32)
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :, 0] = np.tile(base, (H, 1)).astype(np.uint8)
        img[:, :, 1] = np.tile(np.linspace(80, 180, H), (W, 1)).T.astype(np.uint8)
        img[:, :, 2] = 180
        noise = np.random.randint(0, 15, (H, W, 3), dtype=np.uint8)
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        X.append(img); y.append(1)

    for _ in range(n_per_class):
        # Class 2 — Edited: checkerboard artefacts
        img = np.random.randint(100, 200, (H, W, 3), dtype=np.uint8)
        # inject block artefacts
        block = 16
        for bi in range(0, H, block * 2):
            for bj in range(0, W, block * 2):
                img[bi:bi+block, bj:bj+block, :] = np.clip(
                    img[bi:bi+block, bj:bj+block, :].astype(int) + 60, 0, 255
                ).astype(np.uint8)
        X.append(img); y.append(2)

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = 3) -> tf.keras.Model:
    """MobileNetV2 base + custom classification head."""
    base = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    # Freeze all base layers first
    base.trainable = False

    inp = layers.Input(shape=(*IMAGE_SIZE, 3), name="image_input")
    x = base(inp, training=False)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inp, out, name="FakeImageDetector")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  FAKE IMAGE DETECTION — MODEL TRAINING")
    print("=" * 60)

    # GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✔  GPU detected: {gpus}")
    else:
        print("  Running on CPU")

    print("\nGenerating synthetic training images ...")
    X, y = generate_synthetic_images(n_per_class=300)
    print(f"  Images: {X.shape}  Labels: {np.bincount(y)}")

    split = int(len(X) * 0.80)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # One-hot encode
    y_train_oh = tf.keras.utils.to_categorical(y_train, 3)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   3)

    # ── Build ────────────────────────────────────────────────────────────────
    model = build_model(num_classes=3)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # ── Phase 1: train head only ──────────────────────────────────────────────
    print("\nPhase 1: Training classification head (base frozen) ...")
    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6),
    ]
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=20,
        batch_size=32,
        callbacks=cb,
        verbose=1
    )

    # ── Phase 2: fine-tune top layers ────────────────────────────────────────
    print("\nPhase 2: Fine-tuning top 30 layers of MobileNetV2 ...")
    base_model = model.layers[1]  # MobileNetV2
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=10,
        batch_size=32,
        callbacks=cb,
        verbose=1
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    loss, acc = model.evaluate(X_val, y_val_oh, verbose=0)
    print(f"\nValidation  Loss: {loss:.4f}  Accuracy: {acc:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(IMAGE_DETECTOR_MODEL)
    print(f"✔  Image model saved → {IMAGE_DETECTOR_MODEL}")
    print("\n✅  Image Detection model training complete!")


if __name__ == "__main__":
    train()
