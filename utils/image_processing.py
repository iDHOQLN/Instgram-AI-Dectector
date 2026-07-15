"""
utils/image_processing.py - Image preprocessing for the CNN model
"""
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import IMAGE_SIZE


def load_image_from_upload(uploaded_file) -> np.ndarray:
    """
    Load a Streamlit UploadedFile into a numpy array.
    Returns RGB uint8 array (H, W, 3).
    """
    img_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(pil_img)


def preprocess_for_model(img_array: np.ndarray) -> np.ndarray:
    """
    Resize, normalize and expand dims for CNN inference.
    Returns float32 array of shape (1, H, W, 3) in [0,1].
    """
    img = cv2.resize(img_array, IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def extract_image_features(img_array: np.ndarray) -> dict:
    """
    Extract simple hand-crafted features for analysis display.
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Laplacian variance (blur detection)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Average brightness
    brightness = float(np.mean(gray))

    # Colour saturation
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))

    # Noise estimation via standard deviation of residuals
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = float(np.std(gray.astype(int) - blurred.astype(int)))

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / edges.size)

    return {
        "sharpness": round(lap_var, 2),
        "brightness": round(brightness, 2),
        "saturation": round(saturation, 2),
        "noise_level": round(noise, 2),
        "edge_density": round(edge_density * 100, 2),
    }


def pil_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()
