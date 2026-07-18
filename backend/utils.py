"""
backend/utils.py — Shared utility helpers used across the backend.

Responsibilities:
  - Base64 encode/decode for image transport over JSON
  - Timestamped filename generation
  - Saving output files for debugging / logging
"""

import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.config import OUTPUT_DIR


# ── Base64 helpers ─────────────────────────────────────────────────────────────

def encode_array_to_b64(image: np.ndarray, fmt: str = "PNG") -> str:
    """
    Encode a NumPy image array (BGR or BGRA or grayscale) to a base64 string.

    Args:
        image: NumPy array. For BGR/BGRA images OpenCV is used for encoding.
        fmt:   Target image format — "PNG" or "JPEG".

    Returns:
        Base64-encoded string (no data-URI prefix).
    """
    pil_img = _ndarray_to_pil(image, fmt)
    buffer = io.BytesIO()
    save_kwargs: dict = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs["quality"] = 92
    pil_img.save(buffer, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_b64_to_array(b64_string: str) -> np.ndarray:
    """
    Decode a base64 image string back to a NumPy BGR array.

    Args:
        b64_string: Base64-encoded image (no data-URI prefix needed).

    Returns:
        NumPy uint8 array in BGR channel order.
    """
    raw = base64.b64decode(b64_string)
    buf = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


# ── File I/O helpers ───────────────────────────────────────────────────────────

def get_timestamp() -> str:
    """Return a sortable timestamp string safe for filenames."""
    return time.strftime("%Y%m%d_%H%M%S")


def save_output(image: np.ndarray, prefix: str, fmt: str = "png") -> Path:
    """
    Write a NumPy image array to the outputs/ directory.

    Args:
        image:  NumPy array to save.
        prefix: Filename prefix (e.g. "mask", "overlay").
        fmt:    File extension / format — "png" or "jpg".

    Returns:
        Path to the saved file.
    """
    filename = OUTPUT_DIR / f"{prefix}_{get_timestamp()}.{fmt}"
    cv2.imwrite(str(filename), image)
    return filename


# ── Internal helpers ───────────────────────────────────────────────────────────

def _ndarray_to_pil(image: np.ndarray, fmt: str) -> Image.Image:
    """Convert a NumPy array to a PIL Image with correct channel handling."""
    if image.ndim == 2:
        # Grayscale
        return Image.fromarray(image.astype(np.uint8), mode="L")
    elif image.shape[2] == 4:
        # BGRA → RGBA
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")
    else:
        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mode = "RGB"
        if fmt == "PNG":
            return Image.fromarray(rgb.astype(np.uint8), mode=mode)
        return Image.fromarray(rgb.astype(np.uint8), mode=mode)


def compute_mask_area_percent(mask: np.ndarray) -> float:
    """
    Return the percentage of image pixels covered by the mask.

    Args:
        mask: Binary mask (bool or uint8, H×W).

    Returns:
        Float in [0, 100].
    """
    total = mask.size
    covered = int(np.count_nonzero(mask))
    return round(covered / total * 100, 2)
