"""
backend/preprocess.py — Image loading, validation, and normalisation.

Pipeline:
  1. Validate file size and MIME type
  2. Decode bytes → BGR NumPy array via OpenCV
  3. Convert BGR → RGB (SAM2 expects RGB)
  4. Optionally resize so the longest edge ≤ MAX_SIZE
  5. Return the processed array + scale factor (for remapping click coords)

The scale factor is crucial: if we resize a 2000-px image to 1024 px,
a user click at (800, 600) in the *original* display must be mapped to
(~410, ~307) in the resized coordinate space before passing to SAM2.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from backend.config import ALLOWED_CONTENT_TYPES, MAX_SIZE, MAX_UPLOAD_BYTES


@dataclass
class PreprocessResult:
    """Container returned by `preprocess_upload`."""
    image_rgb: np.ndarray          # H×W×3 float32 or uint8 in RGB
    original_size: Tuple[int, int] # (height, width) before any resize
    processed_size: Tuple[int, int]# (height, width) after resize
    scale: float                   # resize_dim / original_longest_edge (1.0 if no resize)


async def preprocess_upload(file: UploadFile, click_x: int, click_y: int) -> Tuple[PreprocessResult, int, int]:
    """
    Full preprocessing pipeline for an uploaded image and click coordinates.

    Args:
        file:    FastAPI UploadFile object.
        click_x: Horizontal click coordinate in *original image* space.
        click_y: Vertical click coordinate in *original image* space.

    Returns:
        Tuple of (PreprocessResult, scaled_click_x, scaled_click_y).

    Raises:
        HTTPException 400: On invalid file type, corrupted image, or size violations.
    """
    # ── 1. Validate content type ───────────────────────────────────────────────
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Accepted: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    # ── 2. Read bytes + validate size ──────────────────────────────────────────
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        mb = len(raw_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({mb:.1f} MB). Maximum allowed: {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )

    # ── 3. Decode with OpenCV ──────────────────────────────────────────────────
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. File may be corrupted or in an unsupported format.",
        )

    original_h, original_w = bgr.shape[:2]
    original_size = (original_h, original_w)

    # ── 4. Validate click coordinates are within image bounds ──────────────────
    if not (0 <= click_x < original_w and 0 <= click_y < original_h):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Click coordinates ({click_x}, {click_y}) are outside image bounds "
                f"({original_w}×{original_h})."
            ),
        )

    # ── 5. Resize if needed ────────────────────────────────────────────────────
    longest_edge = max(original_h, original_w)
    if longest_edge > MAX_SIZE:
        scale = MAX_SIZE / longest_edge
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        new_h, new_w = original_h, original_w

    processed_size = (new_h, new_w)

    # ── 6. BGR → RGB ───────────────────────────────────────────────────────────
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── 7. Scale click coordinates to resized space ────────────────────────────
    scaled_x = int(click_x * scale)
    scaled_y = int(click_y * scale)

    result = PreprocessResult(
        image_rgb=image_rgb,
        original_size=original_size,
        processed_size=processed_size,
        scale=scale,
    )
    return result, scaled_x, scaled_y


def load_image_from_bytes(raw_bytes: bytes) -> np.ndarray:
    """
    Lightweight helper: decode raw bytes → RGB NumPy array.
    Used by tests that bypass the full UploadFile flow.
    """
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
