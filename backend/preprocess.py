from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from backend.config import ALLOWED_CONTENT_TYPES, MAX_SIZE, MAX_UPLOAD_BYTES


@dataclass
class PreprocessResult:
    image_rgb: np.ndarray
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    scale: float


async def preprocess_upload(file: UploadFile, click_x: int, click_y: int) -> Tuple[PreprocessResult, int, int]:
    """Validate, decode, resize the uploaded image and scale click coordinates."""
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Accepted: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        mb = len(raw_bytes) / (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"File too large ({mb:.1f} MB). Max: {MAX_UPLOAD_BYTES // (1024*1024)} MB.")

    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. File may be corrupted.")

    original_h, original_w = bgr.shape[:2]

    if not (0 <= click_x < original_w and 0 <= click_y < original_h):
        raise HTTPException(
            status_code=400,
            detail=f"Click coordinates ({click_x}, {click_y}) are outside image bounds ({original_w}×{original_h}).",
        )

    longest_edge = max(original_h, original_w)
    if longest_edge > MAX_SIZE:
        scale = MAX_SIZE / longest_edge
        bgr = cv2.resize(bgr, (int(original_w * scale), int(original_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    new_h, new_w = bgr.shape[:2]
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    result = PreprocessResult(
        image_rgb=image_rgb,
        original_size=(original_h, original_w),
        processed_size=(new_h, new_w),
        scale=scale,
    )
    return result, int(click_x * scale), int(click_y * scale)


def load_image_from_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to an RGB NumPy array. Used in tests."""
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
