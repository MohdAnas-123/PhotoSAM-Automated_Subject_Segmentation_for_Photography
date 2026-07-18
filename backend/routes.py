"""
backend/routes.py — FastAPI route definitions.

Endpoints:
  GET  /health   → liveness probe + model status
  POST /segment  → main segmentation endpoint
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend import inference as engine
from backend.postprocess import (
    make_binary_mask_png,
    make_overlay,
    make_transparent_png,
    refine_mask,
)
from backend.preprocess import preprocess_upload
from backend.utils import compute_mask_area_percent, encode_array_to_b64

logger = logging.getLogger(__name__)
router = APIRouter()


# ── GET /health ────────────────────────────────────────────────────────────────

@router.get("/health", summary="Liveness probe")
async def health() -> dict:
    """
    Returns the API status and whether the SAM2 model is loaded.

    Response example:
    ```json
    {"status": "healthy", "model_loaded": true}
    ```
    """
    return {
        "status": "healthy",
        "model_loaded": engine.is_model_loaded(),
    }


# ── POST /segment ──────────────────────────────────────────────────────────────

@router.post("/segment", summary="Segment an object via a single click")
async def segment(
    image: UploadFile = File(..., description="JPG or PNG image to segment"),
    click_x: int = Form(..., description="Horizontal pixel coordinate of the user's click (original image space)"),
    click_y: int = Form(..., description="Vertical pixel coordinate of the user's click (original image space)"),
) -> JSONResponse:
    """
    Accepts an image and a click coordinate; returns three segmentation outputs.

    **Request** (`multipart/form-data`):
    - `image`   — JPEG or PNG file
    - `click_x` — x pixel (column) of the user click, in original image coordinates
    - `click_y` — y pixel (row) of the user click, in original image coordinates

    **Response** (JSON):
    ```json
    {
        "mask":            "<base64 PNG>",
        "overlay":         "<base64 JPEG>",
        "transparent_png": "<base64 PNG>",
        "inference_time":  0.87,
        "mask_area":       32.6,
        "image_size":      [480, 640]
    }
    ```
    """
    request_start = time.perf_counter()
    logger.info("Segment request received | click=(%d, %d) | file='%s'", click_x, click_y, image.filename)

    # ── 1. Preprocess ──────────────────────────────────────────────────────────
    try:
        prep, scaled_x, scaled_y = await preprocess_upload(image, click_x, click_y)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Image preprocessing failed.") from exc

    # ── 2. SAM2 Inference ──────────────────────────────────────────────────────
    try:
        result = engine.segment(prep.image_rgb, scaled_x, scaled_y)
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Segmentation inference failed.") from exc

    # ── 3. Post-processing ─────────────────────────────────────────────────────
    try:
        refined = refine_mask(result.mask)
        mask_bgr = make_binary_mask_png(refined)
        overlay_bgr = make_overlay(prep.image_rgb, refined)
        transparent_bgra = make_transparent_png(prep.image_rgb, refined)
    except Exception as exc:
        logger.error("Post-processing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Post-processing failed.") from exc

    # ── 4. Encode outputs ──────────────────────────────────────────────────────
    mask_b64 = encode_array_to_b64(mask_bgr, fmt="PNG")
    overlay_b64 = encode_array_to_b64(overlay_bgr, fmt="JPEG")
    transparent_b64 = encode_array_to_b64(transparent_bgra, fmt="PNG")

    total_time = round(time.perf_counter() - request_start, 3)
    mask_area = compute_mask_area_percent(refined)
    h, w = prep.processed_size

    logger.info(
        "Segment complete | total=%.3fs | inference=%.3fs | mask_area=%.1f%%",
        total_time,
        result.inference_time,
        mask_area,
    )

    return JSONResponse(
        content={
            "mask": mask_b64,
            "overlay": overlay_b64,
            "transparent_png": transparent_b64,
            "inference_time": round(result.inference_time, 3),
            "total_time": total_time,
            "mask_area": mask_area,
            "confidence_score": round(result.score, 4),
            "image_size": [h, w],
        }
    )
