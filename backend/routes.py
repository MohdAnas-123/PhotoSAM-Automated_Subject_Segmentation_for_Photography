from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend import inference as engine
from backend.postprocess import make_binary_mask_png, make_overlay, make_transparent_png, refine_mask
from backend.preprocess import preprocess_upload
from backend.utils import compute_mask_area_percent, encode_array_to_b64

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", summary="Liveness probe")
async def health() -> dict:
    return {"status": "healthy", "model_loaded": engine.is_model_loaded()}


@router.post("/segment", summary="Segment an object via a single click")
async def segment(
    image: UploadFile = File(..., description="JPG or PNG image to segment"),
    click_x: int = Form(..., description="Horizontal pixel coordinate (original image space)"),
    click_y: int = Form(..., description="Vertical pixel coordinate (original image space)"),
) -> JSONResponse:
    """
    POST image + click coordinates → returns mask, overlay, and transparent PNG.
    All image outputs are base64-encoded.
    """
    t0 = time.perf_counter()
    logger.info("Segment request | click=(%d, %d) | file='%s'", click_x, click_y, image.filename)

    try:
        prep, sx, sy = await preprocess_upload(image, click_x, click_y)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Image preprocessing failed.") from exc

    try:
        result = engine.segment(prep.image_rgb, sx, sy)
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Segmentation inference failed.") from exc

    try:
        refined          = refine_mask(result.mask)
        mask_bgr         = make_binary_mask_png(refined)
        overlay_bgr      = make_overlay(prep.image_rgb, refined)
        transparent_bgra = make_transparent_png(prep.image_rgb, refined)
    except Exception as exc:
        logger.error("Post-processing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Post-processing failed.") from exc

    h, w = prep.processed_size

    return JSONResponse(content={
        "mask":             encode_array_to_b64(mask_bgr, fmt="PNG"),
        "overlay":          encode_array_to_b64(overlay_bgr, fmt="JPEG"),
        "transparent_png":  encode_array_to_b64(transparent_bgra, fmt="PNG"),
        "inference_time":   round(result.inference_time, 3),
        "total_time":       round(time.perf_counter() - t0, 3),
        "mask_area":        compute_mask_area_percent(refined),
        "confidence_score": round(result.score, 4),
        "image_size":       [h, w],
    })
