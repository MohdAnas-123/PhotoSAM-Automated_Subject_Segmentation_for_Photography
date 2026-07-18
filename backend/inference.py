"""
backend/inference.py — SAM2 model wrapper.

This module owns the SAM2 predictor lifecycle:
  - Loaded once at application startup (via FastAPI lifespan event)
  - Kept as a module-level singleton to avoid reload overhead per request
  - Exposes a single `segment()` function consumed by routes.py

Mock mode:
  When SAM2 is not installed (e.g. during local development without GPU),
  the engine falls back to a deterministic mock that generates a centred
  ellipse mask — good enough to exercise the full pipeline without the model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Singleton state ────────────────────────────────────────────────────────────
_predictor = None          # SAM2ImagePredictor instance (None until loaded)
_model_loaded: bool = False
_mock_mode: bool = False   # True when SAM2 is unavailable


@dataclass
class InferenceResult:
    """Structured output from a single segmentation inference."""
    mask: np.ndarray       # H×W bool array — True = foreground
    score: float           # SAM2 confidence score for the chosen mask
    inference_time: float  # Wall-clock seconds for SAM2 inference only


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model() -> None:
    """
    Initialise the SAM2 predictor and store it in module state.

    Called once from the FastAPI startup lifespan event.
    Falls back to mock mode gracefully if the package or checkpoint is missing.
    """
    global _predictor, _model_loaded, _mock_mode

    from backend.config import DEVICE, MODEL_CFG, MODEL_PATH

    try:
        # Import SAM2 — installed via:
        #   sam2 @ git+https://github.com/facebookresearch/segment-anything-2.git
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not MODEL_PATH.exists():
            logger.warning(
                "SAM2 checkpoint not found at '%s'. "
                "Run `python download_model.py` first. Falling back to mock mode.",
                MODEL_PATH,
            )
            _mock_mode = True
            return

        logger.info("Loading SAM2 model on device='%s' …", DEVICE)
        sam2_model = build_sam2(MODEL_CFG, str(MODEL_PATH), device=DEVICE)
        _predictor = SAM2ImagePredictor(sam2_model)
        _model_loaded = True
        logger.info("SAM2 model loaded successfully.")

    except ImportError:
        logger.warning(
            "SAM2 package not found. Install it with:\n"
            "  pip install 'sam2 @ git+https://github.com/facebookresearch/segment-anything-2.git'\n"
            "Falling back to mock mode.",
        )
        _mock_mode = True
    except Exception as exc:
        logger.error("Failed to load SAM2 model: %s", exc, exc_info=True)
        _mock_mode = True


def is_model_loaded() -> bool:
    """Return True if the real SAM2 model is loaded and ready."""
    return _model_loaded


# ── Inference ──────────────────────────────────────────────────────────────────

def segment(image_rgb: np.ndarray, click_x: int, click_y: int) -> InferenceResult:
    """
    Run segmentation for a single positive point prompt.

    Args:
        image_rgb: H×W×3 uint8 RGB image (preprocessed, already resized).
        click_x:   Horizontal coordinate in the resized image space.
        click_y:   Vertical coordinate in the resized image space.

    Returns:
        InferenceResult with the best-scoring mask, its confidence, and timing.
    """
    if _mock_mode or not _model_loaded:
        return _mock_segment(image_rgb, click_x, click_y)

    return _sam2_segment(image_rgb, click_x, click_y)


# ── Real SAM2 inference ────────────────────────────────────────────────────────

def _sam2_segment(image_rgb: np.ndarray, click_x: int, click_y: int) -> InferenceResult:
    """Run actual SAM2 inference with a single positive click prompt."""
    import torch

    t0 = time.perf_counter()

    with torch.inference_mode():
        # Set the image — SAM2 builds its internal embedding here
        _predictor.set_image(image_rgb)

        # Single positive point prompt: label 1 = foreground
        point_coords = np.array([[click_x, click_y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        masks, scores, _ = _predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,  # get up to 3 candidates
        )

    inference_time = time.perf_counter() - t0

    # Choose the mask with the highest confidence score
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(bool)
    best_score = float(scores[best_idx])

    logger.info(
        "SAM2 inference done in %.3fs | score=%.4f | coverage=%.1f%%",
        inference_time,
        best_score,
        best_mask.mean() * 100,
    )

    return InferenceResult(mask=best_mask, score=best_score, inference_time=inference_time)


# ── Mock inference (no SAM2 required) ─────────────────────────────────────────

def _mock_segment(image_rgb: np.ndarray, click_x: int, click_y: int) -> InferenceResult:
    """
    Generate a synthetic ellipse mask centred on the click point.

    Useful during development / CI where SAM2 is not installed.
    The ellipse covers roughly 25% of the image area.
    """
    t0 = time.perf_counter()
    h, w = image_rgb.shape[:2]

    mask = np.zeros((h, w), dtype=bool)
    # Ellipse axes: 30% of each dimension
    a = int(w * 0.30)
    b = int(h * 0.30)

    # Draw filled ellipse into the mask
    import cv2 as _cv
    canvas = np.zeros((h, w), dtype=np.uint8)
    _cv.ellipse(canvas, (click_x, click_y), (a, b), 0, 0, 360, 255, -1)
    mask = canvas.astype(bool)

    inference_time = time.perf_counter() - t0
    logger.info("Mock inference done in %.3fs (no SAM2 model)", inference_time)

    return InferenceResult(mask=mask, score=0.99, inference_time=inference_time)
