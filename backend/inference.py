from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_predictor = None
_model_loaded: bool = False
_mock_mode: bool = False


@dataclass
class InferenceResult:
    mask: np.ndarray
    score: float
    inference_time: float


def load_model() -> None:
    """Load SAM2 predictor at startup. Falls back to mock mode gracefully."""
    global _predictor, _model_loaded, _mock_mode

    from backend.config import DEVICE, MODEL_CFG, MODEL_PATH

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not MODEL_PATH.exists():
            logger.warning("Checkpoint not found at '%s'. Run `python download_model.py`. Using mock mode.", MODEL_PATH)
            _mock_mode = True
            return

        logger.info("Loading SAM2 on device='%s' …", DEVICE)
        _predictor = SAM2ImagePredictor(build_sam2(MODEL_CFG, str(MODEL_PATH), device=DEVICE))
        _model_loaded = True
        logger.info("SAM2 loaded.")

    except ImportError:
        logger.warning("SAM2 not installed. Using mock mode. Install with: pip install 'sam2 @ git+https://github.com/facebookresearch/segment-anything-2.git'")
        _mock_mode = True
    except Exception as exc:
        logger.error("Failed to load SAM2: %s", exc, exc_info=True)
        _mock_mode = True


def is_model_loaded() -> bool:
    return _model_loaded


def segment(image_rgb: np.ndarray, click_x: int, click_y: int) -> InferenceResult:
    """Run segmentation for a single positive point prompt."""
    if _mock_mode or not _model_loaded:
        return _mock_segment(image_rgb, click_x, click_y)
    return _sam2_segment(image_rgb, click_x, click_y)


def _sam2_segment(image_rgb: np.ndarray, click_x: int, click_y: int) -> InferenceResult:
    import torch
    t0 = time.perf_counter()
    with torch.inference_mode():
        _predictor.set_image(image_rgb)
        masks, scores, _ = _predictor.predict(
            point_coords=np.array([[click_x, click_y]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
    inference_time = time.perf_counter() - t0
    best_idx = int(np.argmax(scores))
    logger.info("SAM2 done in %.3fs | score=%.4f | coverage=%.1f%%", inference_time, scores[best_idx], masks[best_idx].mean() * 100)
    return InferenceResult(mask=masks[best_idx].astype(bool), score=float(scores[best_idx]), inference_time=inference_time)


def _mock_segment(image_rgb: np.ndarray, click_x: int, click_y: int) -> InferenceResult:
    """Synthetic ellipse mask for local development without SAM2."""
    import cv2
    t0 = time.perf_counter()
    h, w = image_rgb.shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(canvas, (click_x, click_y), (int(w * 0.30), int(h * 0.30)), 0, 0, 360, 255, -1)
    return InferenceResult(mask=canvas.astype(bool), score=0.99, inference_time=time.perf_counter() - t0)
