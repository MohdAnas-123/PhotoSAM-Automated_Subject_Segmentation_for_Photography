from __future__ import annotations

import cv2
import numpy as np

from backend.config import (
    GAUSSIAN_KSIZE,
    GAUSSIAN_SIGMA,
    MORPH_CLOSE_KSIZE,
    MORPH_OPEN_KSIZE,
    OVERLAY_ALPHA,
    OVERLAY_COLOR_BGR,
)


def refine_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    Morphological refinement pipeline:
    closing → opening → largest component → hole fill → Gaussian smooth.
    """
    mask = _to_uint8(raw_mask)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KSIZE, MORPH_CLOSE_KSIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_KSIZE, MORPH_OPEN_KSIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    mask = _keep_largest_component(mask)
    mask = _fill_holes(mask)

    blurred = cv2.GaussianBlur(mask, (GAUSSIAN_KSIZE, GAUSSIAN_KSIZE), GAUSSIAN_SIGMA)
    _, mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return mask


def make_overlay(image_rgb: np.ndarray, refined_mask: np.ndarray) -> np.ndarray:
    """Blend a colour highlight over the masked region and draw a contour."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    colour_layer = np.full_like(bgr, OVERLAY_COLOR_BGR, dtype=np.uint8)
    mask_bool = refined_mask.astype(bool)
    overlay = bgr.copy()
    overlay[mask_bool] = cv2.addWeighted(bgr, 1 - OVERLAY_ALPHA, colour_layer, OVERLAY_ALPHA, 0)[mask_bool]
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, OVERLAY_COLOR_BGR, 2)
    return overlay


def make_transparent_png(image_rgb: np.ndarray, refined_mask: np.ndarray) -> np.ndarray:
    """Return a BGRA image where alpha = refined mask (background is transparent)."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = refined_mask
    return bgra


def make_binary_mask_png(refined_mask: np.ndarray) -> np.ndarray:
    """Return the mask as a 3-channel BGR image (white on black)."""
    return cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)


def _to_uint8(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == bool:
        return mask.astype(np.uint8) * 255
    m = mask.astype(np.uint8)
    return m * 255 if m.max() == 1 else m


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = mask.copy()
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, fill_mask, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))
