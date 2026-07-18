"""
backend/postprocess.py — OpenCV-based mask refinement and output generation.

Pipeline (applied in order):
  1. refine_mask()
     a. Morphological closing  → fills small holes inside the subject
     b. Morphological opening  → removes small isolated blobs
     c. Largest-contour filter → keeps only the primary connected region
     d. Flood-fill hole fill   → eliminates interior holes that closing missed
     e. Gaussian blur + threshold → smooth, anti-aliased edges

  2. make_overlay()
     Blend a semi-transparent colour highlight over the masked region.

  3. make_transparent_png()
     Copy RGB pixels into an RGBA image; alpha channel = refined mask.
"""

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


# ── 1. Mask Refinement ─────────────────────────────────────────────────────────

def refine_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    Apply the full morphological + smoothing pipeline to a raw SAM2 binary mask.

    Args:
        raw_mask: Boolean or uint8 H×W array (True / 255 = foreground).

    Returns:
        Refined uint8 H×W mask (values 0 or 255).
    """
    mask = _to_uint8(raw_mask)

    # a) Morphological closing — close small holes
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KSIZE, MORPH_CLOSE_KSIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # b) Morphological opening — remove tiny isolated blobs
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_OPEN_KSIZE, MORPH_OPEN_KSIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # c) Keep only the largest connected component
    mask = _keep_largest_component(mask)

    # d) Flood-fill to remove any remaining interior holes
    mask = _fill_holes(mask)

    # e) Gaussian blur + re-threshold for smooth edges
    blurred = cv2.GaussianBlur(mask, (GAUSSIAN_KSIZE, GAUSSIAN_KSIZE), GAUSSIAN_SIGMA)
    _, mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    return mask


def _to_uint8(mask: np.ndarray) -> np.ndarray:
    """Normalise any mask representation to uint8 with values 0 / 255."""
    if mask.dtype == bool:
        return (mask.astype(np.uint8)) * 255
    m = mask.astype(np.uint8)
    # Handle masks that are already 0/1 instead of 0/255
    if m.max() == 1:
        m = m * 255
    return m


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Zero out all connected components except the largest one."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask  # Nothing or only background

    # stats row 0 is the background; find the largest foreground component
    foreground_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(foreground_areas)) + 1  # offset by 1 (background)

    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Use flood-fill from the border to identify and fill holes in the mask."""
    h, w = mask.shape
    flood = mask.copy()
    # Flood-fill canvas must be 2 pixels larger on each side
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, fill_mask, (0, 0), 255)
    # Invert the flood — these are the holes
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


# ── 2. Overlay Generation ──────────────────────────────────────────────────────

def make_overlay(image_rgb: np.ndarray, refined_mask: np.ndarray) -> np.ndarray:
    """
    Blend a colour highlight over the segmented region.

    Args:
        image_rgb:    Original RGB image (H×W×3, uint8).
        refined_mask: Refined binary mask (H×W, uint8, values 0/255).

    Returns:
        BGR overlay image (H×W×3, uint8) — BGR for easy OpenCV saving.
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    colour_layer = np.full_like(bgr, OVERLAY_COLOR_BGR, dtype=np.uint8)

    # Only blend where the mask is active
    mask_bool = refined_mask.astype(bool)
    overlay = bgr.copy()
    overlay[mask_bool] = cv2.addWeighted(
        bgr, 1 - OVERLAY_ALPHA, colour_layer, OVERLAY_ALPHA, 0
    )[mask_bool]

    # Draw a thin contour around the segmented region for clarity
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, OVERLAY_COLOR_BGR, 2)

    return overlay


# ── 3. Transparent PNG Generation ─────────────────────────────────────────────

def make_transparent_png(image_rgb: np.ndarray, refined_mask: np.ndarray) -> np.ndarray:
    """
    Create a 4-channel BGRA image where the mask defines the alpha channel.

    Background pixels (mask == 0) become fully transparent.
    Foreground pixels (mask == 255) remain fully opaque.

    Args:
        image_rgb:    Original RGB image (H×W×3, uint8).
        refined_mask: Refined binary mask (H×W, uint8, values 0/255).

    Returns:
        BGRA image (H×W×4, uint8) suitable for saving as PNG with transparency.
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = refined_mask  # alpha channel = mask
    return bgra


# ── 4. Binary Mask PNG ─────────────────────────────────────────────────────────

def make_binary_mask_png(refined_mask: np.ndarray) -> np.ndarray:
    """
    Return the refined mask as a 3-channel BGR image (white on black).
    Useful for the /segment response's `mask` field.
    """
    return cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)
