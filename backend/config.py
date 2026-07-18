"""
backend/config.py — Centralised application settings.

All tuneable parameters live here. Override via environment variables
where noted; everything else can be changed in a single place.
"""

import os
import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# SAM2 checkpoint that download_model.py fetches
MODEL_PATH: Path = BASE_DIR / "models" / "sam2_hiera_tiny.pt"

# SAM2 config YAML — bundled inside the sam2 package after pip install
# Reference: https://github.com/facebookresearch/segment-anything-2/tree/main/sam2/configs
MODEL_CFG: str = "sam2_hiera_t.yaml"

# Where runtime outputs are written (per-request files, for debugging)
OUTPUT_DIR: Path = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Inference ──────────────────────────────────────────────────────────────────
# Automatically use GPU when available; fall back to CPU
DEVICE: str = os.getenv("PHOTOSAM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Images larger than MAX_SIZE (on the longest edge) are downscaled before inference
MAX_SIZE: int = int(os.getenv("PHOTOSAM_MAX_SIZE", 1024))

# SAM2 returns up to 3 mask candidates; we always pick the one with highest score
NUM_MASKS: int = 1

# ── Post-processing ────────────────────────────────────────────────────────────
# Morphological kernel sizes
MORPH_CLOSE_KSIZE: int = 15   # fills small holes inside the mask
MORPH_OPEN_KSIZE: int = 5     # removes small isolated blobs

# Gaussian blur for edge smoothing (must be odd)
GAUSSIAN_KSIZE: int = 7
GAUSSIAN_SIGMA: float = 2.0

# Overlay blend alpha (0.0 = invisible, 1.0 = solid colour)
OVERLAY_ALPHA: float = 0.45

# Highlight colour for overlay (BGR format for OpenCV)
OVERLAY_COLOR_BGR: tuple = (0, 200, 100)  # vibrant green

# ── API ────────────────────────────────────────────────────────────────────────
API_TITLE: str = "PhotoSAM API"
API_DESCRIPTION: str = (
    "Interactive AI subject segmentation powered by Meta SAM2.\n\n"
    "POST an image + a click coordinate → get back a binary mask, "
    "an overlay visualisation, and a transparent PNG cutout."
)
API_VERSION: str = "1.0.0"

# Accepted MIME types for uploaded images
ALLOWED_CONTENT_TYPES: set = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_MB: int = int(os.getenv("PHOTOSAM_MAX_UPLOAD_MB", 20))
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024
