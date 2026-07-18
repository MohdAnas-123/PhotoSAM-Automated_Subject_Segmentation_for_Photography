import os
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "sam2_hiera_tiny.pt"
MODEL_CFG  = "sam2_hiera_t.yaml"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE      = os.getenv("PHOTOSAM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_SIZE    = int(os.getenv("PHOTOSAM_MAX_SIZE", 1024))
NUM_MASKS   = 1

MORPH_CLOSE_KSIZE  = 15
MORPH_OPEN_KSIZE   = 5
GAUSSIAN_KSIZE     = 7
GAUSSIAN_SIGMA     = 2.0
OVERLAY_ALPHA      = 0.45
OVERLAY_COLOR_BGR  = (0, 200, 100)

API_TITLE       = "PhotoSAM API"
API_DESCRIPTION = "Interactive AI subject segmentation powered by Meta SAM2."
API_VERSION     = "1.0.0"

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES      = int(os.getenv("PHOTOSAM_MAX_UPLOAD_MB", 20)) * 1024 * 1024
