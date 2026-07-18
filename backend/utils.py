import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.config import OUTPUT_DIR


def encode_array_to_b64(image: np.ndarray, fmt: str = "PNG") -> str:
    """Encode a NumPy image array (BGR / BGRA / grayscale) to a base64 string."""
    pil_img = _ndarray_to_pil(image, fmt)
    buffer = io.BytesIO()
    save_kwargs: dict = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs["quality"] = 92
    pil_img.save(buffer, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_b64_to_array(b64_string: str) -> np.ndarray:
    """Decode a base64 image string to a NumPy BGR array."""
    raw = base64.b64decode(b64_string)
    buf = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def get_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_output(image: np.ndarray, prefix: str, fmt: str = "png") -> Path:
    """Write a NumPy image to the outputs/ directory."""
    filename = OUTPUT_DIR / f"{prefix}_{get_timestamp()}.{fmt}"
    cv2.imwrite(str(filename), image)
    return filename


def compute_mask_area_percent(mask: np.ndarray) -> float:
    """Return the percentage of pixels covered by the mask."""
    return round(int(np.count_nonzero(mask)) / mask.size * 100, 2)


def _ndarray_to_pil(image: np.ndarray, fmt: str) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image.astype(np.uint8), mode="L")
    elif image.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA).astype(np.uint8), mode="RGBA")
    else:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8), mode="RGB")
