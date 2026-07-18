"""
tests/test_utils.py — Unit tests for backend utility functions.

Tests:
  - Base64 encode/decode round-trips
  - Mask area computation
  - Postprocessing helpers (refine_mask, make_overlay, make_transparent_png)
  - Preprocessing (resize, coordinate scaling, validation)

These tests have zero I/O and no model dependency.

Run:
  pytest tests/test_utils.py -v
"""

from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _random_rgb(h: int = 100, w: int = 120) -> np.ndarray:
    """Return a random H×W×3 uint8 RGB array."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _circle_mask(h: int = 100, w: int = 120) -> np.ndarray:
    """Return a uint8 mask with a filled circle in the centre."""
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 4, 255, -1)
    return mask


# ── utils.encode_array_to_b64 / decode_b64_to_array ───────────────────────────

class TestBase64RoundTrip:
    def test_rgb_png_round_trip(self):
        from backend.utils import decode_b64_to_array, encode_array_to_b64
        import cv2

        original_rgb = _random_rgb()
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

        b64 = encode_array_to_b64(original_bgr, fmt="PNG")
        assert isinstance(b64, str)
        assert len(b64) > 0

        recovered = decode_b64_to_array(b64)
        assert recovered.shape == original_bgr.shape

    def test_jpeg_round_trip_shape(self):
        from backend.utils import decode_b64_to_array, encode_array_to_b64
        import cv2

        rgb = _random_rgb()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        b64 = encode_array_to_b64(bgr, fmt="JPEG")
        recovered = decode_b64_to_array(b64)
        # JPEG is lossy — only check shape, not pixel values
        assert recovered.shape == bgr.shape

    def test_bgra_png_round_trip(self):
        """Transparent PNG (4-channel) encode/decode."""
        from backend.utils import decode_b64_to_array, encode_array_to_b64
        import cv2

        rgb = _random_rgb()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        b64 = encode_array_to_b64(bgra, fmt="PNG")
        recovered = decode_b64_to_array(b64)
        assert recovered.shape == bgra.shape
        assert recovered.shape[2] == 4  # still 4 channels


# ── utils.compute_mask_area_percent ───────────────────────────────────────────

class TestMaskAreaPercent:
    def test_empty_mask_is_zero(self):
        from backend.utils import compute_mask_area_percent
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert compute_mask_area_percent(mask) == 0.0

    def test_full_mask_is_hundred(self):
        from backend.utils import compute_mask_area_percent
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        assert compute_mask_area_percent(mask) == 100.0

    def test_half_mask(self):
        from backend.utils import compute_mask_area_percent
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:50, :] = 255  # top half is foreground
        area = compute_mask_area_percent(mask)
        assert abs(area - 50.0) < 0.1

    def test_bool_mask_accepted(self):
        from backend.utils import compute_mask_area_percent
        mask = np.zeros((50, 50), dtype=bool)
        mask[:25, :] = True
        area = compute_mask_area_percent(mask)
        assert abs(area - 50.0) < 0.1


# ── postprocess.refine_mask ────────────────────────────────────────────────────

class TestRefineMask:
    def test_output_is_uint8(self):
        from backend.postprocess import refine_mask
        raw = _circle_mask()
        result = refine_mask(raw)
        assert result.dtype == np.uint8

    def test_output_values_binary(self):
        from backend.postprocess import refine_mask
        raw = _circle_mask()
        result = refine_mask(raw)
        unique = set(np.unique(result).tolist())
        assert unique.issubset({0, 255}), f"Non-binary values found: {unique}"

    def test_bool_input_accepted(self):
        from backend.postprocess import refine_mask
        raw = _circle_mask().astype(bool)
        result = refine_mask(raw)
        assert result.dtype == np.uint8

    def test_all_zeros_stays_zeros(self):
        from backend.postprocess import refine_mask
        raw = np.zeros((100, 120), dtype=np.uint8)
        result = refine_mask(raw)
        assert np.all(result == 0)

    def test_shape_preserved(self):
        from backend.postprocess import refine_mask
        raw = _circle_mask(80, 90)
        result = refine_mask(raw)
        assert result.shape == (80, 90)


# ── postprocess.make_overlay ───────────────────────────────────────────────────

class TestMakeOverlay:
    def test_output_shape_matches_input(self):
        from backend.postprocess import make_overlay, refine_mask
        rgb = _random_rgb(100, 120)
        mask = refine_mask(_circle_mask(100, 120))
        overlay = make_overlay(rgb, mask)
        assert overlay.shape == (100, 120, 3)

    def test_output_is_uint8(self):
        from backend.postprocess import make_overlay, refine_mask
        rgb = _random_rgb()
        mask = refine_mask(_circle_mask())
        overlay = make_overlay(rgb, mask)
        assert overlay.dtype == np.uint8


# ── postprocess.make_transparent_png ──────────────────────────────────────────

class TestMakeTransparentPng:
    def test_output_has_4_channels(self):
        from backend.postprocess import make_transparent_png, refine_mask
        rgb = _random_rgb()
        mask = refine_mask(_circle_mask())
        bgra = make_transparent_png(rgb, mask)
        assert bgra.shape[2] == 4

    def test_background_is_fully_transparent(self):
        from backend.postprocess import make_transparent_png
        rgb = _random_rgb(100, 120)
        mask = np.zeros((100, 120), dtype=np.uint8)  # no foreground
        bgra = make_transparent_png(rgb, mask)
        # All alpha values should be 0
        assert np.all(bgra[:, :, 3] == 0)

    def test_foreground_is_fully_opaque(self):
        from backend.postprocess import make_transparent_png
        rgb = _random_rgb(100, 120)
        mask = np.ones((100, 120), dtype=np.uint8) * 255  # all foreground
        bgra = make_transparent_png(rgb, mask)
        assert np.all(bgra[:, :, 3] == 255)


# ── preprocess.load_image_from_bytes ──────────────────────────────────────────

class TestLoadImageFromBytes:
    def _make_jpeg(self, w=80, h=60) -> bytes:
        img = Image.fromarray(_random_rgb(h, w))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_returns_rgb_array(self):
        from backend.preprocess import load_image_from_bytes
        raw = self._make_jpeg()
        arr = load_image_from_bytes(raw)
        assert arr.ndim == 3
        assert arr.shape[2] == 3

    def test_raises_on_garbage(self):
        from backend.preprocess import load_image_from_bytes
        with pytest.raises(ValueError, match="Could not decode"):
            load_image_from_bytes(b"definitely not an image")


# ── inference mock ─────────────────────────────────────────────────────────────

class TestMockInference:
    def test_mock_returns_inference_result(self):
        from backend.inference import _mock_segment
        rgb = _random_rgb(100, 120)
        result = _mock_segment(rgb, click_x=60, click_y=50)
        assert result.mask.shape == (100, 120)
        assert result.mask.dtype == bool
        assert 0.0 <= result.score <= 1.0
        assert result.inference_time >= 0

    def test_mock_mask_covers_reasonable_area(self):
        """Ellipse should cover somewhere between 5% and 95% of the image."""
        from backend.inference import _mock_segment
        rgb = _random_rgb(200, 300)
        result = _mock_segment(rgb, click_x=150, click_y=100)
        coverage = result.mask.mean() * 100
        assert 5 < coverage < 95, f"Suspicious mock coverage: {coverage:.1f}%"
