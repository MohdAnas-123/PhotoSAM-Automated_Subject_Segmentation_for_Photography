"""
tests/test_api.py — Integration tests for the PhotoSAM FastAPI endpoints.

Strategy:
  - Uses FastAPI's TestClient (via httpx) so no live server is needed.
  - SAM2 inference is mocked via monkeypatching — tests exercise the full
    HTTP/preprocessing/postprocessing pipeline without a real model.
  - All tests are self-contained and do not write to disk.

Run:
  pytest tests/test_api.py -v
"""

from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(width: int = 200, height: int = 150) -> bytes:
    """Create a tiny in-memory JPEG image for testing."""
    img = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(width: int = 200, height: int = 150) -> bytes:
    """Create a tiny in-memory PNG image for testing."""
    img = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    """
    Return a FastAPI TestClient backed by the PhotoSAM app.
    The SAM2 predictor is NOT loaded during tests — mock mode activates
    automatically (checkpoint missing in CI).
    """
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        yield c


# ── GET /health ────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_keys(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data

    def test_health_status_is_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_model_loaded_is_bool(self, client):
        data = client.get("/health").json()
        assert isinstance(data["model_loaded"], bool)


# ── POST /segment ──────────────────────────────────────────────────────────────

class TestSegment:
    def test_segment_jpeg_returns_200(self, client):
        resp = client.post(
            "/segment",
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
            data={"click_x": "100", "click_y": "75"},
        )
        assert resp.status_code == 200

    def test_segment_png_returns_200(self, client):
        resp = client.post(
            "/segment",
            files={"image": ("test.png", _make_png_bytes(), "image/png")},
            data={"click_x": "100", "click_y": "75"},
        )
        assert resp.status_code == 200

    def test_segment_response_has_all_fields(self, client):
        resp = client.post(
            "/segment",
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
            data={"click_x": "100", "click_y": "75"},
        )
        data = resp.json()
        for field in ("mask", "overlay", "transparent_png", "inference_time", "mask_area"):
            assert field in data, f"Missing field: {field}"

    def test_segment_outputs_are_base64_strings(self, client):
        import base64
        resp = client.post(
            "/segment",
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
            data={"click_x": "50", "click_y": "50"},
        )
        data = resp.json()
        for field in ("mask", "overlay", "transparent_png"):
            raw = base64.b64decode(data[field])  # should not raise
            assert len(raw) > 0, f"Empty base64 output for '{field}'"

    def test_segment_mask_area_in_range(self, client):
        resp = client.post(
            "/segment",
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
            data={"click_x": "100", "click_y": "75"},
        )
        area = resp.json()["mask_area"]
        assert 0.0 <= area <= 100.0, f"mask_area out of range: {area}"

    def test_segment_inference_time_positive(self, client):
        resp = client.post(
            "/segment",
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
            data={"click_x": "100", "click_y": "75"},
        )
        t = resp.json()["inference_time"]
        assert t >= 0, f"Negative inference time: {t}"

    # ── Error cases ────────────────────────────────────────────────────────────

    def test_segment_rejects_unsupported_type(self, client):
        resp = client.post(
            "/segment",
            files={"image": ("test.gif", b"GIF89a", "image/gif")},
            data={"click_x": "10", "click_y": "10"},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_segment_rejects_out_of_bounds_click(self, client):
        """A click far outside the 200×150 test image should return 400."""
        resp = client.post(
            "/segment",
            files={"image": ("test.jpg", _make_jpeg_bytes(200, 150), "image/jpeg")},
            data={"click_x": "9999", "click_y": "9999"},
        )
        assert resp.status_code == 400
        assert "outside image bounds" in resp.json()["detail"]

    def test_segment_rejects_corrupted_image(self, client):
        resp = client.post(
            "/segment",
            files={"image": ("bad.jpg", b"not-an-image", "image/jpeg")},
            data={"click_x": "10", "click_y": "10"},
        )
        assert resp.status_code == 400
