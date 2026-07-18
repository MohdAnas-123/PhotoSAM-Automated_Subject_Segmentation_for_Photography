"""
app.py — PhotoSAM Streamlit frontend.

Complete user flow:
  1. User uploads a JPG or PNG image
  2. Streamlit renders the image with a click-capture component
  3. User clicks on the object they want to segment
  4. App POSTs image + click coordinates to the FastAPI backend
  5. Backend returns mask, overlay, and transparent PNG (base64)
  6. App displays all three results side-by-side
  7. User can download any of the three outputs

Backend URL configuration:
  Set the BACKEND_URL environment variable to point at any backend instance.
  See .env.example for examples.
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Optional

import httpx
import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# ── Configuration ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhotoSAM — AI Subject Segmentation",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ── Global ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── Hide Streamlit chrome ── */
        #MainMenu { visibility: hidden; }
        footer     { visibility: hidden; }

        /* ── Hero header ── */
        .hero-title {
            font-size: 2.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6EE7B7 0%, #3B82F6 50%, #9333EA 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
        }
        .hero-sub {
            font-size: 1.05rem;
            color: #94A3B8;
            margin-bottom: 1.5rem;
        }

        /* ── Status badges ── */
        .badge-ok   { background:#064E3B; color:#6EE7B7; padding:4px 12px; border-radius:20px; font-size:0.82rem; font-weight:600; }
        .badge-warn { background:#451A03; color:#FDE68A; padding:4px 12px; border-radius:20px; font-size:0.82rem; font-weight:600; }
        .badge-err  { background:#450A0A; color:#FCA5A5; padding:4px 12px; border-radius:20px; font-size:0.82rem; font-weight:600; }

        /* ── Result cards ── */
        .result-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748B;
            margin-bottom: 6px;
        }

        /* ── Metric panel ── */
        .metric-box {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 14px 20px;
            margin-top: 8px;
        }
        .metric-row { display:flex; justify-content:space-between; color:#CBD5E1; font-size:0.88rem; margin:4px 0; }
        .metric-val { font-weight:600; color:#E2E8F0; }

        /* ── Instructions panel ── */
        .instructions {
            background: linear-gradient(135deg, #1E293B, #0F172A);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 18px 22px;
        }
        .step { display:flex; align-items:flex-start; gap:12px; margin:8px 0; color:#CBD5E1; font-size:0.92rem; }
        .step-num { background:#3B82F6; color:#fff; border-radius:50%; width:22px; height:22px; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:700; flex-shrink:0; margin-top:1px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_backend_health() -> dict:
    """Poll /health and return the JSON response, or an error dict."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "connection_refused"}
    except Exception as exc:
        return {"error": str(exc)}


def b64_to_pil(b64_str: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw))


def pil_to_download_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert PIL Image to bytes for st.download_button."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def call_segment_api(
    image_bytes: bytes,
    filename: str,
    content_type: str,
    click_x: int,
    click_y: int,
) -> dict:
    """POST to /segment and return parsed JSON, or raise on failure."""
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{BACKEND_URL}/segment",
            files={"image": (filename, image_bytes, content_type)},
            data={"click_x": str(click_x), "click_y": str(click_y)},
        )
    response.raise_for_status()
    return response.json()


# ── Session state initialisation ────────────────────────────────────────────────

def _init_state():
    defaults = {
        "click_coords": None,        # (x, y) from streamlit-image-coordinates
        "segment_result": None,      # dict from /segment response
        "uploaded_bytes": None,      # raw bytes of the uploaded file
        "uploaded_name": None,
        "uploaded_type": None,
        "original_size": None,       # (W, H) of the PIL image
        "display_size": None,        # (W, H) as rendered in the UI
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown(f"**Backend URL**")
    st.code(BACKEND_URL, language=None)

    # Health check
    health = check_backend_health()
    if "error" in health:
        if health["error"] == "connection_refused":
            st.markdown('<span class="badge-err">⛔ Backend offline</span>', unsafe_allow_html=True)
            st.caption("Start the API with:\n```\nuvicorn backend.main:app --reload\n```")
        else:
            st.markdown(f'<span class="badge-warn">⚠ {health["error"]}</span>', unsafe_allow_html=True)
    elif health.get("model_loaded"):
        st.markdown('<span class="badge-ok">✓ SAM2 model ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">⚠ Mock mode (no model)</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("## 📖 How it works")
    st.markdown(
        """
        <div class="instructions">
          <div class="step"><div class="step-num">1</div><span>Upload a JPG or PNG image</span></div>
          <div class="step"><div class="step-num">2</div><span>Click on the object you want to cut out</span></div>
          <div class="step"><div class="step-num">3</div><span>SAM2 predicts the segmentation mask</span></div>
          <div class="step"><div class="step-num">4</div><span>Download the mask, overlay, or transparent PNG</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("## 🔗 Links")
    st.markdown(
        "[GitHub Repo](https://github.com/) · "
        "[API Docs](http://localhost:8000/docs) · "
        "[Meta SAM2](https://github.com/facebookresearch/segment-anything-2)"
    )


# ── Main Content ───────────────────────────────────────────────────────────────

st.markdown('<h1 class="hero-title">PhotoSAM ✂️</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Interactive AI subject segmentation — click any object, get a perfect cutout.</p>',
    unsafe_allow_html=True,
)

# ── Step 1: Image Upload ───────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Maximum 20 MB. JPG and PNG supported.",
    key="uploader",
)

if uploaded_file is not None:
    # Cache the file bytes so we can reuse across reruns
    raw_bytes = uploaded_file.read()
    if st.session_state.uploaded_bytes != raw_bytes:
        # New image uploaded — reset previous results
        st.session_state.uploaded_bytes = raw_bytes
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.uploaded_type = uploaded_file.type
        st.session_state.click_coords = None
        st.session_state.segment_result = None

    pil_image = Image.open(io.BytesIO(st.session_state.uploaded_bytes)).convert("RGB")
    orig_w, orig_h = pil_image.size
    st.session_state.original_size = (orig_w, orig_h)

    st.divider()

    # ── Step 2: Click on the image ─────────────────────────────────────────────
    st.markdown("### 🖱️ Click on the object you want to segment")
    st.caption("Coordinates are captured in the original image pixel space.")

    # Resize for display (max width 700px for comfortable clicking)
    MAX_DISPLAY_W = 700
    if orig_w > MAX_DISPLAY_W:
        display_scale = MAX_DISPLAY_W / orig_w
        display_w = MAX_DISPLAY_W
        display_h = int(orig_h * display_scale)
        display_img = pil_image.resize((display_w, display_h), Image.LANCZOS)
    else:
        display_scale = 1.0
        display_w, display_h = orig_w, orig_h
        display_img = pil_image

    st.session_state.display_size = (display_w, display_h)

    # Draw crosshair on the image if a click already exists
    display_np = np.array(display_img)
    if st.session_state.click_coords:
        raw_x, raw_y = st.session_state.click_coords
        # Map original → display
        dx = int(raw_x * display_scale)
        dy = int(raw_y * display_scale)
        import cv2
        color = (99, 230, 176)  # mint green
        cv2.drawMarker(display_np, (dx, dy), color, cv2.MARKER_CROSS, markerSize=30, thickness=2)
        cv2.circle(display_np, (dx, dy), 6, color, -1)
        display_img = Image.fromarray(display_np)

    coords = streamlit_image_coordinates(display_img, key="img_click")

    if coords is not None:
        # Map display coords → original image coords
        orig_x = int(coords["x"] / display_scale)
        orig_y = int(coords["y"] / display_scale)
        # Only update if the click changed
        if st.session_state.click_coords != (orig_x, orig_y):
            st.session_state.click_coords = (orig_x, orig_y)
            st.session_state.segment_result = None  # reset results on new click
            st.rerun()

    # ── Click info + Segment button ────────────────────────────────────────────
    if st.session_state.click_coords:
        cx, cy = st.session_state.click_coords
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.info(f"📍 Clicked at pixel **({cx}, {cy})** in original image space")
        with col_btn:
            segment_clicked = st.button("✂️ Segment", type="primary", use_container_width=True)

        if segment_clicked:
            with st.spinner("Running SAM2 inference …"):
                try:
                    result = call_segment_api(
                        image_bytes=st.session_state.uploaded_bytes,
                        filename=st.session_state.uploaded_name,
                        content_type=st.session_state.uploaded_type or "image/jpeg",
                        click_x=cx,
                        click_y=cy,
                    )
                    st.session_state.segment_result = result
                except httpx.HTTPStatusError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text}")
                except httpx.ConnectError:
                    st.error("Cannot connect to the backend. Is it running?")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
    else:
        st.caption("👆 Click anywhere on the image above to set your segmentation point.")

    # ── Step 3: Display Results ────────────────────────────────────────────────
    if st.session_state.segment_result:
        res = st.session_state.segment_result
        st.divider()
        st.markdown("### 🎯 Segmentation Results")

        # Metrics row
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Inference time", f"{res.get('inference_time', 0):.2f}s")
        col_m2.metric("Total time", f"{res.get('total_time', 0):.2f}s")
        col_m3.metric("Mask coverage", f"{res.get('mask_area', 0):.1f}%")
        col_m4.metric("Confidence", f"{res.get('confidence_score', 0):.3f}")

        st.markdown("---")

        # Three output columns
        r_col1, r_col2, r_col3 = st.columns(3)

        with r_col1:
            st.markdown('<div class="result-label">Binary Mask</div>', unsafe_allow_html=True)
            mask_img = b64_to_pil(res["mask"])
            st.image(mask_img, use_container_width=True)
            st.download_button(
                label="⬇ Download Mask",
                data=pil_to_download_bytes(mask_img, "PNG"),
                file_name="photosam_mask.png",
                mime="image/png",
                use_container_width=True,
            )

        with r_col2:
            st.markdown('<div class="result-label">Overlay</div>', unsafe_allow_html=True)
            overlay_img = b64_to_pil(res["overlay"])
            st.image(overlay_img, use_container_width=True)
            st.download_button(
                label="⬇ Download Overlay",
                data=pil_to_download_bytes(overlay_img, "JPEG"),
                file_name="photosam_overlay.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

        with r_col3:
            st.markdown('<div class="result-label">Transparent PNG</div>', unsafe_allow_html=True)
            transparent_img = b64_to_pil(res["transparent_png"])
            # Show on a dark background so transparency is visible
            bg = Image.new("RGBA", transparent_img.size, (30, 41, 59, 255))
            bg.paste(transparent_img, mask=transparent_img.split()[3])
            st.image(bg.convert("RGB"), use_container_width=True)
            st.download_button(
                label="⬇ Download Cutout",
                data=pil_to_download_bytes(transparent_img, "PNG"),
                file_name="photosam_cutout.png",
                mime="image/png",
                use_container_width=True,
            )

else:
    # Empty state — show a welcoming message
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px; color:#475569;">
            <div style="font-size:4rem; margin-bottom:16px;">🖼️</div>
            <div style="font-size:1.2rem; font-weight:600; color:#94A3B8;">Upload an image to get started</div>
            <div style="font-size:0.9rem; margin-top:8px;">
                Supports JPG and PNG · Max 20 MB · Click any object to segment it
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
