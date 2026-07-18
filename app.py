from __future__ import annotations

import base64
import io
import os

import cv2
import httpx
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

load_dotenv()
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="PhotoSAM",
    page_icon="scissors",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        #MainMenu { visibility: hidden; }
        footer     { visibility: hidden; }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6EE7B7 0%, #3B82F6 50%, #9333EA 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.2rem;
        }
        .hero-sub { font-size: 1rem; color: #94A3B8; margin-bottom: 1.5rem; }

        .badge-ok   { background:#064E3B; color:#6EE7B7; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
        .badge-warn { background:#451A03; color:#FDE68A; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
        .badge-err  { background:#450A0A; color:#FCA5A5; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }

        .result-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #64748B;
            margin-bottom: 6px;
        }
        .instructions {
            background: linear-gradient(135deg, #1E293B, #0F172A);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px 20px;
        }
        .step { display:flex; align-items:flex-start; gap:10px; margin:7px 0; color:#CBD5E1; font-size:0.9rem; }
        .step-num { background:#3B82F6; color:#fff; border-radius:50%; width:20px; height:20px; display:flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700; flex-shrink:0; margin-top:2px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_backend_health() -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "connection_refused"}
    except Exception as exc:
        return {"error": str(exc)}


def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def call_segment_api(image_bytes: bytes, filename: str, content_type: str, click_x: int, click_y: int) -> dict:
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{BACKEND_URL}/segment",
            files={"image": (filename, image_bytes, content_type)},
            data={"click_x": str(click_x), "click_y": str(click_y)},
        )
    response.raise_for_status()
    return response.json()


def _init_state():
    defaults = {
        "click_coords": None,
        "segment_result": None,
        "uploaded_bytes": None,
        "uploaded_name": None,
        "uploaded_type": None,
        "original_size": None,
        "display_size": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("**Backend**")
    st.code(BACKEND_URL, language=None)

    health = check_backend_health()
    if "error" in health:
        if health["error"] == "connection_refused":
            st.markdown('<span class="badge-err">Backend offline</span>', unsafe_allow_html=True)
            st.caption("Start with: `uvicorn backend.main:app --reload`")
        else:
            st.markdown(f'<span class="badge-warn">{health["error"]}</span>', unsafe_allow_html=True)
    elif health.get("model_loaded"):
        st.markdown('<span class="badge-ok">SAM2 ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">Mock mode</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("## How it works")
    st.markdown(
        """
        <div class="instructions">
          <div class="step"><div class="step-num">1</div><span>Upload a JPG or PNG image</span></div>
          <div class="step"><div class="step-num">2</div><span>Click the object to segment</span></div>
          <div class="step"><div class="step-num">3</div><span>SAM2 generates the mask</span></div>
          <div class="step"><div class="step-num">4</div><span>Download the output</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("## Links")
    st.markdown(
        "[GitHub](https://github.com/) · "
        "[API Docs](http://localhost:8000/docs) · "
        "[SAM2](https://github.com/facebookresearch/segment-anything-2)"
    )


st.markdown('<h1 class="hero-title">PhotoSAM</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Interactive subject segmentation — click any object, get a precise cutout.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Max 20 MB. JPG and PNG supported.",
    key="uploader",
)

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()
    if st.session_state.uploaded_bytes != raw_bytes:
        st.session_state.uploaded_bytes = raw_bytes
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.uploaded_type = uploaded_file.type
        st.session_state.click_coords = None
        st.session_state.segment_result = None

    pil_image = Image.open(io.BytesIO(st.session_state.uploaded_bytes)).convert("RGB")
    orig_w, orig_h = pil_image.size
    st.session_state.original_size = (orig_w, orig_h)

    st.divider()
    st.markdown("### Click on the object to segment")
    st.caption("Click coordinates are mapped to the original image pixel space.")

    MAX_DISPLAY_W = 700
    if orig_w > MAX_DISPLAY_W:
        display_scale = MAX_DISPLAY_W / orig_w
        display_img = pil_image.resize((MAX_DISPLAY_W, int(orig_h * display_scale)), Image.LANCZOS)
        display_w, display_h = MAX_DISPLAY_W, int(orig_h * display_scale)
    else:
        display_scale = 1.0
        display_img = pil_image
        display_w, display_h = orig_w, orig_h

    st.session_state.display_size = (display_w, display_h)

    display_np = np.array(display_img)
    if st.session_state.click_coords:
        raw_x, raw_y = st.session_state.click_coords
        dx, dy = int(raw_x * display_scale), int(raw_y * display_scale)
        cv2.drawMarker(display_np, (dx, dy), (99, 230, 176), cv2.MARKER_CROSS, markerSize=28, thickness=2)
        cv2.circle(display_np, (dx, dy), 5, (99, 230, 176), -1)
        display_img = Image.fromarray(display_np)

    coords = streamlit_image_coordinates(display_img, key="img_click")

    if coords is not None:
        orig_x = int(coords["x"] / display_scale)
        orig_y = int(coords["y"] / display_scale)
        if st.session_state.click_coords != (orig_x, orig_y):
            st.session_state.click_coords = (orig_x, orig_y)
            st.session_state.segment_result = None
            st.rerun()

    if st.session_state.click_coords:
        cx, cy = st.session_state.click_coords
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.info(f"Point set at ({cx}, {cy})")
        with col_btn:
            segment_clicked = st.button("Segment", type="primary", use_container_width=True)

        if segment_clicked:
            with st.spinner("Running inference …"):
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
                    st.error("Cannot connect to backend.")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.caption("Click anywhere on the image above to set a segmentation point.")

    if st.session_state.segment_result:
        res = st.session_state.segment_result
        st.divider()
        st.markdown("### Results")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Inference", f"{res.get('inference_time', 0):.2f}s")
        col_m2.metric("Total", f"{res.get('total_time', 0):.2f}s")
        col_m3.metric("Coverage", f"{res.get('mask_area', 0):.1f}%")
        col_m4.metric("Confidence", f"{res.get('confidence_score', 0):.3f}")

        st.markdown("---")
        r_col1, r_col2, r_col3 = st.columns(3)

        with r_col1:
            st.markdown('<div class="result-label">Binary Mask</div>', unsafe_allow_html=True)
            mask_img = b64_to_pil(res["mask"])
            st.image(mask_img)
            st.download_button("Download Mask", pil_to_bytes(mask_img), "photosam_mask.png", "image/png", use_container_width=True)

        with r_col2:
            st.markdown('<div class="result-label">Overlay</div>', unsafe_allow_html=True)
            overlay_img = b64_to_pil(res["overlay"])
            st.image(overlay_img)
            st.download_button("Download Overlay", pil_to_bytes(overlay_img, "JPEG"), "photosam_overlay.jpg", "image/jpeg", use_container_width=True)

        with r_col3:
            st.markdown('<div class="result-label">Transparent PNG</div>', unsafe_allow_html=True)
            transparent_img = b64_to_pil(res["transparent_png"])
            bg = Image.new("RGBA", transparent_img.size, (30, 41, 59, 255))
            bg.paste(transparent_img, mask=transparent_img.split()[3])
            st.image(bg.convert("RGB"))
            st.download_button("Download Cutout", pil_to_bytes(transparent_img), "photosam_cutout.png", "image/png", use_container_width=True)

else:
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; padding:60px 20px; color:#475569;">
            <div style="font-size:1.2rem; font-weight:600; color:#94A3B8;">Upload an image to get started</div>
            <div style="font-size:0.88rem; margin-top:8px;">Supports JPG and PNG &middot; Max 20 MB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
