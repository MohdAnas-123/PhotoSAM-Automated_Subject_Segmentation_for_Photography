# PhotoSAM ✂️

> **Interactive AI subject segmentation** powered by [Meta SAM2](https://github.com/facebookresearch/segment-anything-2).  
> Click any object in a photo — get a refined mask, an overlay, and a transparent PNG cutout in seconds.

---

## Demo

| Original | Click | Mask | Overlay | Transparent PNG |
|----------|-------|------|---------|-----------------|
| ![](docs/screenshots/original.png) | 📍 Click | ![](docs/screenshots/mask.png) | ![](docs/screenshots/overlay.png) | ![](docs/screenshots/cutout.png) |

> *(Screenshots will be added after first successful end-to-end run)*

---

## Features

| Feature | Description |
|---------|-------------|
| 🖱️ **Interactive click** | Click any pixel to select the subject — no bounding box or brush needed |
| 🧠 **SAM2 inference** | Meta's `sam2-hiera-tiny` — fast, accurate, CPU-capable |
| 🔬 **Mask refinement** | Morphological open/close + Gaussian smoothing via OpenCV |
| 🖼️ **Three outputs** | Binary mask · Colour overlay · Transparent PNG cutout |
| ⬇️ **Download** | One-click download for each output |
| 🌐 **REST API** | Clean FastAPI backend with Swagger docs at `/docs` |
| 🔀 **Env-var backend** | Single `BACKEND_URL` var → switch local / Colab / HF instantly |

---

## Architecture

```
┌─────────────────────┐         HTTP (multipart)         ┌─────────────────────┐
│  Streamlit Frontend │  ──────────────────────────────▶  │   FastAPI Backend   │
│      app.py         │  ◀──────────────────────────────  │  backend/main.py    │
└─────────────────────┘          JSON response            └──────────┬──────────┘
                                                                     │
                                              ┌──────────────────────┼──────────────────────┐
                                              ▼                      ▼                      ▼
                                       preprocess.py          inference.py           postprocess.py
                                       • Validate             • SAM2 predictor       • Morphology
                                       • BGR→RGB              • Point prompt         • Overlay
                                       • Resize               • Best mask            • Alpha PNG
```

---

## Project Structure

```
PhotoSAM/
├── app.py                  ← Streamlit frontend
├── download_model.py       ← Downloads SAM2 checkpoint
├── requirements.txt
├── .env.example            ← Copy to .env and set BACKEND_URL
├── README.md
│
├── backend/
│   ├── main.py             ← FastAPI app factory + lifespan
│   ├── routes.py           ← GET /health · POST /segment
│   ├── inference.py        ← SAM2 predictor wrapper (+ mock fallback)
│   ├── preprocess.py       ← Validate · resize · BGR→RGB
│   ├── postprocess.py      ← Morphology · overlay · transparent PNG
│   ├── config.py           ← All tuneable settings
│   └── utils.py            ← Base64 helpers · file I/O
│
├── models/                 ← Checkpoint lives here (git-ignored)
├── outputs/                ← Runtime outputs (git-ignored)
├── tests/
│   ├── test_api.py         ← FastAPI integration tests
│   └── test_utils.py       ← Utility + postprocess unit tests
└── examples/
    ├── input/              ← Sample input images
    └── output/             ← Sample output images
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Git

### 1 — Clone

```bash
git clone https://github.com/your-username/PhotoSAM.git
cd PhotoSAM
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users**: Replace the CPU PyTorch lines in `requirements.txt` with:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 3 — Download the SAM2 checkpoint

```bash
python download_model.py
```

Downloads `sam2_hiera_tiny.pt` (~38 MB) into `models/`.

### 4 — Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)

### 5 — Start the frontend

```bash
# In a new terminal
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Environment Variables

Copy `.env.example` to `.env` and set your backend URL:

```env
# Local
BACKEND_URL=http://localhost:8000

# Google Colab (Phase 2)
# BACKEND_URL=https://xxxx.ngrok-free.app

# Hugging Face Spaces (Phase 3)
# BACKEND_URL=https://your-username-photosam.hf.space
```

Other optional overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `PHOTOSAM_DEVICE` | `auto` | `cuda` or `cpu` |
| `PHOTOSAM_MAX_SIZE` | `1024` | Max image dimension before resize |
| `PHOTOSAM_MAX_UPLOAD_MB` | `20` | Upload size limit |

---

## API Reference

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /segment`

**Request** (`multipart/form-data`):

| Field | Type | Description |
|-------|------|-------------|
| `image` | File | JPEG or PNG image |
| `click_x` | int | Horizontal pixel coordinate (original image space) |
| `click_y` | int | Vertical pixel coordinate (original image space) |

**Response**:

```json
{
  "mask":            "<base64 PNG>",
  "overlay":         "<base64 JPEG>",
  "transparent_png": "<base64 PNG>",
  "inference_time":  0.87,
  "total_time":      0.95,
  "mask_area":       32.6,
  "confidence_score": 0.9821,
  "image_size":      [480, 640]
}
```

---

## Internal Pipeline

```
Upload Image
    │
    ▼
Validate (MIME type · size · bounds)
    │
    ▼
Decode (OpenCV) · BGR→RGB · Resize (≤1024px)
    │
    ▼
SAM2 Predictor ← positive point prompt (x, y)
    │
    ▼
Choose best mask (highest confidence score)
    │
    ▼
Morphological Closing  → fill holes
Morphological Opening  → remove blobs
Largest component      → keep subject
Flood-fill             → fill interior holes
Gaussian smooth        → anti-alias edges
    │
    ├──▶ Binary mask PNG
    ├──▶ Colour overlay JPEG
    └──▶ Transparent cutout PNG
    │
    ▼
JSON Response
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests run **without** the SAM2 model — the inference engine automatically
activates mock mode when the checkpoint is missing (e.g. in CI).

---

## Development Workflow (Phase 2 — Colab + ngrok)

For GPU inference without a local GPU:

1. Open the Colab notebook in `notebooks/colab_backend.ipynb`
2. Run all cells — installs SAM2, downloads checkpoint, starts FastAPI, creates ngrok tunnel
3. Copy the printed ngrok URL
4. Set `BACKEND_URL=https://xxxx.ngrok-free.app` in your local `.env`
5. Run `streamlit run app.py` locally — it now talks to the Colab GPU

---

## Model Choice

| Model | Params | Speed (CPU) | Quality |
|-------|--------|-------------|---------|
| `sam2-hiera-tiny` ✅ | 38M | ~1–2s | Good |
| `sam2-hiera-small` | 46M | ~2–4s | Better |
| `sam2-hiera-large` | 224M | ~10s+ | Best |

`sam2-hiera-tiny` was chosen for CPU viability and Hugging Face ZeroGPU compatibility.

---

## Resume Highlights

- **Interactive AI tool** — not a static demo; real click-based segmentation
- **Production-grade REST API** — FastAPI with Swagger docs, async I/O, proper error handling
- **Full OpenCV pipeline** — morphological refinement, anti-aliasing, alpha compositing
- **Graceful degradation** — mock mode enables full-pipeline testing without the model
- **Environment-driven config** — one env var to switch backends; zero code changes
- **Documented, tested, structured** — engineering best practices throughout

---

## Stack

`Python` · `FastAPI` · `SAM2` · `PyTorch` · `OpenCV` · `Streamlit` · `Pillow` · `NumPy`

---

## License

This project is released under the [MIT License](LICENSE).  
Meta SAM2 is released under the [Apache 2.0 License](https://github.com/facebookresearch/segment-anything-2/blob/main/LICENSE).
