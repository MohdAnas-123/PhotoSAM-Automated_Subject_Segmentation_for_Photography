"""
download_model.py — Download the SAM2 model checkpoint and config YAML.

Scope (as decided):
  - Downloads sam2_hiera_tiny.pt  →  models/sam2_hiera_tiny.pt
  - Downloads sam2_hiera_t.yaml   →  models/sam2_hiera_t.yaml  (fallback only —
    the YAML is normally bundled inside the sam2 package after pip install)
  - Safe to run multiple times: skips files that already exist with correct size
  - Does NOT pip-install SAM2 (that belongs in requirements.txt)

Usage:
  python download_model.py

Source:
  https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#model-description
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

# ── Constants ──────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"

# Official checkpoint URLs (Meta's GitHub releases)
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
    "sam2_hiera_tiny.pt"
)

# Config YAML — bundled in the sam2 package, but we keep a fallback copy
CONFIG_URL = (
    "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/"
    "main/sam2/configs/sam2/sam2_hiera_t.yaml"
)

FILES = [
    {
        "url": CHECKPOINT_URL,
        "dest": MODELS_DIR / "sam2_hiera_tiny.pt",
        "label": "SAM2 Hiera-Tiny checkpoint",
        "min_size_mb": 35,   # checkpoint is ~38 MB; reject < 35 MB as corrupt
    },
    {
        "url": CONFIG_URL,
        "dest": MODELS_DIR / "sam2_hiera_t.yaml",
        "label": "SAM2 Hiera-Tiny config YAML",
        "min_size_mb": 0,
    },
]

CHUNK_SIZE = 1024 * 1024  # 1 MB per chunk


# ── Download logic ─────────────────────────────────────────────────────────────

def _download(url: str, dest: Path, label: str, min_size_mb: float) -> None:
    """
    Stream-download `url` to `dest`, showing a progress bar.
    Skips if the file already exists and is large enough.
    """
    min_bytes = int(min_size_mb * 1024 * 1024)

    if dest.exists() and dest.stat().st_size >= max(min_bytes, 1):
        print(f"  ✓ {label} already present — skipping ({dest.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"  ↓ Downloading {label} …")
    print(f"    URL : {url}")
    print(f"    Dest: {dest}")

    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"  ✗ Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    total_bytes = int(resp.headers.get("Content-Length", 0))
    downloaded = 0

    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                fh.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = downloaded / total_bytes * 100
                    bar_len = 40
                    filled = int(bar_len * downloaded / total_bytes)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r    [{bar}] {pct:5.1f}%  {downloaded/1e6:6.1f}/{total_bytes/1e6:.1f} MB", end="", flush=True)

    print()  # newline after progress bar

    if min_bytes and dest.stat().st_size < min_bytes:
        print(
            f"  ✗ Downloaded file is too small ({dest.stat().st_size} bytes). "
            "The checkpoint may be incomplete or the URL has changed.",
            file=sys.stderr,
        )
        dest.unlink(missing_ok=True)
        sys.exit(1)

    print(f"  ✓ {label} saved ({dest.stat().st_size / 1e6:.1f} MB)")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n╔══════════════════════════════════════════╗")
    print("║   PhotoSAM — Model Download              ║")
    print("╚══════════════════════════════════════════╝\n")

    MODELS_DIR.mkdir(exist_ok=True)
    print(f"Models directory: {MODELS_DIR.resolve()}\n")

    for entry in FILES:
        _download(
            url=entry["url"],
            dest=entry["dest"],
            label=entry["label"],
            min_size_mb=entry["min_size_mb"],
        )

    print("\n✅ All model files are ready.")
    print("   You can now start the backend with:")
    print("   uvicorn backend.main:app --reload\n")


if __name__ == "__main__":
    main()
