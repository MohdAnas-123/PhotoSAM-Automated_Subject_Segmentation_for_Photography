"""
backend/main.py — FastAPI application factory.

Responsibilities:
  - Create the FastAPI application instance
  - Register startup / shutdown lifespan events
  - Mount the router
  - Configure logging
  - Configure CORS (permissive for development)

Usage:
  uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import inference as engine
from backend.config import API_DESCRIPTION, API_TITLE, API_VERSION
from backend.routes import router

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Startup:  Load the SAM2 model into memory once.
    Shutdown: (future) clean up GPU memory / temp files.
    """
    logger.info("PhotoSAM API starting up …")
    engine.load_model()
    if engine.is_model_loaded():
        logger.info("✓ SAM2 model is ready.")
    else:
        logger.warning("⚠ Running in mock mode — SAM2 model not loaded.")
    yield
    # --- shutdown ---
    logger.info("PhotoSAM API shutting down.")


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    application = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow all origins during development (tighten in production)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)
    return application


app = create_app()
