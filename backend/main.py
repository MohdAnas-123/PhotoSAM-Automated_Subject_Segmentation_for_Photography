from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import inference as engine
from backend.config import API_DESCRIPTION, API_TITLE, API_VERSION
from backend.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PhotoSAM starting …")
    engine.load_model()
    if engine.is_model_loaded():
        logger.info("SAM2 model ready.")
    else:
        logger.warning("Running in mock mode — SAM2 not loaded.")
    yield
    logger.info("PhotoSAM shutting down.")


def create_app() -> FastAPI:
    application = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
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
