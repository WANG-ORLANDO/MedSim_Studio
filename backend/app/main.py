"""
CT Simulator - Backend Main Entry Point

FastAPI application initialization with middleware, routers,
and lifecycle management. Provides the RESTful API backend
for DICOM management, simulation, and AI segmentation.

Start command:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__, __app_name__
from app.core.config import settings
from app.api.v1 import health_router, dicom_router, simulation_router, segmentation_router
from app.api.v1.artifact import router as artifact_router
from app.dicom.storage import StorageBackend, get_storage_backend
from app.database.session import engine, Base
import app.models.dicom  # noqa: F401 — register models for create_all
import app.models.simulation  # noqa: F401 — register models for create_all
import app.models.segmentation  # noqa: F401 — register models for create_all
import app.models.artifact  # noqa: F401 — register models for create_all

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Storage backend instance, initialized during lifespan startup.
# health.py imports this to perform real readiness checks.
storage_backend: StorageBackend | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    Handles startup and shutdown events including:
    - Database connection pool initialization
    - Storage backend initialization (MinIO or Local)
    - AI model loading
    - Volume rendering cache warmup
    """
    global storage_backend
    logger.info(f"Starting {__app_name__} v{__version__}")

    # --- Startup ---
    # Run Alembic migrations to create/upgrade database tables.
    # Falls back to create_all + stamp if Alembic is not yet configured
    # (e.g., first run after switching from create_all to Alembic).
    try:
        from alembic.config import Config
        from alembic import command
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        logger.info("Alembic migrations up to date")
    except Exception as e:
        logger.warning("Alembic migration failed (%s); falling back to create_all", e)
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created/verified via create_all")
            # Stamp as head so Alembic is in sync for next startup
            try:
                from alembic.config import Config
                from alembic import command
                alembic_cfg = Config("alembic.ini")
                command.stamp(alembic_cfg, "head")
                logger.info("Alembic stamped at head")
            except Exception as stamp_err:
                logger.warning("Failed to stamp Alembic head (%s); manual stamp may be needed", stamp_err)
        except Exception as create_err:
            logger.exception("Failed to create database tables; continuing startup (%s)", create_err)

    # Initialize storage backend and ensure it is ready.
    # Failure here does NOT crash the app — the backend still starts,
    # but readiness check will report storage as unhealthy.
    try:
        storage_backend = get_storage_backend()
        if storage_backend.ensure_storage():
            if settings.STORAGE_BACKEND.lower() == "minio":
                logger.info(f"MinIO bucket ready: {settings.MINIO_BUCKET}")
            else:
                logger.info(f"Local storage ready: {settings.DICOM_STORAGE_DIR}")
        else:
            logger.error(
                "Storage backend ensure_storage failed; "
                "storage will be reported unhealthy"
            )
    except Exception:
        logger.exception("Storage backend initialization failed; continuing startup without storage")

    # TODO: Load AI models (if enabled)
    # TODO: Initialize volume rendering cache

    yield

    # --- Shutdown ---
    logger.info(f"Shutting down {__app_name__}")
    # TODO: Close database connections
    # TODO: Release AI model resources
    # TODO: Clear rendering cache


# Create FastAPI application
app = FastAPI(
    title=__app_name__,
    version=__version__,
    description=(
        "Web-based CT medical imaging platform with MPR, "
        "volume rendering, lesion simulation, and AI segmentation."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json",
)

# --- Middleware ---

# CORS configuration for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---

# Health checks (no prefix, for load balancers)
app.include_router(health_router, prefix="/api/v1")

# API v1 routes
app.include_router(dicom_router, prefix="/api/v1")
app.include_router(simulation_router, prefix="/api/v1")
app.include_router(segmentation_router, prefix="/api/v1")
app.include_router(artifact_router, prefix="/api/v1")


# --- Legacy / root-level health endpoint ---
@app.get("/health")
async def root_health():
    """Root-level health check for Docker and reverse proxies."""
    return {
        "status": "healthy",
        "app": __app_name__,
        "version": __version__,
    }


# --- Static files (for serving rendered volumes) ---
# app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Startup event for easy configuration check ---
@app.on_event("startup")
async def startup_event():
    """Log configuration on startup for debugging."""
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"CORS origins: {settings.CORS_ORIGINS}")
    logger.info(f"AI enabled: {settings.AI_MONAI_ENABLED}")
    logger.info(f"Database: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
    logger.info(f"MinIO: {settings.MINIO_HOST}:{settings.MINIO_PORT}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
