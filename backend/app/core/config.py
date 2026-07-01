"""
Application Configuration

Central configuration management using pydantic-settings.
Reads from environment variables with sensible defaults
for development, production-ready defaults for deployment.
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "CT Simulator API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    # Database (PostgreSQL in Docker; SQLite fallback for local dev)
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ct_simulator"
    POSTGRES_USER: str = "ctuser"
    POSTGRES_PASSWORD: str = "ctpass123"
    DATABASE_URL: str = "sqlite:///./ct_simulator.db"

    # Storage backend: "minio" or "local"
    STORAGE_BACKEND: str = "minio"

    # MinIO (Object Storage)
    MINIO_HOST: str = "minio"
    MINIO_PORT: int = 9000
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_BUCKET: str = "dicom-studies"
    MINIO_URL: str = "http://minio:9000"

    # Upload
    UPLOAD_DIR: str = "/tmp/uploads"
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500MB

    # DICOM file storage (persistent — not cleaned after upload)
    DICOM_STORAGE_DIR: str = "/app/static/dicom"

    # AI Services
    AI_MONAI_ENABLED: bool = False
    AI_DEVICE: str = "cuda"  # Set to "cpu" to force CPU; "cuda" auto-detects NVIDIA GPU
    AI_MODEL_PATH: str = "/app/ai-models"
    AI_TOTALSEGMENTATOR_ENABLED: bool = True

    # TotalSegmentator model weights directory
    # Points to local nnUNet checkpoint files (Dataset291_..., etc.)
    # so the ~2 GB download is skipped on first inference.
    # Resolved automatically for both Docker and local development.
    TOTALSEGMENTATOR_DIR: str = Field(default="/app/ai-models/totalsegmentator")

    TOTALSEGMENTATOR_FAST: bool = True

    # Custom nnUNet model — Dataset701 (6-class, user-trained)
    # Points to the nnUNetTrainer__nnUNetPlans__3d_fullres folder
    # containing dataset.json, plans.json, and fold_0/checkpoint_best.pth
    NNUNET_CUSTOM_MODEL_PATH: str = Field(default="/app/models/nnunet_handoff")

    # Custom nnUNet 20-class model — Dataset702 (20 organs, user-trained)
    # Same nnUNet results folder layout as above.
    NNUNET_CUSTOM_20_MODEL_PATH: str = Field(default="/app/models/nnunet702_handoff")

    # Simulation
    SIMULATION_DEFAULT_SEED: int = 42
    SIMULATION_MAX_LESIONS: int = 50
    SIMULATION_VOXEL_SIZE: float = 0.5

    # Rendering
    RENDERING_CACHE_SIZE: int = 512
    RENDERING_QUALITY: str = "high"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
