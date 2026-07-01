"""
Storage Backend Factory

Creates the appropriate storage backend instance based on
the STORAGE_BACKEND configuration setting.
"""

import logging
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    from app.dicom.storage.base import StorageBackend

logger = logging.getLogger(__name__)


def get_storage_backend() -> "StorageBackend":
    """
    Factory function to create storage backend instance.

    Returns:
        StorageBackend: MinIOStorage or LocalStorage instance

    Raises:
        ValueError: If STORAGE_BACKEND is not recognized
    """
    backend_type = settings.STORAGE_BACKEND.lower()

    if backend_type == "minio":
        from app.dicom.storage.minio_client import MinIOStorage
        logger.info("Using MinIO storage backend")
        return MinIOStorage()

    elif backend_type == "local":
        from app.dicom.storage.local_storage import LocalStorage
        logger.info(f"Using local storage backend: {settings.DICOM_STORAGE_DIR}")
        return LocalStorage()

    else:
        error_msg = (
            f"Unknown STORAGE_BACKEND: '{settings.STORAGE_BACKEND}'. "
            f"Supported values: 'minio', 'local'"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
