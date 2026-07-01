"""
Local Filesystem Storage Backend

Local storage implementation that uses the filesystem to store DICOM objects.
Uses the same object_key semantics as MinIO for database compatibility.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.dicom.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class LocalStorage(StorageBackend):
    """
    Local filesystem storage backend.

    Stores DICOM files under DICOM_STORAGE_DIR using object_key paths.
    Example: object_key="dicom/1.2.3/4.5.6/7.8.9.dcm"
             → stored at {DICOM_STORAGE_DIR}/dicom/1.2.3/4.5.6/7.8.9.dcm
    """

    def __init__(self):
        self._storage_root = Path(settings.DICOM_STORAGE_DIR)
        logger.info(f"LocalStorage initialized with root: {self._storage_root}")

    def _resolve_path(self, object_key: str) -> Path:
        """Convert object_key to absolute filesystem path."""
        # Prevent path traversal attacks
        clean_key = object_key.lstrip("/\\")
        if ".." in clean_key:
            raise ValueError(f"Invalid object_key (path traversal detected): {object_key}")
        return self._storage_root / clean_key

    def ensure_storage(self) -> bool:
        """Ensure storage directory exists and is writable."""
        try:
            self._storage_root.mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = self._storage_root / ".health_check"
            test_file.write_text("test")
            test_file.unlink()
            logger.info(f"Local storage ready: {self._storage_root}")
            return True
        except Exception:
            logger.exception(f"Failed to ensure local storage directory: {self._storage_root}")
            return False

    def upload_file(
        self,
        object_key: str,
        file_path: str,
        content_type: str = "application/dicom",
    ) -> bool:
        """Copy local file to storage under object_key."""
        try:
            dest_path = self._resolve_path(object_key)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_path)
            logger.info(f"Uploaded {file_path} -> {dest_path}")
            return True
        except Exception:
            logger.exception(f"Local storage upload failed: {object_key}")
            return False

    def get_object_bytes(self, object_key: str) -> Optional[bytes]:
        """Read object bytes from filesystem."""
        try:
            file_path = self._resolve_path(object_key)
            if not file_path.exists():
                logger.warning(f"Object not found in local storage: {object_key}")
                return None
            return file_path.read_bytes()
        except Exception:
            logger.exception(f"Local storage get_object_bytes failed: {object_key}")
            return None

    def download_file(self, object_key: str, destination_path: str) -> bool:
        """Copy object from storage to destination path."""
        try:
            source_path = self._resolve_path(object_key)
            if not source_path.exists():
                logger.warning(f"Object not found in local storage: {object_key}")
                return False
            shutil.copy2(source_path, destination_path)
            logger.info(f"Downloaded {source_path} -> {destination_path}")
            return True
        except Exception:
            logger.exception(f"Local storage download failed: {object_key}")
            return False

    def delete_file(self, object_key: str) -> bool:
        """Delete object from filesystem."""
        try:
            file_path = self._resolve_path(object_key)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted {file_path}")
            return True
        except Exception:
            logger.exception(f"Local storage delete failed: {object_key}")
            return False

    def check_health(self) -> bool:
        """Check if storage directory exists and is writable."""
        try:
            if not self._storage_root.exists():
                return False
            test_file = self._storage_root / ".health_check"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception:
            logger.exception("Local storage health check failed")
            return False
