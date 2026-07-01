"""
Storage Backend Abstract Interface

Unified storage abstraction for DICOM object persistence.
Implementations: MinIOStorage (object storage), LocalStorage (filesystem).
"""

from abc import ABC, abstractmethod
from typing import Optional


class StorageBackend(ABC):
    """
    Abstract storage backend interface.

    All implementations must use the same object_key semantics
    (e.g. "dicom/{study_uid}/{series_uid}/{sop_uid}.dcm") so that
    database fields (pixel_data_path, storage_path) remain consistent
    regardless of which backend is active.
    """

    @abstractmethod
    def ensure_storage(self) -> bool:
        """Ensure the storage location is initialized and ready."""

    @abstractmethod
    def upload_file(
        self,
        object_key: str,
        file_path: str,
        content_type: str = "application/dicom",
    ) -> bool:
        """Upload a local file to storage under the given object_key."""

    @abstractmethod
    def get_object_bytes(self, object_key: str) -> Optional[bytes]:
        """Retrieve the raw bytes of an object. Returns None on failure."""

    @abstractmethod
    def download_file(self, object_key: str, destination_path: str) -> bool:
        """Download an object to a local file path."""

    @abstractmethod
    def delete_file(self, object_key: str) -> bool:
        """Delete an object from storage."""

    @abstractmethod
    def check_health(self) -> bool:
        """Return True if the storage backend is healthy and reachable."""
