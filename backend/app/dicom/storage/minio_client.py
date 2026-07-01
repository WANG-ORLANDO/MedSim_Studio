"""
MinIO Object Storage Client

Thin wrapper around the MinIO Python SDK for uploading, downloading,
and managing DICOM files in S3-compatible object storage.
"""

import logging
from datetime import timedelta
from typing import Optional

from minio import Minio
from minio.error import S3Error

from app.core.config import settings
from app.dicom.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class MinIOStorage(StorageBackend):
    """
    MinIO client wrapper.

    Provides high-level methods for DICOM file storage operations.
    All methods catch exceptions and return bool/None to prevent
    unhandled errors from propagating to callers.
    """

    def __init__(self):
        endpoint = f"{settings.MINIO_HOST}:{settings.MINIO_PORT}"
        self._client = Minio(
            endpoint,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,
        )
        self._bucket = settings.MINIO_BUCKET

    def ensure_bucket(self) -> bool:
        try:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
                logger.info("Created MinIO bucket: %s", self._bucket)
            return True
        except S3Error:
            logger.exception("Failed to ensure MinIO bucket: %s", self._bucket)
            return False
        except Exception:
            logger.exception("Unexpected error ensuring MinIO bucket")
            return False

    def ensure_storage(self) -> bool:
        """Implement StorageBackend interface."""
        return self.ensure_bucket()

    def upload_file(
        self,
        object_key: str,
        file_path: str,
        content_type: str = "application/dicom",
    ) -> bool:
        try:
            self._client.fput_object(
                self._bucket, object_key, file_path, content_type=content_type
            )
            logger.info("Uploaded %s -> %s/%s", file_path, self._bucket, object_key)
            return True
        except S3Error:
            logger.exception("MinIO upload failed: %s", object_key)
            return False
        except Exception:
            logger.exception("Unexpected error uploading to MinIO")
            return False

    def download_file(self, object_key: str, destination_path: str) -> bool:
        try:
            self._client.fget_object(self._bucket, object_key, destination_path)
            logger.info("Downloaded %s/%s -> %s", self._bucket, object_key, destination_path)
            return True
        except S3Error:
            logger.exception("MinIO download failed: %s", object_key)
            return False
        except Exception:
            logger.exception("Unexpected error downloading from MinIO")
            return False

    def delete_file(self, object_key: str) -> bool:
        try:
            self._client.remove_object(self._bucket, object_key)
            logger.info("Deleted %s/%s", self._bucket, object_key)
            return True
        except S3Error:
            logger.exception("MinIO delete failed: %s", object_key)
            return False
        except Exception:
            logger.exception("Unexpected error deleting from MinIO")
            return False

    def get_presigned_url(
        self, object_key: str, expires_seconds: int = 3600
    ) -> Optional[str]:
        try:
            url = self._client.presigned_get_object(
                self._bucket,
                object_key,
                expires=timedelta(seconds=expires_seconds),
            )
            return url
        except S3Error:
            logger.exception("Failed to generate presigned URL: %s", object_key)
            return None
        except Exception:
            logger.exception("Unexpected error generating presigned URL")
            return None

    def get_object_bytes(self, object_key: str) -> Optional[bytes]:
        try:
            response = self._client.get_object(self._bucket, object_key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except S3Error as e:
            if "NoSuchKey" in str(e) or "NoSuchObject" in str(e):
                logger.warning("Object not found in MinIO: %s", object_key)
            else:
                logger.exception("MinIO get_object_bytes failed: %s", object_key)
            return None
        except Exception:
            logger.exception("Unexpected error in get_object_bytes")
            return None

    def check_health(self) -> bool:
        try:
            return self._client.bucket_exists(self._bucket)
        except Exception:
            logger.exception("MinIO health check failed")
            return False
