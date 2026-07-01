"""
DICOM Management API

RESTful endpoints for DICOM study, series, and instance management.
Provides CRUD operations plus DICOM file upload, parsing, and serving.
"""

import os
import uuid
import shutil
import logging
import traceback
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

import pydicom

from app.database.session import get_db
from app.models.dicom import DicomStudy, DicomSeries, DicomInstance
from app.schemas.dicom import (
    DicomStudyResponse,
    DicomSeriesResponse,
    DicomInstanceResponse,
    DicomUploadResponse,
    PaginatedResponse,
)
from app.dicom.parser.dicom_parser import DicomParser
from app.dicom.storage import get_storage_backend
from app.core.config import settings

logger = logging.getLogger(__name__)


def safe_exception_message(e: Exception) -> str:
    """Safely stringify exceptions that may contain non-UTF8 payloads."""
    try:
        return str(e)
    except UnicodeDecodeError:
        return repr(e)
    except Exception:
        return e.__class__.__name__


def safe_dicom_value(value) -> Optional[str]:
    """Safely convert a pydicom attribute value to a plain Python str."""
    if value is None:
        return None
    try:
        result = str(value)
        if not isinstance(result, str):
            result = repr(value)
        return result
    except UnicodeDecodeError:
        try:
            if hasattr(value, "original_string"):
                raw = value.original_string
                if isinstance(raw, bytes):
                    return raw.decode("latin1", errors="replace")
            return repr(value)
        except Exception:
            return repr(value)
    except Exception:
        return repr(value)


_TEXT_VRS = {
    "AE", "AS", "CS", "DA", "DS", "DT", "IS", "LO", "LT",
    "PN", "SH", "ST", "TM", "UC", "UI", "UR", "UT",
}


def _sanitize_dicom_dataset(ds):
    """Pre-decode all text VR values to safe Python strings.

    pydicom lazily decodes text VR values on first str() access.
    If SpecificCharacterSet is missing or wrong, str() raises UnicodeDecodeError.
    This forces all text values to Python str eagerly, falling back to
    latin1 with errors='replace' on failure.
    """
    if not getattr(ds, "SpecificCharacterSet", None):
        ds.SpecificCharacterSet = "ISO_IR 192"

    for elem in ds:
        if elem.VR in _TEXT_VRS and elem.value is not None:
            try:
                safe = str(elem.value)
                if isinstance(safe, str):
                    elem.value = safe
            except (UnicodeDecodeError, Exception):
                try:
                    if hasattr(elem.value, "original_string"):
                        raw = elem.value.original_string
                        if isinstance(raw, bytes):
                            elem.value = raw.decode("latin1", errors="replace")
                            continue
                    elem.value = repr(elem.value)
                except Exception:
                    elem.value = ""


def _patched_dcmread_force(original):
    """Create a wrapper that forces force=True and sanitizes text values on pydicom.dcmread."""
    def wrapper(*args, **kwargs):
        kwargs["force"] = True
        ds = original(*args, **kwargs)
        try:
            _sanitize_dicom_dataset(ds)
        except Exception:
            logger.warning("Failed to sanitize DICOM text values", exc_info=True)
        return ds

    for attr in ("__module__", "__name__", "__qualname__", "__doc__", "__annotations__"):
        try:
            setattr(wrapper, attr, getattr(original, attr))
        except (AttributeError, TypeError):
            pass
    wrapper.__wrapped__ = original
    return wrapper


router = APIRouter(prefix="/dicom", tags=["DICOM"])


def _generate_dicom_object_key(
    study_uid: str, series_uid: str, sop_uid: str
) -> Optional[str]:
    """
    Generate MinIO object key for a DICOM file.

    Format: dicom/{study_uid}/{series_uid}/{sop_uid}.dcm

    Returns None if any UID is missing or empty after cleanup.
    """
    study_uid = str(study_uid).strip().replace("/", "_").replace("\\", "_")
    series_uid = str(series_uid).strip().replace("/", "_").replace("\\", "_")
    sop_uid = str(sop_uid).strip().replace("/", "_").replace("\\", "_")

    if not study_uid or not series_uid or not sop_uid:
        return None

    return f"dicom/{study_uid}/{series_uid}/{sop_uid}.dcm"


@router.get("/studies", response_model=PaginatedResponse)
async def list_studies(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by patient name or ID"),
    db: Session = Depends(get_db),
):
    """
    List all DICOM studies with pagination and optional search.
    Returns basic study metadata for the study list view.
    """
    query = db.query(DicomStudy)

    if search:
        search_filter = f"%{search}%"
        query = query.filter(
            DicomStudy.patient_name.ilike(search_filter) |
            DicomStudy.patient_id.ilike(search_filter)
        )

    total = query.count()
    total_pages = max(1, (total + page_size - 1) // page_size)
    studies = query.order_by(DicomStudy.created_at.desc()) \
        .offset((page - 1) * page_size) \
        .limit(page_size) \
        .all()

    return PaginatedResponse(
        items=[DicomStudyResponse.model_validate(s) for s in studies],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/studies/{study_id}", response_model=DicomStudyResponse)
async def get_study(study_id: str, db: Session = Depends(get_db)):
    """Get detailed information about a specific study."""
    study = db.query(DicomStudy).filter(DicomStudy.id == study_id).first()
    if not study:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Study {study_id} not found",
        )
    return study


@router.get("/studies/{study_id}/series", response_model=List[DicomSeriesResponse])
async def get_study_series(study_id: str, db: Session = Depends(get_db)):
    """Get all series within a study."""
    study = db.query(DicomStudy).filter(DicomStudy.id == study_id).first()
    if not study:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Study {study_id} not found",
        )
    return study.series


@router.get("/studies/{study_id}/series/{series_id}", response_model=DicomSeriesResponse)
async def get_series_detail(study_id: str, series_id: str, db: Session = Depends(get_db)):
    """Get detailed information about a specific series."""
    series = db.query(DicomSeries).filter(
        DicomSeries.id == series_id,
        DicomSeries.study_id == study_id,
    ).first()
    if not series:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Series {series_id} not found in study {study_id}",
        )
    return series


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


@router.get(
    "/series/{series_id}/instances",
    response_model=List[DicomInstanceResponse],
)
async def get_series_instances(series_id: str, db: Session = Depends(get_db)):
    """
    Get all DICOM instances (slices) within a series.
    Ordered by instance number for correct slice ordering.
    """
    series = db.query(DicomSeries).filter(DicomSeries.id == series_id).first()
    if not series:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Series {series_id} not found",
        )

    instances = (
        db.query(DicomInstance)
        .filter(DicomInstance.series_id == series_id)
        .order_by(DicomInstance.instance_number.asc().nulls_last())
        .all()
    )

    return [
        DicomInstanceResponse.model_validate(inst) for inst in instances
    ]


@router.get("/instances/{instance_id}/file")
async def get_instance_file(instance_id: str, db: Session = Depends(get_db)):
    """
    Serve the raw DICOM file for an instance.
    Used by Cornerstone3D's wadouri image loader to load image data.

    Phase 5A: Reads from MinIO object storage instead of local file path.
    """
    instance = db.query(DicomInstance).filter(
        DicomInstance.id == instance_id
    ).first()
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance {instance_id} not found",
        )

    if not instance.pixel_data_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DICOM object key is empty for instance {instance_id}",
        )

    storage = get_storage_backend()

    if not storage.check_health():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Storage service unavailable",
        )

    dicom_bytes = storage.get_object_bytes(instance.pixel_data_path)
    if dicom_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DICOM object not found in storage for instance {instance_id}",
        )

    filename = f"{instance.sop_instance_uid}.dcm" if instance.sop_instance_uid else f"{instance_id}.dcm"

    return Response(
        content=dicom_bytes,
        media_type="application/dicom",
        headers={
            "Content-Disposition": f"inline; filename=\"{filename}\"",
        },
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=DicomUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dicom(
    files: List[UploadFile] = File(..., description="DICOM files to upload"),
    study_id: Optional[str] = Query(None, description="Optional study ID to associate files with"),
    db: Session = Depends(get_db),
):
    """
    Upload and parse DICOM files.

    Accepts multiple DICOM files, pre-parses to extract UIDs, uploads to MinIO,
    then parses metadata and creates database records with MinIO object keys.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided",
        )

    # Use a temp staging area for upload
    upload_id = str(uuid.uuid4())
    staging_dir = f"{settings.UPLOAD_DIR}/{upload_id}"
    os.makedirs(staging_dir, exist_ok=True)

    saved_paths = []
    uploaded_keys: List[str] = []  # Track uploaded MinIO objects for rollback
    storage = get_storage_backend()

    try:
        # Step 1: Save uploaded files to staging directory
        for file in files:
            content = await file.read()
            file_name = file.filename or f"dicom_{uuid.uuid4()}.dcm"
            # Sanitize: strip leading slashes/dots to prevent path traversal
            file_name = file_name.lstrip("/").lstrip("\\").lstrip(".")
            file_path = os.path.join(staging_dir, file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(content)
            saved_paths.append(file_path)

        # Step 2: Pre-parse to extract UIDs and build storage_map
        pre_parse_results: List[tuple[str, str]] = []  # [(file_path, object_key), ...]

        for file_path in saved_paths:
            try:
                # Read only metadata, skip pixel data for performance
                ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)

                # Sanitize text values to prevent UTF-8 decode errors on str() calls
                _sanitize_dicom_dataset(ds)

                # Extract required UIDs
                study_uid = getattr(ds, "StudyInstanceUID", None)
                series_uid = getattr(ds, "SeriesInstanceUID", None)
                sop_uid = getattr(ds, "SOPInstanceUID", None)

                # Generate object key
                object_key = _generate_dicom_object_key(study_uid, series_uid, sop_uid)

                if not object_key:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"DICOM file {os.path.basename(file_path)} missing required UIDs "
                               f"(StudyInstanceUID, SeriesInstanceUID, or SOPInstanceUID)",
                    )

                pre_parse_results.append((file_path, object_key))

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to parse DICOM file {os.path.basename(file_path)}: {safe_exception_message(e)}",
                )

        # Step 3: Build storage_map and upload to MinIO
        storage_map: Dict[str, str] = {
            file_path: object_key for file_path, object_key in pre_parse_results
        }

        for file_path, object_key in pre_parse_results:
            success = storage.upload_file(
                object_key=object_key,
                file_path=file_path,
                content_type="application/dicom",
            )
            if not success:
                # Rollback: delete already uploaded files
                for uploaded_key in uploaded_keys:
                    storage.delete_file(uploaded_key)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Failed to upload {os.path.basename(file_path)} to MinIO storage",
                )
            uploaded_keys.append(object_key)

        # Step 4: Parse DICOM files with storage_map
        # Monkey-patch pydicom.dcmread to force force=True during parsing.
        # The parser calls dcmread without force=True, which causes UnicodeDecodeError
        # on DICOM files with non-UTF8 encoded text tags (e.g. Chinese PatientName).
        parser = DicomParser()
        original_dcmread = pydicom.dcmread
        try:
            pydicom.dcmread = _patched_dcmread_force(original_dcmread)
            result = parser.parse_files(saved_paths, db, storage_map=storage_map)
        except Exception as e:
            # Parser failed: try to clean up MinIO objects
            for uploaded_key in uploaded_keys:
                try:
                    storage.delete_file(uploaded_key)
                except Exception:
                    pass
            raise
        finally:
            pydicom.dcmread = original_dcmread

        return DicomUploadResponse(
            study_id=result["study_id"],
            series_count=result["series_count"],
            instance_count=result["instance_count"],
        )

    except HTTPException:
        raise
    except Exception as e:
        # Unknown error: try to clean up MinIO objects
        for uploaded_key in uploaded_keys:
            try:
                storage.delete_file(uploaded_key)
            except Exception:
                pass
        logger.exception("FULL TRACEBACK in upload_dicom")
        tb = traceback.format_exc()
        logger.error(tb)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TRACEBACK:\n{tb}",
        )
    finally:
        # Clean up staging directory
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


@router.delete("/studies/{study_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_study(study_id: str, db: Session = Depends(get_db)):
    """Delete a study and all associated series, instances, and MinIO objects.

    Order of operations:
      1. Load study (404 if missing).
      2. Cache MinIO object keys from all instances BEFORE db.delete (cascade
         invalidates ORM relationships after delete).
      3. db.delete(study) + db.commit().
      4. On successful commit, best-effort delete cached MinIO objects.

    MinIO cleanup failures are logged but never block the 204 response. If the
    DB commit fails, MinIO is not touched.
    """
    study = db.query(DicomStudy).filter(DicomStudy.id == study_id).first()
    if not study:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Study {study_id} not found",
        )

    series_ids = [s.id for s in study.series]
    object_keys: List[str] = []
    if series_ids:
        instances = (
            db.query(DicomInstance)
            .filter(DicomInstance.series_id.in_(series_ids))
            .all()
        )
        object_keys = [
            inst.pixel_data_path
            for inst in instances
            if inst.pixel_data_path
        ]

    try:
        db.delete(study)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to delete study %s from database", study_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete study {study_id} from database",
        )

    if object_keys:
        storage = get_storage_backend()
        for key in object_keys:
            if not isinstance(key, str) or not key.startswith("dicom/"):
                logger.warning(
                    "Skipping unexpected MinIO object key for study %s: %r",
                    study_id,
                    key,
                )
                continue
            try:
                ok = storage.delete_file(key)
                if not ok:
                    logger.warning(
                        "MinIO delete_file reported failure for key %s (study %s)",
                        key,
                        study_id,
                    )
            except Exception:
                logger.exception(
                    "Unexpected error deleting MinIO object %s (study %s)",
                    key,
                    study_id,
                )
