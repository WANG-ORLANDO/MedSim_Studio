"""
Simulation Result Exporter

Exports completed simulation results from storage backend
in NRRD, NIfTI, or DICOM series (zip) format with streaming download.

Responsibilities:
- Download result.nrrd from storage backend to temp file
- Convert NRRD -> NIfTI via SimpleITK
- Convert NRRD -> DICOM series zip via pydicom
- Stream file responses with cleanup via BackgroundTask
"""

import io
import os
import shutil
import logging
import tempfile
import zipfile
from typing import Iterator, Optional

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from pydicom.sequence import Sequence
import SimpleITK as sitk
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from app.dicom.storage.base import StorageBackend
from app.models.simulation import SimulationJob

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cleanup_temp_dir(temp_dir: str) -> None:
    """Remove a temporary directory and all its contents."""
    try:
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug("Cleaned up temp dir: %s", temp_dir)
    except Exception:
        logger.warning("Failed to clean up temp dir: %s", temp_dir)


def iter_file(file_path: str, chunk_size: int = CHUNK_SIZE) -> Iterator[bytes]:
    """Yield file contents in chunks."""
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk


def stream_file_response(
    file_path: str,
    filename: str,
    media_type: str,
    temp_dir: str,
) -> StreamingResponse:
    """Build a StreamingResponse that streams file_path and cleans up temp_dir after."""
    file_size = os.path.getsize(file_path)
    return StreamingResponse(
        iter_file(file_path),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(file_size),
        },
        background=BackgroundTask(cleanup_temp_dir, temp_dir),
    )


def download_nrrd_to_temp(
    storage: StorageBackend,
    output_path: str,
    temp_dir: str,
) -> str:
    """Download result.nrrd from storage backend to a temp file. Returns the local path."""
    local_path = os.path.join(temp_dir, "result.nrrd")
    ok = storage.download_file(output_path, local_path)
    if not ok or not os.path.isfile(local_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download result from storage (key={output_path})",
        )
    return local_path


def _load_sitk_image(nrrd_path: str) -> sitk.Image:
    """Load an NRRD file as a SimpleITK Image."""
    try:
        return sitk.ReadImage(nrrd_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read NRRD file: {e}",
        )


# ---------------------------------------------------------------------------
# NRRD export
# ---------------------------------------------------------------------------


def export_nrrd(job: SimulationJob, storage: StorageBackend) -> StreamingResponse:
    """Stream the result.nrrd directly from storage."""
    temp_dir = tempfile.mkdtemp(prefix=f"export_{job.id}_nrrd_")
    nrrd_path = download_nrrd_to_temp(storage, job.output_path, temp_dir)
    filename = f"simulation_{job.id}.nrrd"
    return stream_file_response(
        file_path=nrrd_path,
        filename=filename,
        media_type="application/octet-stream",
        temp_dir=temp_dir,
    )


# ---------------------------------------------------------------------------
# NIfTI export
# ---------------------------------------------------------------------------


def export_nifti(job: SimulationJob, storage: StorageBackend) -> StreamingResponse:
    """Convert result.nrrd to .nii.gz and stream it."""
    temp_dir = tempfile.mkdtemp(prefix=f"export_{job.id}_nifti_")
    try:
        nrrd_path = download_nrrd_to_temp(storage, job.output_path, temp_dir)
        sitk_image = _load_sitk_image(nrrd_path)

        nifti_path = os.path.join(temp_dir, "result.nii.gz")
        sitk.WriteImage(sitk_image, nifti_path)
        logger.info(
            "Job %s: converted NRRD -> NIfTI (%.1f MB)",
            job.id, os.path.getsize(nifti_path) / (1024 * 1024),
        )

        filename = f"simulation_{job.id}.nii.gz"
        return stream_file_response(
            file_path=nifti_path,
            filename=filename,
            media_type="application/gzip",
            temp_dir=temp_dir,
        )
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception as e:
        cleanup_temp_dir(temp_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NIfTI export failed: {e}",
        )


# ---------------------------------------------------------------------------
# DICOM zip export
# ---------------------------------------------------------------------------


def _clip_to_int16(volume: np.ndarray) -> np.ndarray:
    """Clip HU values to int16 range and cast."""
    i16_min = np.iinfo(np.int16).min
    i16_max = np.iinfo(np.int16).max
    return np.clip(volume, i16_min, i16_max).astype(np.int16)


def _build_slice_dicom(
    volume_3d: np.ndarray,
    slice_idx: int,
    job_id: str,
    study_uid: str,
    series_uid: str,
    spacing: tuple,
    origin: tuple,
) -> Dataset:
    """
    Build a single DICOM CT slice dataset.

    volume_3d: shape (z, y, x), int16
    spacing: (x, y, z) in mm — SimpleITK order
    origin: (x, y, z) in mm — SimpleITK order
    """
    sop_uid = generate_uid()

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        filename_or_obj="",
        dataset={},
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )

    # Patient
    ds.PatientName = "Simulated^Patient"
    ds.PatientID = job_id[:16]

    # Study / Series
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = sop_uid
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    ds.Modality = "CT"
    ds.StudyDate = ""
    ds.StudyTime = ""
    ds.SeriesDescription = "MedSim Simulation"

    # Image geometry
    rows, cols = volume_3d.shape[1], volume_3d.shape[2]
    ds.Rows = rows
    ds.Columns = cols
    ds.InstanceNumber = slice_idx + 1

    pixel_spacing_x = float(spacing[0])
    pixel_spacing_y = float(spacing[1])
    slice_thickness = float(spacing[2])

    ds.PixelSpacing = [pixel_spacing_y, pixel_spacing_x]  # DICOM: (row=y, col=x)
    ds.SliceThickness = slice_thickness

    # Image position: z position for this slice
    z_pos = float(origin[2]) + slice_idx * slice_thickness
    ds.ImagePositionPatient = [
        float(origin[0]),
        float(origin[1]),
        z_pos,
    ]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

    # Pixel data encoding
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Rescale (identity — pixel values are already HU)
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1

    # Pixel data
    slice_data = volume_3d[slice_idx]
    ds.PixelData = slice_data.tobytes()

    return ds


def export_dicom_zip(job: SimulationJob, storage: StorageBackend) -> StreamingResponse:
    """Convert result.nrrd to DICOM series zip and stream it."""
    temp_dir = tempfile.mkdtemp(prefix=f"export_{job.id}_dicom_")
    try:
        nrrd_path = download_nrrd_to_temp(storage, job.output_path, temp_dir)
        sitk_image = _load_sitk_image(nrrd_path)

        # Extract volume and metadata
        # SimpleITK stores as (x, y, z) internally, GetArrayFromImage returns (z, y, x)
        volume_3d = sitk.GetArrayFromImage(sitk_image)
        volume_int16 = _clip_to_int16(volume_3d)

        spacing = sitk_image.GetSpacing()   # (x, y, z)
        origin = sitk_image.GetOrigin()     # (x, y, z)
        num_slices = volume_int16.shape[0]

        # Generate UIDs shared across the series
        study_uid = generate_uid()
        series_uid = generate_uid()

        # Create DICOM zip
        zip_path = os.path.join(temp_dir, "dicom.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(num_slices):
                ds = _build_slice_dicom(
                    volume_3d=volume_int16,
                    slice_idx=i,
                    job_id=job.id,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    spacing=spacing,
                    origin=origin,
                )
                # Write DICOM to in-memory buffer
                buf = io.BytesIO()
                pydicom.dcmwrite(buf, ds, write_like_original=False)
                buf.seek(0)
                arcname = f"slice_{i + 1:04d}.dcm"
                zf.writestr(arcname, buf.getvalue())

        logger.info(
            "Job %s: created DICOM zip with %d slices (%.1f MB)",
            job.id, num_slices, os.path.getsize(zip_path) / (1024 * 1024),
        )

        filename = f"simulation_{job.id}_dicom.zip"
        return stream_file_response(
            file_path=zip_path,
            filename=filename,
            media_type="application/zip",
            temp_dir=temp_dir,
        )
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception as e:
        cleanup_temp_dir(temp_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DICOM export failed: {e}",
        )
