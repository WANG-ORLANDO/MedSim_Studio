"""
Segmentation API

RESTful endpoints for AI-powered organ and lesion segmentation.
Integrates with MONAI for automatic segmentation and provides
interactive refinement capabilities.

Supports:
- Job-based async segmentation with background task execution
- Multi-organ auto-segmentation (liver, kidney, lung, spleen, etc.)
- Lesion detection
- Interactive click-based refinement
- Single-slice mask retrieval for frontend overlay
- Result export (NRRD, NIfTI, DICOM SEG)
"""

import os
import uuid
import io
import tempfile
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

import numpy as np

from app.database.session import get_db, SessionLocal
from app.models.segmentation import SegmentationJob
from app.models.dicom import DicomInstance
from app.schemas.segmentation import (
    SegmentationJobResponse,
    SegmentationJobCreate,
    SliceMaskResponse,
    InteractiveClickRequest,
    InteractiveClickResponse,
    ModelInfoResponse,
    LabelDef,
)
from app.segmentation.ai.pipeline import run_full_segmentation
from app.segmentation.interactive.refiner import (
    refine_mask_on_click,
    extract_slice_mask,
)
from app.segmentation.export import stream_mask_from_storage
from app.dicom.storage import get_storage_backend
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/segment", tags=["Segmentation"])


# ---------------------------------------------------------------------------
# Background task: run segmentation job
# ---------------------------------------------------------------------------


def run_segmentation_job(job_id: str) -> None:
    """
    Execute a segmentation job in the background.

    Creates its own database session (does NOT reuse the request session)
    to avoid cross-thread / cross-request session reuse issues.

    Status flow:
        pending -> running -> completed / failed

    Pipeline:
        1. Load DICOM instances for the series
        2. Build 3D CT volume from instances
        3. Run MONAI model inference
        4. Optionally run lesion detection
        5. Save mask as NRRD to storage backend
        6. Mark job as completed
    """
    db: Session = SessionLocal()

    try:
        job = db.query(SegmentationJob).filter(SegmentationJob.id == job_id).first()
        if not job:
            logger.error("run_segmentation_job: job %s not found, exiting", job_id)
            return

        # --- Transition: pending -> running ---
        job.status = "running"
        job.progress = 10.0
        job.started_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()
        logger.info("Segmentation job %s transitioned to running", job_id)

        # --- Step 1: Load DICOM instances (progress -> 25) ---
        instances = (
            db.query(DicomInstance)
            .filter(DicomInstance.series_id == job.series_id)
            .order_by(DicomInstance.instance_number.asc().nulls_last())
            .all()
        )
        if not instances:
            raise ValueError(
                f"No DICOM instances found for series {job.series_id}"
            )

        job.progress = 25.0
        job.updated_at = datetime.utcnow()
        db.commit()
        logger.info(
            "Job %s: loaded %d DICOM instances for series %s",
            job_id, len(instances), job.series_id,
        )

        # --- Step 2: Run segmentation pipeline (progress 25 -> 75) ---
        storage = get_storage_backend()

        # Progress updates happen inside run_full_segmentation steps
        job.progress = 35.0
        job.updated_at = datetime.utcnow()
        db.commit()

        logger.info("[JOB] Job %s: calling run_full_segmentation...", job_id)
        import time as _time
        _t = _time.time()
        try:
            mask_key, seg_metadata = run_full_segmentation(
                job_id=job_id,
                storage=storage,
                instances=instances,
                model_name=job.model_name,
                target_organs=job.target_organs if job.target_organs else None,
                detect_lesions=job.detect_lesions or False,
            )
        except Exception as e:
            logger.error("[JOB] Job %s: run_full_segmentation FAILED: %s", job_id, e, exc_info=True)
            raise
        logger.info("[JOB] Job %s: run_full_segmentation done (%.1fs), mask_key=%s",
                    job_id, _time.time() - _t, mask_key)

        job.progress = 75.0
        job.updated_at = datetime.utcnow()
        db.commit()

        # --- Step 3: Create label map metadata (progress -> 90) ---
        from app.ai.totalsegmentator.labels import get_label_defs

        label_defs = get_label_defs(job.model_name, categories=False)
        label_map_json = {
            "labels": label_defs,
            "shape": seg_metadata.get("shape"),
            "spacing": seg_metadata.get("spacing"),
        }

        # Upload label map to storage
        label_key = f"segmentation/{job_id}/labels.json"
        label_bytes = str(label_map_json).encode("utf-8")

        # Write to temp file for storage backend
        tmp_label = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        import json
        json.dump(label_map_json, tmp_label)
        tmp_label_path = tmp_label.name
        tmp_label.close()

        try:
            storage.upload_file(
                object_key=label_key,
                file_path=tmp_label_path,
                content_type="application/json",
            )
        finally:
            os.unlink(tmp_label_path)

        job.progress = 90.0
        job.updated_at = datetime.utcnow()
        db.commit()

        # --- Step 4: Mark completed (progress -> 100) ---
        job.mask_path = mask_key
        job.label_map_path = label_key
        job.status = "completed"
        job.progress = 100.0
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()
        logger.info("Segmentation job %s completed: mask=%s", job_id, mask_key)

    except Exception as e:
        logger.exception("run_segmentation_job: unhandled error for job %s", job_id)
        try:
            db.rollback()
            job = db.query(SegmentationJob).filter(SegmentationJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = f"{type(e).__name__}: {str(e)[:200]}"
                job.updated_at = datetime.utcnow()
                db.commit()
                logger.info("Segmentation job %s marked as failed", job_id)
        except Exception:
            logger.exception(
                "run_segmentation_job: failed to update job %s to failed state",
                job_id,
            )
            try:
                db.rollback()
            except Exception:
                pass
    finally:
        db.close()


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("/jobs", response_model=SegmentationJobResponse, status_code=status.HTTP_201_CREATED)
async def create_segmentation_job(
    config: SegmentationJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Create a new segmentation job and enqueue it for background execution.

    The job starts with status='pending'. After the HTTP response is sent,
    FastAPI triggers run_segmentation_job() in the background, which
    transitions the job through running -> completed/failed.

    Requires a DICOM study/series to be loaded first.
    """
    job_id = str(uuid.uuid4())

    job = SegmentationJob(
        id=job_id,
        study_id=config.study_id,
        series_id=config.series_id,
        status="pending",
        model_name=config.model_name,
        target_organs=config.target_organs,
        detect_lesions=config.detect_lesions,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Enqueue background execution
    background_tasks.add_task(run_segmentation_job, job_id)

    return job


@router.get("/jobs", response_model=List[SegmentationJobResponse])
async def list_segmentation_jobs(
    study_id: Optional[str] = Query(None, description="Filter by study ID"),
    status: Optional[str] = Query(None, description="Filter by status (pending, running, completed, failed)"),
    db: Session = Depends(get_db),
):
    """List segmentation jobs with optional filtering."""
    query = db.query(SegmentationJob)

    if study_id:
        query = query.filter(SegmentationJob.study_id == study_id)
    if status:
        query = query.filter(SegmentationJob.status == status)

    jobs = query.order_by(SegmentationJob.created_at.desc()).all()
    return jobs


@router.get("/jobs/{job_id}", response_model=SegmentationJobResponse)
async def get_segmentation_job(job_id: str, db: Session = Depends(get_db)):
    """Get the status and details of a segmentation job."""
    job = db.query(SegmentationJob).filter(SegmentationJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Segmentation job {job_id} not found",
        )
    return job


@router.post("/jobs/{job_id}/cancel")
async def cancel_segmentation_job(job_id: str, db: Session = Depends(get_db)):
    """Cancel a running or pending segmentation job."""
    job = db.query(SegmentationJob).filter(SegmentationJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Segmentation job {job_id} not found",
        )
    if job.status not in ("pending", "running"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in status '{job.status}'",
        )
    job.status = "failed"
    job.error_message = "Cancelled by user"
    job.updated_at = datetime.utcnow()
    db.commit()
    return {"status": "cancelled", "job_id": job_id}


# ---------------------------------------------------------------------------
# Mask retrieval
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/mask")
async def get_segmentation_mask(
    job_id: str,
    db: Session = Depends(get_db),
):
    """
    Download the full 3D segmentation mask as an NRRD file.

    The mask contains integer label indices (0=background, 1=liver, etc.).
    """
    job = db.query(SegmentationJob).filter(SegmentationJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Segmentation job {job_id} not found",
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is not completed (current status: {job.status})",
        )

    if not job.mask_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job mask_path is empty — segmentation did not complete successfully",
        )

    storage = get_storage_backend()
    return stream_mask_from_storage(
        storage=storage,
        object_key=job.mask_path,
        filename=f"segmentation_{job_id}.nrrd",
        media_type="application/octet-stream",
    )


@router.get("/jobs/{job_id}/mask/slice/{z_index}", response_model=SliceMaskResponse)
async def get_slice_mask(
    job_id: str,
    z_index: int,
    db: Session = Depends(get_db),
):
    """
    Get a single 2D slice of the segmentation mask for overlay rendering.

    Returns the label indices as a 2D array along with label definitions
    so the frontend can color the overlay correctly.
    """
    job = db.query(SegmentationJob).filter(SegmentationJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Segmentation job {job_id} not found",
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is not completed (current status: {job.status})",
        )

    if not job.mask_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mask not available — segmentation did not complete",
        )

    # Read mask from storage backend
    storage = get_storage_backend()
    mask_bytes = storage.get_object_bytes(job.mask_path)
    if mask_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mask not found in storage: {job.mask_path}",
        )

    # Parse NRRD with SimpleITK — read from bytes via temp file
    import SimpleITK as sitk
    tmp = tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False)
    tmp.write(mask_bytes)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        sitk_image = sitk.ReadImage(tmp_path)
        mask_array: np.ndarray = sitk.GetArrayFromImage(sitk_image)  # (z, y, x)
    finally:
        os.unlink(tmp_path)

    # Validate z_index
    if z_index < 0 or z_index >= mask_array.shape[0]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"z_index {z_index} out of range (0-{mask_array.shape[0] - 1})",
        )

    # Extract slice
    slice_2d = mask_array[z_index].astype(np.int32)

    # Build label definitions from the correct model's label map
    from app.ai.totalsegmentator.labels import get_label_defs

    labels_response = [
        LabelDef(**item) for item in get_label_defs(job.model_name, categories=False)
    ]

    # Serialize 2D array as list of lists for JSON transport
    mask_data = slice_2d.tolist()

    return SliceMaskResponse(
        z_index=z_index,
        rows=slice_2d.shape[0],
        cols=slice_2d.shape[1],
        labels=labels_response,
        mask_data=mask_data,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/export")
async def export_segmentation_mask(
    job_id: str,
    format: str = Query("nrrd", description="Export format: nifti, nrrd, dicom_seg"),
    db: Session = Depends(get_db),
):
    """
    Export the segmentation mask in the specified format.

    Streams a downloadable file containing the full 3D label map.
    Formats:
      - nrrd: NRRD format (default)
      - nifti: NIfTI (.nii.gz)
      - dicom_seg: DICOM SEG object (.dcm)
    """
    job = db.query(SegmentationJob).filter(SegmentationJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Segmentation job {job_id} not found",
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is not completed (current status: {job.status})",
        )

    if not job.mask_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job mask_path is empty — segmentation did not complete successfully",
        )

    allowed_formats = ("nifti", "nrrd", "dicom_seg")
    if format not in allowed_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid format '{format}'. Allowed: {', '.join(allowed_formats)}",
        )

    storage = get_storage_backend()

    try:
        if format == "nrrd":
            return stream_mask_from_storage(
                storage=storage,
                object_key=job.mask_path,
                filename=f"segmentation_{job_id}.nrrd",
            )

        # For nifti and dicom_seg, we need to read and convert the mask
        mask_bytes = storage.get_object_bytes(job.mask_path)
        if mask_bytes is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Mask not found in storage",
            )

        import SimpleITK as sitk
        tmp = tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False)
        tmp.write(mask_bytes)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        try:
            sitk_image = sitk.ReadImage(tmp_path)
            mask_array = sitk.GetArrayFromImage(sitk_image)

            if format == "nifti":
                from app.segmentation.export import export_mask_nifti

                nifti_tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
                nifti_tmp.close()

                export_mask_nifti(
                    mask_array=mask_array,
                    output_path=nifti_tmp.name,
                    spacing=sitk_image.GetSpacing()[::-1],  # SimpleITK (x,y,z) -> our (z,y,x)
                    origin=sitk_image.GetOrigin()[::-1],
                )

                return FileResponse(
                    path=nifti_tmp.name,
                    media_type="application/gzip",
                    filename=f"segmentation_{job_id}.nii.gz",
                )

            elif format == "dicom_seg":
                # DICOM SEG export requires original DICOM series files
                from app.segmentation.export import export_mask_dicom_seg

                # We need a directory of source DICOM files. Since we use object storage,
                # DICOM SEG is marked as a preview feature and requires manual setup.
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="DICOM SEG export requires source DICOM files "
                           "on the local filesystem. Please use 'nrrd' or 'nifti' format.",
                )

        finally:
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Export failed for job %s, format %s", job_id, format)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)[:200]}",
        )


# ---------------------------------------------------------------------------
# Interactive refinement
# ---------------------------------------------------------------------------


@router.post("/interactive/click", response_model=InteractiveClickResponse)
async def interactive_click_refinement(
    request: InteractiveClickRequest,
    db: Session = Depends(get_db),
):
    """
    Refine a segmentation mask with a single click.

    Uses intensity-constrained region growing from the click point
    to add or remove a label in the local region.
    """
    job = db.query(SegmentationJob).filter(SegmentationJob.id == request.job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Segmentation job {request.job_id} not found",
        )

    if job.status != "completed" or not job.mask_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot refine a job that is not completed",
        )

    storage = get_storage_backend()

    # Read mask from storage
    mask_bytes = storage.get_object_bytes(job.mask_path)
    if mask_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mask not found in storage",
        )

    # Read mask NRRD
    import SimpleITK as sitk
    tmp = tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False)
    tmp.write(mask_bytes)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        sitk_image = sitk.ReadImage(tmp_path)
        mask_array = sitk.GetArrayFromImage(sitk_image).astype(np.int32)
    finally:
        os.unlink(tmp_path)

    # We also need the original CT volume for intensity-constrained growing.
    # Load it from DICOM instances.
    instances = (
        db.query(DicomInstance)
        .filter(DicomInstance.series_id == job.series_id)
        .order_by(DicomInstance.instance_number.asc().nulls_last())
        .all()
    )
    from app.simulation.volume_builder import build_volume_from_dicom
    volume, _ = build_volume_from_dicom(storage, instances)

    # Run refinement
    updated_mask, bbox = refine_mask_on_click(
        mask_array=mask_array,
        volume=volume,
        z=request.z_index,
        x=request.x,
        y=request.y,
        label=request.label,
        operation=request.operation,
    )

    # Extract updated patch for response
    zs, ze, ys, ye, xs, xe = bbox
    patch = updated_mask[zs:ze, ys:ye, xs:xe]

    # Save updated mask back to storage
    import tempfile as tf
    tmp_dir = tf.mkdtemp(prefix=f"seg_refine_{request.job_id}_")
    tmp_nrrd = os.path.join(tmp_dir, "mask.nrrd")
    try:
        from app.segmentation.export import export_mask_nrrd
        export_mask_nrrd(
            mask_array=updated_mask,
            output_path=tmp_nrrd,
            spacing=sitk_image.GetSpacing()[::-1],
            origin=sitk_image.GetOrigin()[::-1],
        )
        storage.upload_file(
            object_key=job.mask_path,
            file_path=tmp_nrrd,
            content_type="application/octet-stream",
        )
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(
        "Interactive refinement on job %s: %s label=%d at (%d, %d, %d)",
        request.job_id, request.operation, request.label,
        request.z_index, request.x, request.y,
    )

    return InteractiveClickResponse(
        z_index=request.z_index,
        updated_rows=patch.shape[0],
        updated_cols=patch.shape[1],
        patch_data=patch.tolist(),
    )


# ---------------------------------------------------------------------------
# Model / Label discovery
# ---------------------------------------------------------------------------


@router.get("/models", response_model=List[ModelInfoResponse])
async def list_available_models():
    """List available segmentation models and the organs they support."""
    from app.ai.totalsegmentator import is_available as ts_available
    from app.ai.nnunet_custom import is_available as nnunet_available
    from app.ai.nnunet_custom_20 import is_available as nnunet20_available
    return [
        ModelInfoResponse(
            name="totalsegmentator",
            description="TotalSegmentator v2 — pretrained segmentation of 117 anatomical structures "
                        "(organs, bones, muscles, vessels, glands). No training required.",
            organs=["all (117 structures)"],
            status="available" if ts_available() else "coming_soon",
        ),
        ModelInfoResponse(
            name="nnunet_handoff",
            description="Custom nnUNet (Dataset701_TotalSegOrgans6) — 6 organs: liver, kidney, "
                        "lung, spleen, pancreas, bladder",
            organs=["liver", "kidney", "lung", "spleen", "pancreas", "bladder"],
            status="available" if nnunet_available() else "coming_soon",
        ),
        ModelInfoResponse(
            name="nnunet702_20organs",
            description="Custom nnUNet (Dataset702_TotalSegOrgans20) — 20 anatomical structures: "
                        "liver, kidneys, lungs (5 lobes), spleen, pancreas, bladder, "
                        "adrenal glands, GI tract, gallbladder, trachea",
            organs=[
                "left_adrenal_gland", "right_adrenal_gland", "colon", "duodenum",
                "esophagus", "gallbladder", "left_kidney", "right_kidney", "liver",
                "left_lung_lower_lobe", "right_lung_lower_lobe", "right_lung_middle_lobe",
                "left_lung_upper_lobe", "right_lung_upper_lobe", "pancreas",
                "small_bowel", "spleen", "stomach", "trachea", "urinary_bladder",
            ],
            status="available" if nnunet20_available() else "coming_soon",
        ),
        ModelInfoResponse(
            name="unet",
            description="Custom-trained 3D U-Net for abdominal organ segmentation (liver, kidney, lung, spleen, pancreas)",
            organs=["liver", "kidney", "lung", "spleen", "pancreas"],
            status="available",
        ),
        ModelInfoResponse(
            name="segresnet",
            description="SegResNet for lesion segmentation (brain, liver lesions)",
            organs=["liver", "brain"],
            status="available",
        ),
        ModelInfoResponse(
            name="swin_unetr",
            description="Swin UNETR for whole-body segmentation",
            organs=["all"],
            status="available",
        ),
    ]


@router.get("/labels")
async def get_segmentation_labels(
    model_name: str = Query(
        "unet",
        description="Model name: 'totalsegmentator' for 117-class labels, 'unet' for legacy 10-class",
    ),
):
    """Get available segmentation label definitions with colors.

    Args:
        model_name: Select which model's label set to return.
            - 'totalsegmentator': Returns all 117 TotalSegmentator labels
            - 'unet' / others: Returns the legacy 10-class label map

    Returns:
        Dict with 'labels' list, each containing index, name, color,
        and for TotalSegmentator: category and category_label.
    """
    from app.ai.totalsegmentator.labels import get_label_defs

    include_categories = model_name and model_name.lower() == "totalsegmentator"
    labels = get_label_defs(model_name, categories=include_categories)
    return {"labels": labels}
