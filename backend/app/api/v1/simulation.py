"""
Simulation API

RESTful endpoints for lesion and organ simulation management.
Provides job creation, status tracking, preview generation,
CT phantom generation, and result export for synthetic medical image generation.
"""

import os
import io
import base64
import uuid
import tempfile
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from PIL import Image

import numpy as np

from app.database.session import get_db, SessionLocal
from app.models.simulation import SimulationJob, LesionConfig, OrganConfig
from app.models.dicom import DicomInstance, DicomSeries
from app.schemas.simulation import (
    SimulationJobResponse,
    SimulationJobCreate,
    LesionConfigResponse,
    SimulationPreviewResponse,
    DicomLesionPreviewRequest,
    DicomLesionPreviewResponse,
    CTParamsPreviewRequest,
    CTParamsPreviewResponse,
)
from app.simulation.lesion.generator import LesionGenerator
from app.simulation.organ.simulator import OrganSimulator
from app.simulation.volume_builder import build_volume_from_dicom, build_synthetic_volume
from app.simulation.exporter import export_nrrd, export_nifti, export_dicom_zip
from app.simulation.ct_params_simulator import simulate_ct_scan_params
from app.simulation.phantom_generator import (
    generate_atlas_ct_phantom,
    generate_procedural_ct_phantom,
)
from app.dicom.storage import get_storage_backend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulation", tags=["Simulation"])
DEFAULT_STANDARDIZED_NOTES = [
    "axis_order = zyx",
    "dtype = float32",
    "spacing order = z,y,x",
    "volume data is stored in top-level simulated_volume_base64",
    "standardized_case is intended for downstream artifact/lesion modules",
]


def _identity_direction_matrix() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _normalize_origin(origin: Optional[Any]) -> List[float]:
    if isinstance(origin, (list, tuple)) and len(origin) >= 3:
        try:
            return [float(origin[0]), float(origin[1]), float(origin[2])]
        except (TypeError, ValueError):
            pass
    return [0.0, 0.0, 0.0]


def _normalize_direction(direction: Optional[Any]) -> List[List[float]]:
    if isinstance(direction, (list, tuple)):
        if len(direction) == 9:
            try:
                values = [float(v) for v in direction]
                return [values[0:3], values[3:6], values[6:9]]
            except (TypeError, ValueError):
                return _identity_direction_matrix()
        if len(direction) == 3 and all(isinstance(row, (list, tuple)) and len(row) >= 3 for row in direction):
            try:
                return [[float(row[0]), float(row[1]), float(row[2])] for row in direction]
            except (TypeError, ValueError):
                return _identity_direction_matrix()
    return _identity_direction_matrix()


def _build_standardized_ct_case(
    *,
    source: str,
    source_case_id: Optional[str],
    simulated_volume: np.ndarray,
    spacing: tuple[float, float, float],
    params_json: Dict[str, Any],
    metadata: Dict[str, Any],
    origin: Optional[Any] = None,
    direction: Optional[Any] = None,
    body_part: Optional[str] = None,
) -> Dict[str, Any]:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    hu_range = metadata.get("hu_range") or [
        float(np.min(simulated_volume)),
        float(np.max(simulated_volume)),
    ]
    case_id = (
        f"sim_{source}_{source_case_id}_{timestamp}"
        if source_case_id
        else f"sim_{source}_{timestamp}"
    )
    return {
        "case_id": case_id,
        "source": source,
        "source_case_id": source_case_id,
        "volume": {
            "encoding": "base64",
            "dtype": "float32",
            "byte_order": "little_endian",
            "axis_order": "zyx",
            "shape": [int(v) for v in simulated_volume.shape],
            "spacing": [float(v) for v in spacing],
            "origin": _normalize_origin(origin),
            "direction": _normalize_direction(direction),
            "hu_range": [float(hu_range[0]), float(hu_range[1])],
            "slice_count": int(simulated_volume.shape[0]),
            "modality": "CT",
            "body_part": body_part or "unknown",
            "image_kind": "simulated_ct",
            "image_data_field": "simulated_volume_base64",
        },
        "simulation": {
            "type": "ct_scan_params",
            "params_json": params_json,
            "algorithm": "image_domain_approximation",
            "approximation_warning": "This is an educational image-domain approximation, not a scanner-physics simulation.",
        },
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Background task: run simulation job
# ---------------------------------------------------------------------------


def run_simulation_job(job_id: str) -> None:
    """
    Execute a simulation job in the background.

    Creates its own database session (does NOT reuse the request session)
    to avoid cross-thread / cross-request session reuse issues.

    Status flow:
        pending -> running -> completed / failed

    Phase 2 implementation:
      1. Build base volume (from DICOM source or synthetic fallback)
      2. Apply LesionGenerator / OrganSimulator overlays
      3. Write result as NRRD via SimpleITK to a temp file
      4. Upload temp file to storage backend
      5. Set job.output_path = "simulation/{job_id}/result.nrrd"
      6. Clean up temp files

    output_path always holds a storage object_key (never a local path).
    """
    import SimpleITK as sitk
    import numpy as np

    db: Session = SessionLocal()
    temp_nrrd_path: Optional[str] = None

    try:
        job = db.query(SimulationJob).filter(SimulationJob.id == job_id).first()
        if not job:
            logger.error("run_simulation_job: job %s not found, exiting", job_id)
            return

        # --- Transition: pending -> running ---
        job.status = "running"
        job.progress = 10.0
        job.started_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()
        logger.info("Job %s transitioned to running", job_id)

        # --- Step 1: Build base volume (progress -> 25) ---
        storage = get_storage_backend()
        volume: Optional[np.ndarray] = None
        metadata: Optional[dict] = None

        if job.series_id:
            # Try to read source DICOM from storage backend
            instances = (
                db.query(DicomInstance)
                .filter(DicomInstance.series_id == job.series_id)
                .order_by(DicomInstance.instance_number.asc().nulls_last())
                .all()
            )
            if instances:
                try:
                    volume, metadata = build_volume_from_dicom(storage, instances)
                    logger.info(
                        "Job %s: built volume from DICOM series %s (%d slices)",
                        job_id, job.series_id, len(instances),
                    )
                except Exception as e:
                    logger.warning(
                        "Job %s: failed to build volume from DICOM (%s), "
                        "falling back to synthetic",
                        job_id, e,
                    )
                    volume, metadata = None, None

        if volume is None:
            # No source DICOM or read failed 鈥?generate synthetic base volume
            volume, metadata = build_synthetic_volume()
            logger.info("Job %s: using synthetic base volume", job_id)

        job.progress = 25.0
        job.updated_at = datetime.utcnow()
        db.commit()

        # --- Step 2: Apply simulation overlays (progress -> 50) ---
        result_volume = volume.copy()

        # Apply lesions
        lesion_configs = (
            db.query(LesionConfig)
            .filter(LesionConfig.job_id == job_id)
            .all()
        )
        if lesion_configs:
            lesion_gen = LesionGenerator()
            spacing = metadata.get("spacing")
            for lc in lesion_configs:
                # Normalize center: if all zeros (frontend default), place at volume center
                cz, cy, cx = lc.center_z, lc.center_y, lc.center_x
                if cz == 0.0 and cy == 0.0 and cx == 0.0:
                    cz = float(result_volume.shape[0] // 2)
                    cy = float(result_volume.shape[1] // 2)
                    cx = float(result_volume.shape[2] // 2)
                config_dict = {
                    "lesion_type": lc.lesion_type,
                    "shape": lc.shape,
                    "center_x": cx,
                    "center_y": cy,
                    "center_z": cz,
                    "radius_x": lc.radius_x,
                    "radius_y": lc.radius_y,
                    "radius_z": lc.radius_z,
                    "hu_mean": lc.hu_mean,
                    "hu_std": lc.hu_std,
                    "margin_sharpness": lc.margin_sharpness,
                    "calcification_fraction": lc.calcification_fraction,
                    "necrosis_fraction": lc.necrosis_fraction,
                    "spiculation_degree": lc.spiculation_degree,
                }
                lesion_vol = lesion_gen.generate_lesion(
                    volume_shape=result_volume.shape,
                    config=config_dict,
                    spacing=spacing,
                )
                result_volume = result_volume + lesion_vol
                logger.info(
                    "Job %s: applied lesion %s (%s)",
                    job_id, lc.id, lc.lesion_type,
                )

        # Apply organs
        organ_configs = (
            db.query(OrganConfig)
            .filter(OrganConfig.job_id == job_id)
            .all()
        )
        if organ_configs:
            organ_sim = OrganSimulator()
            for oc in organ_configs:
                config_dict = {
                    "organ_type": oc.organ_type,
                    "hu_mean": oc.hu_mean,
                    "hu_std": oc.hu_std,
                    "enable_noise": oc.enable_noise,
                    "noise_level": oc.noise_level,
                    "enable_enhancement": oc.enable_enhancement,
                    "enhancement_pattern": oc.enhancement_pattern,
                }
                organ_vol = organ_sim.generate_organ(
                    volume_shape=result_volume.shape,
                    config=config_dict,
                )
                # Only add organ where result is still background (avoid overwriting lesions)
                organ_mask = organ_vol != 0
                result_volume[organ_mask] = organ_vol[organ_mask]
                logger.info(
                    "Job %s: applied organ %s (%s)",
                    job_id, oc.id, oc.organ_type,
                )

        job.progress = 50.0
        job.updated_at = datetime.utcnow()
        db.commit()

        # --- Step 3: Write NRRD to temp file (progress -> 75) ---
        spacing = metadata.get("spacing", (1.0, 1.0, 1.0))
        origin = metadata.get("origin", (0.0, 0.0, 0.0))
        direction = metadata.get("direction", (1, 0, 0, 0, 1, 0, 0, 0, 1))

        # SimpleITK expects spacing/origin in (x, y, z) order, but our
        # volume is (z, y, x). Reverse for SimpleITK compatibility.
        sitk_spacing = (spacing[2], spacing[1], spacing[0])
        sitk_origin = (origin[2], origin[1], origin[0])

        sitk_image = sitk.GetImageFromArray(result_volume.astype(np.float32))
        sitk_image.SetSpacing(sitk_spacing)
        sitk_image.SetOrigin(sitk_origin)
        if len(direction) == 9:
            sitk_image.SetDirection(direction)

        # Write to temp file
        temp_dir = tempfile.mkdtemp(prefix=f"sim_{job_id}_")
        temp_nrrd_path = os.path.join(temp_dir, "result.nrrd")
        sitk.WriteImage(sitk_image, temp_nrrd_path)
        logger.info(
            "Job %s: wrote temp NRRD %s (%.1f MB)",
            job_id, temp_nrrd_path,
            os.path.getsize(temp_nrrd_path) / (1024 * 1024),
        )

        job.progress = 75.0
        job.updated_at = datetime.utcnow()
        db.commit()

        # --- Step 4: Upload to storage backend (progress -> 90) ---
        object_key = f"simulation/{job_id}/result.nrrd"
        upload_ok = storage.upload_file(
            object_key=object_key,
            file_path=temp_nrrd_path,
            content_type="application/octet-stream",
        )
        if not upload_ok:
            raise RuntimeError(
                f"Failed to upload result NRRD to storage (key={object_key})"
            )
        logger.info("Job %s: uploaded result to storage key=%s", job_id, object_key)

        job.progress = 90.0
        job.updated_at = datetime.utcnow()
        db.commit()

        # --- Step 5: Mark completed with output_path (progress -> 100) ---
        job.output_path = object_key
        job.output_format = "nrrd"
        job.status = "completed"
        job.progress = 100.0
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        db.commit()
        logger.info("Job %s completed: output_path=%s", job_id, object_key)

    except Exception as e:
        logger.exception("run_simulation_job: unhandled error for job %s", job_id)
        try:
            db.rollback()
            job = db.query(SimulationJob).filter(SimulationJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = f"{type(e).__name__}: {str(e)[:200]}"
                job.updated_at = datetime.utcnow()
                # output_path stays None on failure 鈥?never write a fake value
                db.commit()
                logger.info("Job %s marked as failed", job_id)
        except Exception:
            logger.exception(
                "run_simulation_job: failed to update job %s to failed state",
                job_id,
            )
            try:
                db.rollback()
            except Exception:
                pass
    finally:
        # Clean up temp files
        if temp_nrrd_path:
            try:
                temp_dir = os.path.dirname(temp_nrrd_path)
                if os.path.isdir(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info("Job %s: cleaned up temp dir %s", job_id, temp_dir)
            except Exception:
                logger.warning("Job %s: failed to clean temp dir", job_id)

        db.close()


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("/jobs", response_model=SimulationJobResponse, status_code=status.HTTP_201_CREATED)
async def create_simulation_job(
    config: SimulationJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Create a new simulation job and enqueue it for background execution.

    The job starts with status='pending'. After the HTTP response is sent,
    FastAPI triggers run_simulation_job() in the background, which
    transitions the job through running -> completed/failed.
    """
    job_id = str(uuid.uuid4())

    job = SimulationJob(
        id=job_id,
        study_id=config.study_id,
        series_id=config.series_id,
        status="pending",
        lesion_count=len(config.lesions),
        organ_count=len(config.organs),
        output_format=config.output_format,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(job)

    # Create lesion configurations
    for i, lesion_config in enumerate(config.lesions):
        lesion = LesionConfig(
            id=str(uuid.uuid4()),
            job_id=job_id,
            lesion_type=lesion_config.lesion_type,
            shape=lesion_config.shape,
            center_x=lesion_config.center_x,
            center_y=lesion_config.center_y,
            center_z=lesion_config.center_z,
            radius_x=lesion_config.radius_x,
            radius_y=lesion_config.radius_y,
            radius_z=lesion_config.radius_z,
            hu_mean=lesion_config.hu_mean,
            hu_std=lesion_config.hu_std,
            margin_sharpness=lesion_config.margin_sharpness,
            calcification_fraction=lesion_config.calcification_fraction,
            necrosis_fraction=lesion_config.necrosis_fraction,
            spiculation_degree=lesion_config.spiculation_degree,
        )
        db.add(lesion)

    db.commit()
    db.refresh(job)

    # Enqueue background execution 鈥?job_id is a plain string,
    # the background task creates its own DB session.
    background_tasks.add_task(run_simulation_job, job_id)

    return job


@router.get("/jobs", response_model=List[SimulationJobResponse])
async def list_simulation_jobs(
    study_id: Optional[str] = Query(None, description="Filter by study"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db),
):
    """List simulation jobs with optional filtering."""
    query = db.query(SimulationJob)

    if study_id:
        query = query.filter(SimulationJob.study_id == study_id)
    if status_filter:
        query = query.filter(SimulationJob.status == status_filter)

    jobs = query.order_by(SimulationJob.created_at.desc()).all()
    return jobs


@router.get("/jobs/{job_id}", response_model=SimulationJobResponse)
async def get_simulation_job(job_id: str, db: Session = Depends(get_db)):
    """Get the status and details of a simulation job."""
    job = db.query(SimulationJob).filter(SimulationJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation job {job_id} not found",
        )
    return job


@router.post("/jobs/{job_id}/cancel")
async def cancel_simulation_job(job_id: str, db: Session = Depends(get_db)):
    """Cancel a running simulation job."""
    job = db.query(SimulationJob).filter(SimulationJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation job {job_id} not found",
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
# Preview
# ---------------------------------------------------------------------------


@router.post("/preview/lesion", response_model=SimulationPreviewResponse)
async def preview_lesion(config: dict):
    """
    Generate a fast preview of a lesion configuration.

    Synchronous endpoint for real-time preview of lesion parameters
    without creating a full simulation job.
    """
    try:
        generator = LesionGenerator()
        preview = generator.generate_preview(config)
        return SimulationPreviewResponse(
            job_id=str(uuid.uuid4()),
            preview_data=preview,
            voxel_count=preview.get("voxel_count", 0),
            hu_range=(preview.get("hu_min", 0), preview.get("hu_max", 0)),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preview generation failed: {str(e)}",
        )


@router.post("/preview/lesion-on-dicom", response_model=DicomLesionPreviewResponse)
async def preview_lesion_on_dicom(
    request: DicomLesionPreviewRequest,
    db: Session = Depends(get_db),
):
    """
    Preview a lesion overlaid on a real DICOM series.

    Loads the DICOM series from storage, generates the lesion on it,
    and returns a base64-encoded side-by-side PNG (original | with lesion)
    of the axial slice through the lesion center.

    Synchronous 鈥?intended for interactive parameter tuning.
    """
    try:
        # 鈹€鈹€ 1. Load DICOM instances 鈹€鈹€
        instances = (
            db.query(DicomInstance)
            .filter(DicomInstance.series_id == request.series_id)
            .order_by(DicomInstance.instance_number.asc().nulls_last())
            .all()
        )
        if not instances:
            raise HTTPException(status_code=404, detail=f"Series {request.series_id} not found")

        storage = get_storage_backend()
        volume, metadata = build_volume_from_dicom(storage, instances)

        # 鈹€鈹€ 2. Generate lesion 鈹€鈹€
        generator = LesionGenerator()
        lesion_cfg = request.lesion

        # Normalize center: if all zeros (frontend default), place at volume center
        cz, cy, cx = lesion_cfg.center_z, lesion_cfg.center_y, lesion_cfg.center_x
        if cz == 0.0 and cy == 0.0 and cx == 0.0:
            cz = float(volume.shape[0] // 2)
            cy = float(volume.shape[1] // 2)
            cx = float(volume.shape[2] // 2)

        config_dict = {
            "lesion_type": lesion_cfg.lesion_type,
            "shape": lesion_cfg.shape,
            "center_x": cx,
            "center_y": cy,
            "center_z": cz,
            "radius_x": lesion_cfg.radius_x,
            "radius_y": lesion_cfg.radius_y,
            "radius_z": lesion_cfg.radius_z,
            "hu_mean": lesion_cfg.hu_mean,
            "hu_std": lesion_cfg.hu_std,
            "margin_sharpness": lesion_cfg.margin_sharpness,
            "calcification_fraction": lesion_cfg.calcification_fraction,
            "necrosis_fraction": lesion_cfg.necrosis_fraction,
            "spiculation_degree": lesion_cfg.spiculation_degree,
        }
        spacing = metadata.get("spacing")
        lesion_vol = generator.generate_lesion(
            volume_shape=volume.shape,
            config=config_dict,
            spacing=spacing,
        )
        result_volume = volume + lesion_vol

        # 鈹€鈹€ 3. Compute lesion region stats 鈹€鈹€
        lesion_mask = lesion_vol != 0
        if lesion_mask.any():
            hu_values = result_volume[lesion_mask]
            stats_voxels = int(np.sum(lesion_mask))
            stats_hu_min = float(np.min(hu_values))
            stats_hu_max = float(np.max(hu_values))
            stats_hu_mean = float(np.mean(hu_values))
            stats_hu_std = float(np.std(hu_values))
            spacing = metadata.get("spacing", (1.0, 1.0, 1.0))
            voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
            stats_volume_mm3 = float(stats_voxels * voxel_vol_mm3)
        else:
            stats_voxels = 0
            stats_hu_min = stats_hu_max = stats_hu_mean = stats_hu_std = 0.0
            stats_volume_mm3 = 0.0

        # 鈹€鈹€ 4. Find the slice closest to the lesion center 鈹€鈹€
        cz_idx = int(round(cz))
        cz_idx = max(0, min(cz_idx, volume.shape[0] - 1))

        # 鈹€鈹€ 5. Render side-by-side preview PNG 鈹€鈹€
        wc = request.window_center
        ww = request.window_width
        half = ww / 2.0
        lower = wc - half
        upper = wc + half

        def slice_to_8bit(slice_2d: np.ndarray) -> np.ndarray:
            return np.clip((slice_2d - lower) / ww * 255, 0, 255).astype(np.uint8)

        before_slice = slice_to_8bit(volume[cz_idx, :, :])
        after_slice = slice_to_8bit(result_volume[cz_idx, :, :])

        # Side-by-side
        h, w = before_slice.shape
        combined = np.zeros((h, w * 2 + 4), dtype=np.uint8)
        combined[:, :w] = before_slice
        # Divider (light gray line)
        combined[:, w:w + 4] = 160
        combined[:, w + 4:] = after_slice

        # Add labels (original / with lesion) 鈥?simple pixel-based label
        # "Original" label at top-left
        combined[4:10, 6:6 + 8 * 6] = 200  # placeholder bar
        # "With Lesion" label at top-right
        combined[4:10, w + 10:w + 10 + 11 * 6] = 200

        img = Image.fromarray(combined, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return DicomLesionPreviewResponse(
            image_base64=b64,
            slice_index=cz_idx,
            total_slices=volume.shape[0],
            lesion_center_voxel=[cz_idx, int(round(cy)), int(round(cx))],
            hu_min=stats_hu_min,
            hu_max=stats_hu_max,
            hu_mean=stats_hu_mean,
            hu_std=stats_hu_std,
            voxel_count=stats_voxels,
            volume_mm3=stats_volume_mm3,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("DICOM lesion preview failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DICOM lesion preview failed: {str(e)[:300]}",
        )


@router.post("/preview/organ", response_model=SimulationPreviewResponse)
async def preview_organ(config: dict):
    """
    Generate a fast preview of an organ configuration.

    Synchronous endpoint for real-time preview of organ parameters
    without creating a full simulation job.
    """
    try:
        simulator = OrganSimulator()
        preview = simulator.generate_preview(config)
        return SimulationPreviewResponse(
            job_id=str(uuid.uuid4()),
            preview_data=preview,
            voxel_count=preview.get("voxel_count", 0),
            hu_range=(preview.get("hu_min", 0), preview.get("hu_max", 0)),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preview generation failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# CT Phantom 鈥?synthetic upper-body CT volume for frontend demo
# ---------------------------------------------------------------------------


@router.post("/ct-params/preview", response_model=CTParamsPreviewResponse)
async def preview_ct_scan_params(
    request: CTParamsPreviewRequest,
    db: Session = Depends(get_db),
):
    """
    Generate a CT parameter preview for atlas, procedural, or uploaded DICOM CT volumes.

    The frontend already has access to the original phantom volume, so this
    endpoint returns only the simulated volume plus metadata and params_json.
    """
    try:
        if request.source not in {"atlas", "procedural", "dicom"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported source for CT parameter preview: {request.source}",
            )

        label_volume = None
        source_origin: Optional[Any] = None
        source_direction: Optional[Any] = None
        body_part = "upper_body"
        standardized_notes = list(DEFAULT_STANDARDIZED_NOTES)
        extra_metadata: Dict[str, Any] = {}

        if request.source == "atlas":
            source_case_id = request.case_id or "s0001"
            ct_volume, label_volume, phantom_metadata = generate_atlas_ct_phantom(
                case_id=source_case_id,
                size=request.size,
                scan_direction=request.scan_direction,
            )
            source_spacing = tuple(phantom_metadata.get("spacing", (1.0, 1.0, 1.0)))
            standardized_notes.extend([
                "origin uses default [0, 0, 0] because atlas origin is not propagated in this preview response.",
                "direction uses identity matrix because atlas direction is not propagated in this preview response.",
            ])
            extra_metadata = {
                "case_id": source_case_id,
                "scan_direction": request.scan_direction,
                "phantom_metadata": {
                    "original_shape": phantom_metadata.get("original_shape"),
                    "output_shape": phantom_metadata.get("output_shape"),
                    "original_spacing": phantom_metadata.get("original_spacing"),
                    "output_spacing": phantom_metadata.get("output_spacing"),
                    "flipped_z": phantom_metadata.get("flipped_z"),
                },
            }
        elif request.source == "procedural":
            source_case_id = "procedural"
            ct_volume, label_volume, phantom_metadata = generate_procedural_ct_phantom(
                size=request.size,
            )
            source_spacing = tuple(phantom_metadata.get("spacing", (1.0, 1.0, 1.0)))
            standardized_notes.extend([
                "origin uses default [0, 0, 0] because procedural phantom origin is not propagated in this preview response.",
                "direction uses identity matrix because procedural phantom direction is not propagated in this preview response.",
            ])
            extra_metadata = {
                "case_id": None,
                "scan_direction": request.scan_direction,
                "phantom_metadata": {
                    "original_shape": phantom_metadata.get("original_shape"),
                    "output_shape": phantom_metadata.get("output_shape"),
                    "original_spacing": phantom_metadata.get("original_spacing"),
                    "output_spacing": phantom_metadata.get("output_spacing"),
                    "flipped_z": phantom_metadata.get("flipped_z"),
                },
            }
        else:
            if not request.series_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="series_id is required when source='dicom'",
                )

            series_query = db.query(DicomSeries).filter(DicomSeries.id == request.series_id)
            if request.study_id:
                series_query = series_query.filter(DicomSeries.study_id == request.study_id)
            series = series_query.first()
            if not series:
                if request.study_id:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Series {request.series_id} not found in study {request.study_id}",
                    )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Series {request.series_id} not found",
                )

            if series.modality and str(series.modality).upper() != "CT":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Series {request.series_id} modality '{series.modality}' is not supported for CT parameter preview",
                )

            instances = (
                db.query(DicomInstance)
                .filter(DicomInstance.series_id == series.id)
                .order_by(DicomInstance.instance_number.asc().nulls_last())
                .all()
            )
            if not instances:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No DICOM instances found for series {series.id}",
                )

            storage = get_storage_backend()
            try:
                ct_volume, dicom_metadata = build_volume_from_dicom(storage, instances)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to build DICOM volume for series {series.id}: {str(e)}",
                ) from e
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to build DICOM volume for series {series.id}: {str(e)[:300]}",
                ) from e

            source_case_id = series.id
            source_spacing = tuple(dicom_metadata.get("spacing", (1.0, 1.0, 1.0)))
            source_origin = dicom_metadata.get("origin")
            source_direction = dicom_metadata.get("direction")
            body_part = series.body_part_examined or "unknown"

            if not series.modality:
                standardized_notes.append(
                    "DICOM series modality is missing in metadata; CT compatibility could not be verified beyond HU reconstruction."
                )
            if source_origin is None or _normalize_origin(source_origin) == [0.0, 0.0, 0.0]:
                standardized_notes.append(
                    "origin defaults to [0, 0, 0] or partial z-only origin because complete DICOM patient-space origin is not fully reconstructed by the current volume builder."
                )
            if source_direction is None or _normalize_direction(source_direction) == _identity_direction_matrix():
                standardized_notes.append(
                    "direction uses identity because the current volume builder does not reconstruct patient orientation from DICOM ImageOrientationPatient."
                )
            if not series.body_part_examined:
                standardized_notes.append(
                    "body_part defaults to 'unknown' because BodyPartExamined is missing in DICOM metadata."
                )

            extra_metadata = {
                "case_id": None,
                "study_id": series.study_id,
                "series_id": series.id,
                "dicom_metadata": {
                    "series_instance_uid": series.series_instance_uid,
                    "series_number": series.series_number,
                    "series_description": series.series_description,
                    "modality": series.modality,
                    "body_part_examined": series.body_part_examined,
                    "image_count": series.image_count,
                    "instance_count": len(instances),
                    "builder_num_slices": dicom_metadata.get("num_slices"),
                },
            }

        simulation_result = simulate_ct_scan_params(
            volume=ct_volume,
            spacing=source_spacing,
            params=request.params.model_dump(),
            label_volume=label_volume,
        )

        simulated_volume = simulation_result["simulated_volume"]
        simulated_spacing = tuple(
            simulation_result.get(
                "simulated_spacing",
                extra_metadata.get("phantom_metadata", {}).get("output_spacing", source_spacing),
            )
        )
        metadata = {
            **simulation_result["metadata"],
            "source": request.source,
            "preview_stats": simulation_result["preview_stats"],
            **extra_metadata,
            "notes": standardized_notes,
        }
        params_json = {
            **simulation_result["params_json"],
            "notes": standardized_notes,
        }
        standardized_case = _build_standardized_ct_case(
            source=request.source,
            source_case_id=source_case_id,
            simulated_volume=simulated_volume,
            spacing=simulated_spacing,
            params_json=params_json,
            metadata=metadata,
            origin=source_origin,
            direction=source_direction,
            body_part=body_part,
        )

        volume_b64 = base64.b64encode(
            np.asarray(simulated_volume, dtype="<f4").tobytes()
        ).decode("ascii")

        return CTParamsPreviewResponse(
            simulated_volume_base64=volume_b64,
            metadata=metadata,
            params_json=params_json,
            standardized_case=standardized_case,
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("CT parameter preview failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CT parameter preview failed: {str(e)[:300]}",
        )


@router.get("/phantom")
async def generate_ct_phantom(
    source: str = Query("procedural", description="Phantom source: 'procedural' or 'atlas'"),
    size: int = Query(192, ge=64, le=320, description="Volume max edge size in voxels"),
    case_id: str = Query("s0001", description="Atlas case ID (only used when source='atlas')"),
    scan_direction: str = Query(
        "head_to_feet",
        description="Z-axis scan direction: 'head_to_feet' (z=0=head/chest) or 'feet_to_head'",
    ),
):
    """
    Generate a CT phantom and return it as base64-encoded volume data.

    Two sources are supported:

    - **procedural** (default):
      Synthetic upper-body CT phantom built from geometric primitives
      (ellipses, arcs).  NOT a real medical image 鈥?suitable for UI
      development and demo.

    - **atlas**:
      Loads a real CT volume from models/phantom_atlas/{case_id}/.
      The CT is resampled so its largest dimension 鈮?size.  If an
      organs_label.nii.gz file is present it is converted to a uint8
      label map and returned alongside the CT volume.

      The z-axis direction is auto-detected from the NIfTI affine header.
      If it doesn't match `scan_direction` (default: head_to_feet), the
      volume is flipped so z=0 corresponds to the head/chest.

    Returns:
        JSON with:
        - volumeBase64:  base64-encoded raw Float32 bytes (little-endian)
        - labelBase64:   base64-encoded raw Uint8 bytes (optional, atlas only)
        - metadata:      {width, height, depth, spacing, source, case_id?,
                          originalShape, outputShape, originalSpacing,
                          outputSpacing, scanAxis, scanDirection, flippedZ,
                          labelNonzeroCounts?, sliceLabelPresence?,
                          label_map?, windowPresets, bodyThresholdHU}
    """
    try:
        if source == "atlas":
            ct_volume, label_volume, metadata = generate_atlas_ct_phantom(
                case_id=case_id,
                size=size,
                scan_direction=scan_direction,
            )

            # Encode CT volume
            raw_bytes = ct_volume.astype(np.float32).tobytes()
            volume_b64 = base64.b64encode(raw_bytes).decode("ascii")

            response_content: dict = {
                "volume_base64": volume_b64,
                "metadata": metadata,
            }

            # Encode label volume (if present)
            if label_volume is not None:
                label_bytes = label_volume.astype(np.uint8).tobytes()
                label_b64 = base64.b64encode(label_bytes).decode("ascii")
                response_content["label_base64"] = label_b64
            else:
                response_content["label_base64"] = None

            return JSONResponse(content=response_content)

        else:
            # --- procedural (default) ---
            volume, _, metadata = generate_procedural_ct_phantom(size=size)

            raw_bytes = volume.astype(np.float32).tobytes()
            volume_b64 = base64.b64encode(raw_bytes).decode("ascii")

            return JSONResponse(
                content={
                    "volume_base64": volume_b64,
                    "label_base64": None,
                    "metadata": metadata,
                }
            )

    except FileNotFoundError as e:
        logger.warning("Phantom generation 鈥?file not found: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Failed to generate CT phantom")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Phantom generation failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Export (placeholder 鈥?Phase 3 will implement real logic)
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/export")
async def export_simulation_results(
    job_id: str,
    format: str = Query("dicom", description="Export format: dicom, nifti, nrrd"),
    db: Session = Depends(get_db),
):
    """
    Export simulation results in the specified format.

    Streams downloadable DICOM (zip), NIfTI (.nii.gz), or NRRD files
    containing the simulated lesions and organs.

    Returns:
        StreamingResponse with Content-Disposition attachment
    """
    job = db.query(SimulationJob).filter(SimulationJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation job {job_id} not found",
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is not completed (current status: {job.status})",
        )

    if not job.output_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job output_path is empty 鈥?simulation did not complete successfully",
        )

    allowed_formats = ("dicom", "nifti", "nrrd")
    if format not in allowed_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid format '{format}'. Allowed: {', '.join(allowed_formats)}",
        )

    storage = get_storage_backend()

    try:
        if format == "nrrd":
            return export_nrrd(job, storage)
        elif format == "nifti":
            return export_nifti(job, storage)
        elif format == "dicom":
            return export_dicom_zip(job, storage)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Export failed for job %s, format %s", job_id, format)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)[:200]}",
        )


