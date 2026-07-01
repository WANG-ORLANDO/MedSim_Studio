"""
Segmentation Pipeline Orchestration

Combines MONAI model loading, inference, and postprocessing into
a single workflow. Called by the background task runner.
"""

import os
import time
import logging
import tempfile
import shutil
from typing import List, Optional, Tuple, Dict

import numpy as np

from app.ai.monai import (
    SegmentationModelManager,
    ModelNotAvailableError,
    run_segmentation,
    run_lesion_detection,
    ORGAN_LABEL_MAP,
)
from app.dicom.storage.base import StorageBackend
from app.simulation.volume_builder import build_volume_from_dicom
from app.models.dicom import DicomInstance
from app.segmentation.export import export_mask_nrrd
from app.core.config import settings

logger = logging.getLogger(__name__)


def run_full_segmentation(
    job_id: str,
    storage: StorageBackend,
    instances: List[DicomInstance],
    model_name: str = "unet",
    target_organs: Optional[List[str]] = None,
    detect_lesions: bool = False,
) -> Tuple[str, dict]:
    """
    Execute the full segmentation pipeline for a job.

    1. Build CT volume from DICOM instances
    2. Run MONAI model inference
    3. Optionally run lesion detection
    4. Save mask as NRRD to storage backend
    5. Return (object_key, metadata)

    Args:
        job_id: Segmentation job ID (for storage key naming)
        storage: StorageBackend instance
        instances: Sorted list of DicomInstance objects
        model_name: Model identifier
        target_organs: Organs to segment; None = all
        detect_lesions: Whether to run lesion detection

    Returns:
        Tuple of (mask_object_key, metadata_dict)

    Raises:
        ModelNotAvailableError: If MONAI/torch not installed
        ValueError: If DICOM volume cannot be built
    """
    import time as _time
    _t0 = _time.time()
    logger.info("[PIPELINE] Job %s: START (model=%s, detect_lesions=%s)", job_id, model_name, detect_lesions)

    # Step 1: Build volume from DICOM instances
    _t1 = _time.time()
    try:
        volume, metadata = build_volume_from_dicom(storage, instances)
    except Exception as e:
        logger.error("[PIPELINE] Job %s: build_volume_from_dicom FAILED: %s", job_id, e, exc_info=True)
        raise
    logger.info(
        "[PIPELINE] Job %s: built volume shape=%s spacing=%s (%.1fs)",
        job_id, volume.shape, metadata.get("spacing"), _time.time() - _t1,
    )
    logger.info("[PIPELINE] Job %s: volume dtype=%s min=%.1f max=%.1f mean=%.1f",
                job_id, volume.dtype, volume.min(), volume.max(), volume.mean())

    spacing = metadata.get("spacing", (1.0, 1.0, 1.0))

    # Step 2: Run segmentation (TotalSegmentator or MONAI)
    _t2 = _time.time()

    if model_name and model_name.lower() in ("nnunet_handoff", "nnunet701_full_handoff"):
        # --- Custom nnUNet path (Dataset701_TotalSegOrgans6, 6 organs) ---
        logger.info("[PIPELINE] Job %s: calling custom nnUNet (model=%s)...", job_id, model_name)
        try:
            from app.ai.nnunet_custom import run_nnunet_custom, is_available as nnunet_available
            if not nnunet_available():
                raise RuntimeError(
                    f"Custom nnUNet model not found at {settings.NNUNET_CUSTOM_MODEL_PATH}. "
                    "Verify the model directory is mounted in Docker."
                )
            label_map = run_nnunet_custom(
                volume=volume,
                spacing=spacing,
            )
        except Exception as e:
            logger.error("[PIPELINE] Job %s: custom nnUNet FAILED: %s", job_id, e, exc_info=True)
            raise

        # This model does not do lesion detection
        detect_lesions = False

    elif model_name and model_name.lower() in ("nnunet702_20organs",):
        # --- Custom nnUNet 20-class path (Dataset702_TotalSegOrgans20, 20 organs) ---
        logger.info("[PIPELINE] Job %s: calling custom nnUNet 20-class (model=%s)...", job_id, model_name)
        try:
            from app.ai.nnunet_custom_20 import run_nnunet_custom_20, is_available as nnunet20_available
            if not nnunet20_available():
                raise RuntimeError(
                    f"Custom nnUNet 20-class model not found at "
                    f"{settings.NNUNET_CUSTOM_20_MODEL_PATH}. "
                    "Verify the model directory is mounted in Docker."
                )
            # Use merge_to_6=False to keep all 20 classes for the frontend;
            # set to True if the frontend expects the old 6-class scheme.
            label_map = run_nnunet_custom_20(
                volume=volume,
                spacing=spacing,
                merge_to_6=False,
            )
        except Exception as e:
            logger.error("[PIPELINE] Job %s: custom nnUNet 20-class FAILED: %s", job_id, e, exc_info=True)
            raise

        detect_lesions = False

    elif model_name and model_name.lower() == "totalsegmentator":
        # --- TotalSegmentator path (pretrained, 117 structures) ---
        logger.info("[PIPELINE] Job %s: calling TotalSegmentator (model=%s)...", job_id, model_name)
        try:
            from app.ai.totalsegmentator import run_totalsegmentator, is_available
            if not is_available():
                raise ImportError(
                    "TotalSegmentator is not installed. "
                    "Run: pip install TotalSegmentator"
                )
            label_map = run_totalsegmentator(
                volume=volume,
                spacing=spacing,
                task="total",
                fast=settings.TOTALSEGMENTATOR_FAST,
            )
        except Exception as e:
            logger.error("[PIPELINE] Job %s: TotalSegmentator FAILED: %s", job_id, e, exc_info=True)
            raise

        # TotalSegmentator already segments lesions/abnormalities natively,
        # so skip the separate lesion detection step
        detect_lesions = False

    else:
        # --- MONAI path (existing, fallback) ---
        logger.info(
            "[PIPELINE] Job %s: calling run_segmentation(model=%s)...",
            job_id, model_name,
        )
        try:
            label_map = run_segmentation(
                volume=volume,
                model_name=model_name,
                target_organs=target_organs,
                spacing=spacing,
            )
        except Exception as e:
            logger.error("[PIPELINE] Job %s: run_segmentation FAILED: %s", job_id, e, exc_info=True)
            raise

        # Step 3: Optionally run lesion detection (merges into label_map)
        if detect_lesions:
            _t3 = _time.time()
            logger.info("[PIPELINE] Job %s: running lesion detection...", job_id)
            try:
                lesion_map = run_lesion_detection(
                    volume=volume,
                    spacing=spacing,
                )
            except Exception as e:
                logger.error("[PIPELINE] Job %s: lesion detection FAILED: %s", job_id, e, exc_info=True)
                raise
            organ_indices = set(range(1, 8))
            lesion_mask = (lesion_map > 0) & ~np.isin(label_map, list(organ_indices))
            label_map[lesion_mask] = lesion_map[lesion_mask]
            logger.info("[PIPELINE] Job %s: lesion detection done (%.1fs)", job_id, _time.time() - _t3)

    logger.info(
        "[PIPELINE] Job %s: segmentation complete, shape=%s unique_labels=%s (%.1fs)",
        job_id, label_map.shape, np.unique(label_map), _time.time() - _t2,
    )

    # Step 4: Save mask as NRRD to temp file, then upload to storage
    _t4 = _time.time()
    tmp_dir = tempfile.mkdtemp(prefix=f"seg_{job_id}_")
    tmp_nrrd_path = os.path.join(tmp_dir, "mask.nrrd")

    try:
        export_mask_nrrd(
            mask_array=label_map,
            output_path=tmp_nrrd_path,
            spacing=spacing,
            origin=metadata.get("origin", (0.0, 0.0, 0.0)),
        )
        logger.info("[PIPELINE] Job %s: NRRD written to temp (%.1f MB, %.1fs)",
                    job_id, os.path.getsize(tmp_nrrd_path) / (1024*1024), _time.time() - _t4)

        object_key = f"segmentation/{job_id}/mask.nrrd"
        _t5 = _time.time()
        upload_ok = storage.upload_file(
            object_key=object_key,
            file_path=tmp_nrrd_path,
            content_type="application/octet-stream",
        )
        if not upload_ok:
            raise RuntimeError(f"Failed to upload mask to storage (key={object_key})")
        logger.info("[PIPELINE] Job %s: uploaded to storage key=%s (%.1fs)",
                    job_id, object_key, _time.time() - _t5)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    segment_metadata = {
        "shape": list(label_map.shape),
        "spacing": list(spacing),
        "num_labels": int(label_map.max()),
        "source_model": model_name,
        "detect_lesions": detect_lesions,
    }

    logger.info("[PIPELINE] Job %s: DONE (total %.1fs)", job_id, _time.time() - _t0)
    return object_key, segment_metadata


def get_available_organs() -> Dict[str, int]:
    """Return the organ-to-label-index mapping."""
    return dict(ORGAN_LABEL_MAP)
