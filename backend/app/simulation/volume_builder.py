"""
Volume Builder

Builds 3D CT volumes from DICOM series (read from storage backend)
or generates synthetic base volumes when no source DICOM is provided.

Responsibilities:
- Read DICOM instances from storage backend
- Parse pixel data with HU conversion (RescaleSlope / RescaleIntercept)
- Sort slices by ImagePositionPatient or slice_location
- Stack into 3D numpy array
- Generate synthetic base volume as fallback

Returns (volume_array, metadata_dict) where metadata_dict contains
spacing, origin, direction for use with SimpleITK.
"""

import io
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pydicom

from app.dicom.storage.base import StorageBackend
from app.models.dicom import DicomInstance

logger = logging.getLogger(__name__)

# Default volume parameters for synthetic mode
DEFAULT_SHAPE = (64, 128, 128)  # (z, y, x)
DEFAULT_SPACING = (1.0, 0.5, 0.5)  # mm (z, y, x)
DEFAULT_ORIGIN = (0.0, 0.0, 0.0)


def build_volume_from_dicom(
    storage: StorageBackend,
    instances: List[DicomInstance],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a 3D CT volume from DICOM instances stored in the storage backend.

    Reads each instance's pixel data via storage.get_object_bytes(),
    applies RescaleSlope / RescaleIntercept for HU conversion,
    sorts slices by ImagePositionPatient (z-axis) or slice_location,
    and stacks into a 3D numpy array.

    Args:
        storage: StorageBackend instance (MinIO or Local)
        instances: List of DicomInstance ORM objects, already ordered by instance_number

    Returns:
        Tuple of (volume_array, metadata_dict) where:
        - volume_array: np.ndarray of shape (z, y, x) with HU values (float32)
        - metadata_dict: {"spacing": (z,y,x), "origin": (z,y,x), "direction": tuple}

    Raises:
        ValueError: If no valid instances found or pixel data unreadable
    """
    if not instances:
        raise ValueError("No DICOM instances provided")

    # Collect slices with their z-positions for sorting
    slices: List[Tuple[float, np.ndarray]] = []
    slice_thickness: Optional[float] = None
    pixel_spacing: Optional[Tuple[float, float]] = None
    origin: Tuple[float, float, float] = DEFAULT_ORIGIN

    for inst in instances:
        if not inst.pixel_data_path:
            logger.warning("Instance %s has no pixel_data_path, skipping", inst.id)
            continue

        # Read DICOM bytes from storage backend
        dicom_bytes = storage.get_object_bytes(inst.pixel_data_path)
        if dicom_bytes is None:
            logger.warning(
                "Failed to read instance %s from storage (key=%s), skipping",
                inst.id, inst.pixel_data_path,
            )
            continue

        try:
            ds = pydicom.dcmread(io.BytesIO(dicom_bytes), force=True)
        except Exception as e:
            logger.warning("Failed to parse DICOM for instance %s: %s, skipping", inst.id, e)
            continue

        # Extract pixel data
        if not hasattr(ds, "pixel_array"):
            logger.warning("Instance %s has no pixel_array, skipping", inst.id)
            continue

        try:
            pixel_array = ds.pixel_array.astype(np.float32)
        except Exception as e:
            logger.warning("Failed to extract pixel_array for instance %s: %s", inst.id, e)
            continue

        # Apply RescaleSlope / RescaleIntercept for HU conversion
        rescale_slope = getattr(ds, "RescaleSlope", None)
        rescale_intercept = getattr(ds, "RescaleIntercept", None)
        if rescale_slope is not None:
            pixel_array = pixel_array * float(rescale_slope)
        if rescale_intercept is not None:
            pixel_array = pixel_array + float(rescale_intercept)

        # Determine z-position for sorting
        z_pos = _get_slice_z_position(ds, inst)

        slices.append((z_pos, pixel_array))

        # Capture metadata from first valid slice
        if slice_thickness is None:
            slice_thickness = _safe_float(getattr(ds, "SliceThickness", None))
        if pixel_spacing is None:
            ps = getattr(ds, "PixelSpacing", None)
            if ps is not None:
                try:
                    pixel_spacing = (float(ps[0]), float(ps[1]))
                except (IndexError, TypeError, ValueError):
                    pass

    if not slices:
        raise ValueError("No valid DICOM slices could be read from storage backend")

    # Sort slices by z-position
    slices.sort(key=lambda x: x[0])

    # Stack into 3D volume: shape (z, y, x)
    volume = np.stack([s[1] for s in slices], axis=0)

    # Compute spacing
    if len(slices) > 1:
        z_spacing = abs(slices[1][0] - slices[0][0])
        if z_spacing < 1e-6:
            z_spacing = slice_thickness or 1.0
    else:
        z_spacing = slice_thickness or 1.0

    y_spacing = pixel_spacing[0] if pixel_spacing else 1.0
    x_spacing = pixel_spacing[1] if pixel_spacing else 1.0

    spacing = (z_spacing, y_spacing, x_spacing)

    # Compute origin from first slice position
    if slices:
        origin_z = slices[0][0]
        origin = (float(origin_z), 0.0, 0.0)

    metadata = {
        "spacing": spacing,
        "origin": origin,
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),  # identity
        "num_slices": len(slices),
        "source": "dicom",
    }

    logger.info(
        "Built DICOM volume: shape=%s, spacing=%s, slices=%d",
        volume.shape, spacing, len(slices),
    )

    return volume, metadata


def build_synthetic_volume(
    shape: Tuple[int, int, int] = DEFAULT_SHAPE,
    spacing: Tuple[float, float, float] = DEFAULT_SPACING,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate a synthetic base CT volume for simulation without source DICOM.

    Creates a volume with:
    - Air background (-1000 HU)
    - Central soft-tissue ellipsoid (~40 HU with noise)
    - Realistic spacing metadata

    This is the "no-source simulation mode" — used when a simulation job
    is created without a study_id/series_id reference.

    Args:
        shape: Volume shape (z, y, x)
        spacing: Voxel spacing in mm (z, y, x)

    Returns:
        Tuple of (volume_array, metadata_dict)
    """
    rng = np.random.default_rng(42)

    # Air background
    volume = np.full(shape, -1000.0, dtype=np.float32)

    # Central soft-tissue ellipsoid
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    rz, ry, rx = shape[0] * 0.35, shape[1] * 0.35, shape[2] * 0.35

    z, y, x = np.indices(shape, dtype=float)
    distance = np.sqrt(
        ((z - cz) / rz) ** 2 +
        ((y - cy) / ry) ** 2 +
        ((x - cx) / rx) ** 2
    )

    tissue_mask = distance <= 1.0

    # Soft tissue HU: ~40 ± 10
    tissue_hu = rng.normal(40.0, 10.0, shape).astype(np.float32)
    volume[tissue_mask] = tissue_hu[tissue_mask]

    # Smooth edges
    from scipy.ndimage import gaussian_filter
    edge_mask = (distance > 0.9) & (distance <= 1.1)
    if edge_mask.any():
        smooth_factor = 1.0 - (distance[edge_mask] - 0.9) / 0.2
        smooth_factor = np.clip(smooth_factor, 0, 1)
        volume[edge_mask] = volume[edge_mask] * smooth_factor + (-1000.0) * (1.0 - smooth_factor)

    metadata = {
        "spacing": spacing,
        "origin": DEFAULT_ORIGIN,
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
        "num_slices": shape[0],
        "source": "synthetic",
    }

    logger.info(
        "Built synthetic volume: shape=%s, spacing=%s",
        shape, spacing,
    )

    return volume, metadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_slice_z_position(ds, inst: DicomInstance) -> float:
    """
    Determine the z-axis position of a DICOM slice.

    Priority:
    1. ImagePositionPatient[2] (most reliable)
    2. DicomInstance.slice_location (from DB metadata)
    3. DicomInstance.instance_number (fallback)
    """
    # Try ImagePositionPatient from DICOM dataset
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None:
        try:
            return float(ipp[2])
        except (IndexError, TypeError, ValueError):
            pass

    # Try from DB metadata
    if inst.slice_location is not None:
        return float(inst.slice_location)

    # Try from DB image_position JSON
    if inst.image_position is not None:
        try:
            if isinstance(inst.image_position, (list, tuple)) and len(inst.image_position) >= 3:
                return float(inst.image_position[2])
        except (IndexError, TypeError, ValueError):
            pass

    # Last resort: instance_number
    if inst.instance_number is not None:
        return float(inst.instance_number)

    return 0.0


def _safe_float(value, default: float = 1.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
