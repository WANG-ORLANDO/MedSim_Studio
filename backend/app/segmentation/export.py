"""
Segmentation Export Utilities

Export segmentation masks in NRRD, NIfTI, and DICOM SEG formats.
Follows the same pattern as 'app/simulation/exporter.py'.
"""

import os
import tempfile
import logging
from typing import Optional

import numpy as np
from fastapi.responses import FileResponse, StreamingResponse

from app.dicom.storage.base import StorageBackend

logger = logging.getLogger(__name__)


def export_mask_nrrd(
    mask_array: np.ndarray,
    output_path: str,
    spacing: tuple = (1.0, 1.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
) -> None:
    """
    Save a segmentation mask as an NRRD file using SimpleITK.

    Args:
        mask_array: 3D int32 label map, shape (z, y, x)
        output_path: Path to write the .nrrd file
        spacing: (z, y, x) voxel spacing in mm
        origin: (z, y, x) origin in mm
    """
    import SimpleITK as sitk

    # SimpleITK expects (x, y, z) ordering
    sitk_spacing = (spacing[2], spacing[1], spacing[0])
    sitk_origin = (origin[2], origin[1], origin[0])

    sitk_image = sitk.GetImageFromArray(mask_array.astype(np.int32))
    sitk_image.SetSpacing(sitk_spacing)
    sitk_image.SetOrigin(sitk_origin)

    sitk.WriteImage(sitk_image, output_path, useCompression=True)
    logger.info("Saved NRRD mask to %s (shape=%s)", output_path, mask_array.shape)


def export_mask_nifti(
    mask_array: np.ndarray,
    output_path: str,
    spacing: tuple = (1.0, 1.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
    affine: Optional[np.ndarray] = None,
) -> None:
    """
    Save a segmentation mask as a NIfTI file using nibabel.

    Args:
        mask_array: 3D int32 label map, shape (z, y, x)
        output_path: Path to write the .nii.gz file
        spacing: (z, y, x) voxel spacing in mm
        origin: (z, y, x) origin in mm
        affine: 4x4 affine matrix; computed from spacing/origin if None
    """
    import nibabel as nib

    if affine is None:
        # Construct a simple affine from spacing and origin
        affine = np.eye(4, dtype=np.float64)
        affine[0, 0] = spacing[2]  # x
        affine[1, 1] = spacing[1]  # y
        affine[2, 2] = spacing[0]  # z
        affine[0, 3] = origin[2]
        affine[1, 3] = origin[1]
        affine[2, 3] = origin[0]

    # nibabel expects (x, y, z) — transpose from our (z, y, x)
    nifti_data = np.transpose(mask_array, (2, 1, 0)).astype(np.int16)

    nifti_img = nib.Nifti1Image(nifti_data, affine)
    nib.save(nifti_img, output_path)
    logger.info("Saved NIfTI mask to %s", output_path)


def export_mask_dicom_seg(
    mask_array: np.ndarray,
    output_path: str,
    dicom_series_dir: str,
) -> None:
    """
    Export a segmentation mask as a DICOM SEG object.

    Uses highdicom to create a DICOM Segmentation image from
    the label map and a reference DICOM series.

    Args:
        mask_array: 3D int32 label map, shape (z, y, x)
        output_path: Path to write the .dcm file
        dicom_series_dir: Directory containing the original DICOM series files
    """
    import highdicom as hd
    import pydicom
    from pydicom.sopclass import generate_uid
    import os

    # Read the source DICOM series for reference
    source_images = []
    for fname in sorted(os.listdir(dicom_series_dir)):
        if fname.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(dicom_series_dir, fname), force=True)
            if hasattr(ds, "pixel_array"):
                source_images.append(ds)

    if not source_images:
        raise ValueError("No source DICOM images found in %s", dicom_series_dir)

    # Build label descriptions from the known segment list
    from app.ai.monai.model_loader import ORGAN_LABEL_MAP
    segment_descriptions = []
    for name, idx in ORGAN_LABEL_MAP.items():
        if idx == 0:
            continue  # skip background
        segment_descriptions.append(
            hd.seg.SegmentDescription(
                segment_number=int(idx),
                segment_label=name.replace("_", " ").title(),
                segmented_property_category=hd.code.CID7150.AnatomicalStructure,
                segmented_property_type=hd.code.CID7166.OrganTissue,
                algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            )
        )

    # Create DICOM SEG object
    seg_dataset = hd.seg.Segmentation(
        source_images=source_images,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=generate_uid(),
        series_number=999,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="CT Simulator",
        manufacturer_model_name="MONAI Segmentation",
        software_versions="0.1.0",
        device_serial_number="CTSIM-001",
    )

    # The segmentation source image pixel arrays define the frame geometry.
    # We convert our label map to a per-segment binary frame list.
    total_frames = len(source_images)
    num_segments = len(segment_descriptions)
    frames = np.zeros((num_segments, total_frames, mask_array.shape[1], mask_array.shape[2]), dtype=bool)

    for seg_idx, (name, label_idx) in enumerate(
        [(n, i) for n, i in ORGAN_LABEL_MAP.items() if i > 0]
    ):
        for z in range(min(total_frames, mask_array.shape[0])):
            frames[seg_idx, z] = mask_array[z] == label_idx

    seg_dataset.PixelData = frames.tobytes()
    seg_dataset.NumberOfFrames = total_frames
    seg_dataset.Rows = mask_array.shape[1]
    seg_dataset.Columns = mask_array.shape[2]

    seg_dataset.save_as(output_path)
    logger.info("Saved DICOM SEG to %s (%d segments)", output_path, num_segments)


def stream_mask_from_storage(
    storage: StorageBackend,
    object_key: str,
    filename: str,
    media_type: str = "application/octet-stream",
) -> FileResponse:
    """
    Stream a mask file from the storage backend as an HTTP response.

    Downloads to a temp file first (storage backends may not support
    direct streaming), then returns a FileResponse that cleans up
    the temp file after sending.
    """
    mask_bytes = storage.get_object_bytes(object_key)
    if mask_bytes is None:
        raise FileNotFoundError(f"Mask not found in storage: {object_key}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
    tmp.write(mask_bytes)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    return FileResponse(
        path=tmp_path,
        media_type=media_type,
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
