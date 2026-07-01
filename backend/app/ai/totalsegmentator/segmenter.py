"""
TotalSegmentator Inference Wrapper

Provides a drop-in replacement for MONAI-based segmentation that uses
TotalSegmentator's pretrained models.

Architecture flow:
  1. Receive CT volume as numpy array (z, y, x)
  2. Save as temporary NIfTI file with correct affine/spacing
  3. Run TotalSegmentator CLI via Python API
  4. Read result NIfTI back into numpy array
  5. Return int32 label map compatible with the rest of the pipeline

TotalSegmentator v2 "total" task segments 117 anatomical structures
including all major organs, bones, muscles, vessels, and glands.

The pretrained model weights (~2 GB) are downloaded automatically
on first use from the HuggingFace model hub.

Dependencies:
  - totalsegmentator (or TotalSegmentator v2)
    Install: pip install TotalSegmentator
  - nibabel (for NIfTI I/O, installed as dependency of TotalSegmentator)
"""

import logging
import os
import tempfile
import shutil
from typing import Optional, Tuple

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# Default CT window for abdomen/soft tissue (same as MONAI pipeline)
CT_WINDOW_CENTER = 40.0
CT_WINDOW_WIDTH = 350.0
CT_WINDOW_MIN = CT_WINDOW_CENTER - CT_WINDOW_WIDTH / 2   # -135
CT_WINDOW_MAX = CT_WINDOW_CENTER + CT_WINDOW_WIDTH / 2   #  215


class TotalSegmentatorNotAvailableError(RuntimeError):
    """Raised when TotalSegmentator is not installed."""
    pass


def is_available() -> bool:
    """Check if TotalSegmentator is installed and importable.

    Returns:
        True if TotalSegmentator can be imported, False otherwise.
    """
    try:
        import totalsegmentator  # noqa: F401
        return True
    except ImportError:
        try:
            import TotalSegmentator  # noqa: F401
            return True
        except ImportError:
            return False


def run_totalsegmentator(
    volume: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    task: str = "total",
    fast: bool = False,
) -> np.ndarray:
    """Run TotalSegmentator inference on a CT volume.

    TotalSegmentator is a nnU-Net based segmentation model that can
    segment 117 anatomical structures. The pretrained weights are
    automatically downloaded on first use (~2 GB).

    Args:
        volume: 3D numpy array of HU values, shape (z, y, x)
        spacing: Voxel spacing in mm as (z_spacing, y_spacing, x_spacing)
        task: TotalSegmentator task identifier.
              "total" — all 117 structures (default, recommended)
              "body"  — body composition (muscle, fat, bone)
              "lungs" — lung lobes and segments
              "vertebrae" — vertebral bodies
              "cardiac" — heart substructures
        fast: If True, use TotalSegmentator's fast mode (lower resolution).
              Faster but slightly less accurate. Default: False.

    Returns:
        3D int32 label map with shape (z, y, x). Each voxel contains the
        label index corresponding to TOTAL_SEGMENTATOR_LABEL_MAP.
        0 = background.

    Raises:
        TotalSegmentatorNotAvailableError: If TotalSegmentator not installed.
        ValueError: If the input volume is empty or has invalid shape.
    """
    if not is_available():
        raise TotalSegmentatorNotAvailableError(
            "TotalSegmentator is not installed.\n"
            "Install it with: pip install TotalSegmentator\n"
            "Note: Requires PyTorch. On CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume (z, y, x), got shape {volume.shape}"
        )
    if volume.size == 0:
        raise ValueError("Input volume is empty")

    # Point TotalSegmentator to local pretrained weights.
    # TOTALSEG_WEIGHTS_PATH is read by totalsegmentator.config.setup_nnunet()
    # to set nnUNet_results. If the dataset dirs (Dataset291_..., etc.) exist
    # at that location, download_pretrained_weights() skips the ~2 GB download.
    weights_dir = str(settings.TOTALSEGMENTATOR_DIR)
    if os.path.isdir(weights_dir):
        os.environ.setdefault("TOTALSEG_WEIGHTS_PATH", weights_dir)
        logger.info("[TS] Using local weights at TOTALSEG_WEIGHTS_PATH=%s", weights_dir)
    else:
        logger.info("[TS] TOTALSEGMENTATOR_DIR=%s not found — will download weights on first use", weights_dir)

    # Thread/multiprocessing limits — aggressively reduced for Docker
    # environments where RAM is constrained (Docker Desktop default VM
    # is ~2 GB, and nnU-Net preprocessing workers each load the full
    # volume, causing OOM kills with the default 4 workers).
    #
    # nnUNet_def_n_proc: nnU-Net v2 worker processes.
    #   Set to 1 (was 4) to stay within Docker Desktop RAM limits.
    # nnUNet_n_proc_DA: data augmentation processes, also 1.
    _is_cpu = settings.AI_DEVICE == "cpu"
    _n_proc = "1"
    os.environ["nnUNet_def_n_proc"] = _n_proc
    os.environ["nnUNet_n_proc_DA"] = "1"
    os.environ.setdefault("NUMEXPR_MAX_THREADS", _n_proc)
    os.environ.setdefault("OMP_NUM_THREADS", _n_proc)
    os.environ.setdefault("MKL_NUM_THREADS", _n_proc)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", _n_proc)

    # Import here (lazy) so the module can be imported without TotalSegmentator
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError:
        try:
            from TotalSegmentator.python_api import totalsegmentator
        except ImportError:
            raise TotalSegmentatorNotAvailableError(
                "TotalSegmentator import failed after availability check. "
                "Try reinstalling: pip install TotalSegmentator"
            )

    # Limit PyTorch threads. nnUNetv2_predict internally calls:
    #   torch.set_num_threads(multiprocessing.cpu_count())
    # which overrides our setting and allocates per-thread buffers for ALL
    # cores. Patch cpu_count() so it returns a reasonable value:
    #   RAM-constrained Docker: 1 worker, 1 thread
    import multiprocessing
    _cpus = 1
    _orig_cpu_count = multiprocessing.cpu_count
    multiprocessing.cpu_count = lambda: _cpus

    import torch
    torch.set_num_threads(_cpus)

    import time as _time
    _t0 = _time.time()
    logger.info(
        "[TS] Starting TotalSegmentator inference: shape=%s spacing=%s task=%s fast=%s",
        volume.shape, spacing, task, fast,
    )

    # Create temp directory for input/output NIfTI files
    tmp_dir = tempfile.mkdtemp(prefix="totalseg_")
    try:
        input_path = os.path.join(tmp_dir, "input.nii.gz")
        output_path = os.path.join(tmp_dir, "output.nii.gz")

        # ---- Convert numpy array to NIfTI ----
        _t1 = _time.time()
        _save_volume_as_nifti(volume, input_path, spacing)
        logger.info("[TS] NIfTI write took %.1fs", _time.time() - _t1)

        # ---- Run TotalSegmentator ----
        _t2 = _time.time()
        logger.info("[TS] Calling TotalSegmentator Python API...")
        seg_result = totalsegmentator(
            input=input_path,
            output=output_path,
            task=task,
            fast=fast,
            device="gpu" if not _is_cpu else "cpu",
            # GPU mode: resampling/saving threads can be higher
            nr_thr_resamp=2 if _is_cpu else 4,
            nr_thr_saving=2 if _is_cpu else 4,
            nora_tag=None,
        )
        logger.info("[TS] TotalSegmentator inference done (%.1fs)", _time.time() - _t2)

        # ---- Extract label map from result ----
        # TotalSegmentator v2 returns a nibabel.Nifti1Image directly.
        # Use the returned object in preference to reading from disk,
        # since the file may not always be written to the requested path.
        _t3 = _time.time()
        if seg_result is not None:
            label_map = _nifti_to_numpy(seg_result, volume.shape)
            logger.info(
                "[TS] Result extracted from return value: shape=%s unique_labels=%d (%.1fs)",
                label_map.shape, len(np.unique(label_map)), _time.time() - _t3,
            )
        else:
            label_map = _read_segmentation_as_volume(output_path, volume.shape)
            logger.info(
                "[TS] Result read from disk: shape=%s unique_labels=%d (%.1fs)",
                label_map.shape, len(np.unique(label_map)), _time.time() - _t3,
            )

        logger.info(
            "[TS] TotalSegmentator complete: total %.1fs",
            _time.time() - _t0,
        )
        return label_map

    except Exception as e:
        logger.error("[TS] TotalSegmentator inference failed: %s", e, exc_info=True)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.debug("[TS] Temp directory cleaned up")


def _save_volume_as_nifti(
    volume: np.ndarray,
    output_path: str,
    spacing: Tuple[float, float, float],
) -> None:
    """Save a numpy volume as a NIfTI file with correct orientation.

    TotalSegmentator expects NIfTI files in neurological convention
    (LAS). We set the affine matrix to encode the correct voxel spacing.

    Args:
        volume: 3D numpy array (z, y, x) — HU values
        output_path: Path for the output .nii.gz file
        spacing: (z_spacing, y_spacing, x_spacing) in mm
    """
    import nibabel as nib

    # Create simple affine: voxel spacing on diagonal
    # nibabel convention: (x, y, z) in the array, so we need to
    # transpose from our (z, y, x) to (x, y, z)
    volume_transposed = volume.transpose(2, 1, 0)  # (x, y, z)

    # Affine matrix mapping voxel indices to world coordinates (mm)
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = spacing[2]  # x spacing
    affine[1, 1] = spacing[1]  # y spacing
    affine[2, 2] = spacing[0]  # z spacing

    img = nib.Nifti1Image(volume_transposed, affine)
    nib.save(img, output_path)


def _nifti_to_numpy(
    nifti_img: "nib.Nifti1Image",
    original_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Convert a nibabel Nifti1Image object directly to a numpy label map.

    Preferred over _read_segmentation_as_volume because TotalSegmentator v2
    returns a Nifti1Image as its return value — no disk I/O needed.

    Args:
        nifti_img: nibabel Nifti1Image (as returned by totalsegmentator())
        original_shape: Expected output shape (z, y, x)

    Returns:
        int32 numpy array with shape (z, y, x)
    """
    seg = np.asanyarray(nifti_img.dataobj)  # shape: (x, y, z)

    # Transpose back to (z, y, x)
    seg = seg.transpose(2, 1, 0)
    seg = np.round(seg).astype(np.int32)

    # If shape doesn't match (e.g., TotalSegmentator resampled slightly),
    # resize to match original
    if seg.shape != original_shape:
        logger.warning(
            "[TS] Shape mismatch: output=%s expected=%s, resampling...",
            seg.shape, original_shape,
        )
        from scipy.ndimage import zoom
        factors = (
            original_shape[0] / seg.shape[0],
            original_shape[1] / seg.shape[1],
            original_shape[2] / seg.shape[2],
        )
        seg = zoom(seg.astype(np.float32), factors, order=0)
        seg = np.round(seg).astype(np.int32)

    return seg


def _read_segmentation_as_volume(
    nifti_path: str,
    original_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Read a TotalSegmentator output NIfTI from disk back into a numpy array.

    Fallback for when the returned Nifti1Image is not available.
    Transposes from (x, y, z) nibabel convention back to our (z, y, x).

    Args:
        nifti_path: Path to the NIfTI file
        original_shape: Expected output shape (z, y, x)

    Returns:
        int32 numpy array with shape (z, y, x)
    """
    import nibabel as nib

    img = nib.load(nifti_path)
    seg = np.asanyarray(img.dataobj)  # shape: (x, y, z)

    # Transpose back to (z, y, x)
    seg = seg.transpose(2, 1, 0)
    seg = np.round(seg).astype(np.int32)

    # If shape doesn't match (e.g., TotalSegmentator resampled slightly),
    # resize to match original
    if seg.shape != original_shape:
        logger.warning(
            "[TS] Shape mismatch: output=%s expected=%s, resampling...",
            seg.shape, original_shape,
        )
        from scipy.ndimage import zoom
        factors = (
            original_shape[0] / seg.shape[0],
            original_shape[1] / seg.shape[1],
            original_shape[2] / seg.shape[2],
        )
        seg = zoom(seg.astype(np.float32), factors, order=0)
        seg = np.round(seg).astype(np.int32)

    return seg
