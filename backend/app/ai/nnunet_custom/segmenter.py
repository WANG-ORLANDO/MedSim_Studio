"""
Custom nnUNet Inference Wrapper (Dataset701_TotalSegOrgans6, 6-class)

Loads a trained nnUNetv2 model from a local checkpoint and runs
inference against a CT volume.  Uses nnUNetPredictor's single-numpy-array
path so no intermediate files are needed.

The model segments 6 organs:
  liver (1), kidney (2), lung (3), spleen (4), pancreas (5), bladder (6)

Architecture flow:
  1. Receive CT volume as numpy array (z, y, x)
  2. Transpose to nnUNet convention: (1, x, y, z)  [C=1 for CT]
  3. Run nnUNetPredictor.predict_single_npy_array
  4. Transpose back to (z, y, x)
  5. Return int32 label map
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class CustomModelNotAvailableError(RuntimeError):
    """Raised when the custom nnUNet model folder is not found."""
    pass


def is_available() -> bool:
    """Check that the trained model folder exists and contains a fold_0 subdir.

    Returns:
        True if the model checkpoint can be loaded, False otherwise.
    """
    model_folder = _get_model_folder()
    if not model_folder or not model_folder.exists():
        return False
    return True


def _get_model_folder() -> Optional[Path]:
    """Return the Path to the nnUNetTrainer__nnUNetPlans__3d_fullres folder.

    The expected layout on disk (after Docker volume mount):
      /app/models/nnunet_handoff/
        dataset.json
        plans.json
        fold_0/
          checkpoint_best.pth
          ...
    """
    candidate = Path(settings.NNUNET_CUSTOM_MODEL_PATH)
    if candidate.is_dir():
        # Make sure it looks like an nnUNet trained model folder
        if (candidate / "dataset.json").is_file() and (candidate / "fold_0").is_dir():
            return candidate

    # Fallback: search for the nnUNetTrainer__nnUNetPlans__3d_fullres subfolder
    for child in candidate.iterdir():
        if child.is_dir() and "nnUNetTrainer" in child.name:
            return child
    return None


def run_nnunet_custom(
    volume: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Run custom nnUNet inference on a CT volume.

    Args:
        volume: 3D numpy array of HU values, shape (z, y, x)
        spacing: Voxel spacing in mm as (z_spacing, y_spacing, x_spacing)

    Returns:
        3D int32 label map with shape (z, y, x). Each voxel contains the
        label index:
          0 = background
          1 = liver
          2 = kidney
          3 = lung
          4 = spleen
          5 = pancreas
          6 = bladder

    Raises:
        CustomModelNotAvailableError: If the model folder is not found.
    """
    model_folder = _get_model_folder()
    if model_folder is None:
        raise CustomModelNotAvailableError(
            f"Custom nnUNet model not found at {settings.NNUNET_CUSTOM_MODEL_PATH}. "
            "Make sure the model directory is mounted correctly."
        )

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume (z, y, x), got shape {volume.shape}")
    if volume.size == 0:
        raise ValueError("Input volume is empty")

    # ---- Environment tuning (same as TotalSegmentator module) ----
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

    import torch
    import multiprocessing

    _orig_cpu_count = multiprocessing.cpu_count
    multiprocessing.cpu_count = lambda: 4
    torch.set_num_threads(4)

    is_cpu = settings.AI_DEVICE == "cpu"
    device = torch.device("cpu") if is_cpu else torch.device("cuda", 0)

    # ---- Transpose volume: (z, y, x) → (x, y, z) → add channel dim (1, x, y, z) ----
    _t0 = __import__("time").time()
    vol_xyz = volume.transpose(2, 1, 0).astype(np.float32)       # (x, y, z)
    input_image = vol_xyz[np.newaxis, ...]                        # (1, x, y, z)

    # nnUNet expects spacing in (x, y, z) order
    spacing_xyz = (spacing[2], spacing[1], spacing[0])

    logger.info(
        "[nnUNet-Custom] Starting inference: shape=%s spacing=%s device=%s",
        volume.shape, spacing, device,
    )

    # ---- Initialize nnUNetPredictor ----
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    _t1 = __import__("time").time()
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=("0",),
        checkpoint_name="checkpoint_best.pth",
    )
    logger.info(
        "[nnUNet-Custom] Model loaded (%.1fs)  trainer=%s",
        __import__("time").time() - _t1,
        model_folder.name,
    )

    # ---- Run prediction ----
    _t2 = __import__("time").time()
    spacing_data = {
        "spacing": spacing_xyz,
    }

    prediction = predictor.predict_single_npy_array(
        input_image,
        spacing_data,
    )

    logger.info(
        "[nnUNet-Custom] Inference complete (%.1fs)",
        __import__("time").time() - _t2,
    )

    # ---- Transpose result back: (x, y, z) → (z, y, x) ----
    # predict_single_npy_array returns argmax, shape (x, y, z)
    label_map = prediction.astype(np.int32)   # (x, y, z)
    label_map = label_map.transpose(2, 1, 0) # (z, y, x)

    # Handle shape mismatch from resampling
    if label_map.shape != volume.shape:
        logger.warning(
            "[nnUNet-Custom] Shape mismatch: output=%s expected=%s, resampling...",
            label_map.shape, volume.shape,
        )
        from scipy.ndimage import zoom
        factors = (
            volume.shape[0] / label_map.shape[0],
            volume.shape[1] / label_map.shape[1],
            volume.shape[2] / label_map.shape[2],
        )
        label_map = zoom(label_map.astype(np.float32), factors, order=0)
        label_map = np.round(label_map).astype(np.int32)

    logger.info(
        "[nnUNet-Custom] Complete (total %.1fs): shape=%s unique_labels=%s",
        __import__("time").time() - _t0,
        label_map.shape,
        np.unique(label_map),
    )

    return label_map
