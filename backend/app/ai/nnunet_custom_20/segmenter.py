"""
Custom nnUNet 20-class Inference Wrapper (Dataset702_TotalSegOrgans20)

Loads the trained nnUNetv2 20-class model from a local checkpoint and runs
inference against a CT volume.  Uses nnUNetPredictor's single-numpy-array
path so no intermediate files are needed.

The model segments 20 anatomical structures:
  adrenal glands, colon, duodenum, esophagus, gallbladder,
  kidneys (left/right), liver, lung lobes (5), pancreas,
  small bowel, spleen, stomach, trachea, urinary bladder

Optionally merges output to 6 classes for frontend backward compatibility.

Architecture flow:
  1. Receive CT volume as numpy array (z, y, x)
  2. Transpose to nnUNet convention: (1, x, y, z)  [C=1 for CT]
  3. Run nnUNetPredictor.predict_single_npy_array
  4. Transpose back to (z, y, x)
  5. Optionally merge 20→6 classes
  6. Return int32 label map
"""

import logging
import os
import shutil
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
      /app/models/nnunet702_handoff/
        dataset.json
        plans.json
        fold_0/
          checkpoint_best.pth
          ...
    """
    candidate = Path(settings.NNUNET_CUSTOM_20_MODEL_PATH)
    if candidate.is_dir():
        if (candidate / "dataset.json").is_file() and (candidate / "fold_0").is_dir():
            return candidate

    # Fallback: search for the nnUNetTrainer__nnUNetPlans__3d_fullres subfolder
    for child in candidate.iterdir():
        if child.is_dir() and "nnUNetTrainer" in child.name:
            return child
    return None


def merge_to_6_classes(label_map: np.ndarray) -> np.ndarray:
    """Merge a 20-class label map into 6 classes for frontend compat.

    Merge mapping:
      1  liver    <- 9
      2  kidney   <- 7, 8
      3  lung     <- 10, 11, 12, 13, 14, 19 (trachea)
      4  spleen   <- 17
      5  pancreas <- 15
      6  bladder  <- 20

    All other labels (adrenal glands, GI tract, gallbladder)
    become background (0).

    Args:
        label_map: int32 array with values 0-20

    Returns:
        int32 array with values 0-6
    """
    from app.ai.nnunet_custom_20.labels import MERGE_TO_6_MAP

    out = np.zeros_like(label_map, dtype=np.int32)
    for src, dst in MERGE_TO_6_MAP.items():
        out[label_map == src] = dst
    return out


def run_nnunet_custom_20(
    volume: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    merge_to_6: bool = False,
) -> np.ndarray:
    """Run custom nnUNet 20-class inference on a CT volume.

    Args:
        volume: 3D numpy array of HU values, shape (z, y, x)
        spacing: Voxel spacing in mm as (z_spacing, y_spacing, x_spacing)
        merge_to_6: If True, collapse the 20-class output to 6 classes
                    (liver, kidney, lung, spleen, pancreas, bladder)

    Returns:
        3D int32 label map with shape (z, y, x). Each voxel contains the
        label index:
          Without merge (merge_to_6=False):
            0  = background
            1  = left_adrenal_gland
            2  = right_adrenal_gland
            3  = colon
            4  = duodenum
            5  = esophagus
            6  = gallbladder
            7  = left_kidney
            8  = right_kidney
            9  = liver
            10 = left_lung_lower_lobe
            11 = right_lung_lower_lobe
            12 = right_lung_middle_lobe
            13 = left_lung_upper_lobe
            14 = right_lung_upper_lobe
            15 = pancreas
            16 = small_bowel
            17 = spleen
            18 = stomach
            19 = trachea
            20 = urinary_bladder

          With merge (merge_to_6=True):
            1 = liver, 2 = kidney, 3 = lung, 4 = spleen,
            5 = pancreas, 6 = bladder

    Raises:
        CustomModelNotAvailableError: If the model folder is not found.
    """
    model_folder = _get_model_folder()
    if model_folder is None:
        raise CustomModelNotAvailableError(
            f"Custom nnUNet 20-class model not found at "
            f"{settings.NNUNET_CUSTOM_20_MODEL_PATH}. "
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

    # ---- Transpose volume: (z, y, x) → (x, y, z) ----
    _t0 = __import__("time").time()
    vol_xyz = volume.transpose(2, 1, 0).astype(np.float32)       # (x, y, z)

    # nnUNet expects spacing in (x, y, z) order
    spacing_xyz = (spacing[2], spacing[1], spacing[0])

    logger.info(
        "[nnUNet-Custom-20] Starting inference: shape=%s spacing=%s device=%s",
        volume.shape, spacing, device,
    )

    # ---- Initialize nnUNetPredictor ----
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.75,            # Larger step → fewer tiles → less GPU memory
        use_gaussian=True,
        use_mirroring=False,             # Disabled: TTA 3D mirroring costs 8× GPU memory
        perform_everything_on_device=True,   # Preprocess on GPU (no mirroring → enough VRAM)
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
        "[nnUNet-Custom-20] Model loaded (%.1fs)  trainer=%s",
        __import__("time").time() - _t1,
        model_folder.name,
    )

    import tempfile

    # ---- Run prediction via file-based predict_from_raw_data ----
    # predict_single_npy_array is known to hang with some nnUNet v2
    # configurations; the file-based path is more reliable.
    _t2 = __import__("time").time()

    tmp_dir = tempfile.mkdtemp(prefix="nnunet20_")
    try:
        # Save input volume as NIfTI (nnUNet naming: case_0000.nii.gz)
        import nibabel as nib
        input_path = os.path.join(tmp_dir, "input_0000.nii.gz")
        # Volume is transposed to (x, y, z) — matching nnUNet convention
        nifti_img = nib.Nifti1Image(vol_xyz, np.eye(4))
        nib.save(nifti_img, input_path)
        logger.debug("[nnUNet-Custom-20] Saved input to %s", input_path)

        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        predictor.predict_from_files(
            list_of_lists_or_source_folder=tmp_dir,
            output_folder_or_list_of_truncated_output_files=output_dir,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
        )

        # Read back result
        import glob
        result_files = glob.glob(os.path.join(output_dir, "*.nii.gz"))
        if not result_files:
            raise RuntimeError(f"nnUNet produced no output files in {output_dir}")
        result_path = result_files[0]
        logger.debug("[nnUNet-Custom-20] Reading result from %s", result_path)
        prediction = nib.load(result_path).get_fdata().astype(np.int32)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(
        "[nnUNet-Custom-20] Inference complete (%.1fs)",
        __import__("time").time() - _t2,
    )

    # ---- Transpose result back: (x, y, z) → (z, y, x) ----
    label_map = prediction.astype(np.int32)   # (x, y, z)
    label_map = label_map.transpose(2, 1, 0)  # (z, y, x)

    # Handle shape mismatch from resampling
    if label_map.shape != volume.shape:
        logger.warning(
            "[nnUNet-Custom-20] Shape mismatch: output=%s expected=%s, resampling...",
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

    # ---- Optional 20→6 merge ----
    if merge_to_6:
        _t3 = __import__("time").time()
        label_map = merge_to_6_classes(label_map)
        logger.debug("[nnUNet-Custom-20] Merged to 6 classes (%.3fs)",
                     __import__("time").time() - _t3)

    logger.info(
        "[nnUNet-Custom-20] Complete (total %.1fs): shape=%s unique_labels=%s%s",
        __import__("time").time() - _t0,
        label_map.shape,
        np.unique(label_map),
        " (merged to 6)" if merge_to_6 else "",
    )

    return label_map
