"""
Interactive Segmentation Refinement

Click-based refinement for segmentation masks using local region growing.
Provides tools to add or remove labels from a segmentation mask
based on user click points, using intensity-constrained region growing
within the original CT volume.
"""

import logging
from typing import Tuple, Optional
import numpy as np

from scipy.ndimage import binary_dilation, generate_binary_structure

logger = logging.getLogger(__name__)

# Default region growing parameters
DEFAULT_RADIUS = 15  # voxels — local patch size around click
INTENSITY_TOLERANCE = 30.0  # HU — max intensity difference for region growing


def refine_mask_on_click(
    mask_array: np.ndarray,
    volume: np.ndarray,
    z: int,
    x: int,
    y: int,
    label: int,
    operation: str = "add",
) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
    """
    Refine a segmentation mask based on a user click.

    Uses intensity-constrained region growing from the click point,
    constrained to a local patch around the click.

    Args:
        mask_array: Current 3D label map, shape (z, y, x), int32
        volume: Original CT volume, shape (z, y, x), HU values
        z: Click slice index (z-axis)
        x: Click x voxel coordinate
        y: Click y voxel coordinate
        label: Label index to assign (1-9)
        operation: "add" to assign label, "remove" to set to 0

    Returns:
        Tuple of:
          - Updated mask_array with the local region modified
          - Bounding box (z_start, z_end, y_start, y_end, x_start, x_end)
            indicating the region that was modified
    """
    Z, Y, X = mask_array.shape

    # Clamp click coordinates to volume bounds
    z = max(0, min(z, Z - 1))
    x = max(0, min(x, X - 1))
    y = max(0, min(y, Y - 1))

    # Define local patch bounds
    z_start = max(0, z - DEFAULT_RADIUS)
    z_end = min(Z, z + DEFAULT_RADIUS + 1)
    y_start = max(0, y - DEFAULT_RADIUS)
    y_end = min(Y, y + DEFAULT_RADIUS + 1)
    x_start = max(0, x - DEFAULT_RADIUS)
    x_end = min(X, x + DEFAULT_RADIUS + 1)

    bbox = (z_start, z_end, y_start, y_end, x_start, x_end)

    if operation == "add":
        # Region growing: find all connected voxels within intensity tolerance
        patch_volume = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        seed_value = volume[z, y, x]

        # Binary condition: within intensity tolerance
        intensity_mask = np.abs(patch_volume - seed_value) <= INTENSITY_TOLERANCE

        # Also constrain: don't grow into regions far from the seed in intensity
        seed_z = z - z_start
        seed_y = y - y_start
        seed_x = x - x_start

        # Flood-fill within intensity mask starting from seed
        region = _flood_fill_3d(intensity_mask, (seed_z, seed_y, seed_x))

        # Apply the new label to the grown region
        mask_array[z_start:z_end, y_start:y_end, x_start:x_end][region] = label

    elif operation == "remove":
        # Remove label from local region around click
        patch_mask = mask_array[z_start:z_end, y_start:y_end, x_start:x_end]
        target_label = int(mask_array[z, y, x])

        if target_label > 0:
            # Find the connected component at the click point within the patch
            seed_region = patch_mask == target_label
            seed_z = z - z_start
            seed_y = y - y_start
            seed_x = x - x_start

            connected = _flood_fill_3d(seed_region, (seed_z, seed_y, seed_x))

            # Erase the connected component
            mask_array[z_start:z_end, y_start:y_end, x_start:x_end][connected] = 0

    return mask_array, bbox


def _flood_fill_3d(
    binary_mask: np.ndarray,
    seed: Tuple[int, int, int],
) -> np.ndarray:
    """
    Perform 3D flood fill constrained by a binary mask.

    Uses iterative dilation from the seed point, masked by `binary_mask`.
    Returns a boolean array of the same shape as binary_mask, True where
    connected to the seed.

    Args:
        binary_mask: Boolean mask where True = allowed region
        seed: (z, y, x) seed point within the mask

    Returns:
        Boolean array of same shape, True = connected region
    """
    if not binary_mask[seed]:
        # Seed is outside the allowed region — return empty
        return np.zeros_like(binary_mask, dtype=bool)

    result = np.zeros_like(binary_mask, dtype=bool)
    result[seed] = True

    # 3D structural element (26-connectivity)
    struct = generate_binary_structure(rank=3, connectivity=2)

    # Iteratively dilate, constrained by the binary mask
    prev_count = 0
    current_count = 1

    # Max iterations to prevent runaway
    max_iters = int(np.prod(binary_mask.shape) * 0.5)

    iteration = 0
    while current_count > prev_count and iteration < max_iters:
        prev_count = current_count
        result = binary_dilation(result, structure=struct) & binary_mask
        current_count = np.sum(result)
        iteration += 1

    return result


def extract_slice_mask(
    mask_array: np.ndarray,
    z_index: int,
) -> np.ndarray:
    """
    Extract a single 2D slice from a 3D label map.

    Args:
        mask_array: 3D label map, shape (z, y, x)
        z_index: Slice index to extract

    Returns:
        2D int32 array of label indices, shape (y, x)
    """
    z_index = max(0, min(z_index, mask_array.shape[0] - 1))
    return mask_array[z_index].astype(np.int32)
