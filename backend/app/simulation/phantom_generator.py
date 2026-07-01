"""
Synthetic & Atlas-Based Upper-Body CT Phantom Generator
========================================================

Two modes:
  1. Procedural (default) — geometric primitives approximating anatomy
  2. Atlas-based — loads a real CT volume + organ label map from disk

This is for DEMONSTRATION and UI development — NOT for medical diagnosis.

Usage — procedural:
    from app.simulation.phantom_generator import generate_upper_body_ct_phantom
    volume, metadata = generate_upper_body_ct_phantom(shape=(128, 128, 128))

Usage — atlas:
    from app.simulation.phantom_generator import generate_atlas_ct_phantom
    ct_volume, label_volume, metadata = generate_atlas_ct_phantom(
        case_id="s0001", size=160,
    )
"""

import os
import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy.ndimage import gaussian_filter, zoom

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AIR_HU: float = -1000.0
SOFT_TISSUE_HU: float = 40.0
SOFT_TISSUE_STD: float = 10.0

# Window/Level presets (WL, WW)
WINDOW_PRESETS: Dict[str, Dict[str, float]] = {
    "soft": {"window_level": 40.0, "window_width": 400.0},
    "lung": {"window_level": -600.0, "window_width": 1500.0},
    "bone": {"window_level": 500.0, "window_width": 2000.0},
}

# ---------------------------------------------------------------------------
# Organ label map — index → name
#   0  = background
#   1  = left_adrenal_gland
#   2  = right_adrenal_gland
#   3  = colon
#   4  = duodenum
#   5  = esophagus
#   6  = gallbladder
#   7  = left_kidney
#   8  = right_kidney
#   9  = liver
#   10 = left_lung_lower_lobe
#   11 = right_lung_lower_lobe
#   12 = right_lung_middle_lobe
#   13 = left_lung_upper_lobe
#   14 = right_lung_upper_lobe
#   15 = pancreas
#   16 = small_bowel
#   17 = spleen
#   18 = stomach
#   19 = trachea
#   20 = urinary_bladder
# ---------------------------------------------------------------------------

ORGAN_LABEL_MAP: Dict[int, str] = {
    0: "background",
    1: "left_adrenal_gland",
    2: "right_adrenal_gland",
    3: "colon",
    4: "duodenum",
    5: "esophagus",
    6: "gallbladder",
    7: "left_kidney",
    8: "right_kidney",
    9: "liver",
    10: "left_lung_lower_lobe",
    11: "right_lung_lower_lobe",
    12: "right_lung_middle_lobe",
    13: "left_lung_upper_lobe",
    14: "right_lung_upper_lobe",
    15: "pancreas",
    16: "small_bowel",
    17: "spleen",
    18: "stomach",
    19: "trachea",
    20: "urinary_bladder",
}


def generate_upper_body_ct_phantom(
    shape: Tuple[int, int, int] = (128, 128, 128),
    spacing: Tuple[float, float, float] = (1.5, 1.2, 1.2),
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate a synthetic upper-body CT phantom.

    The volume is axial: shape[0] = number of slices (z, superior→inferior),
    shape[1] = rows (y, anterior→posterior), shape[2] = columns (x, left→right).

    Args:
        shape:   Volume dimensions as (depth_z, height_y, width_x).
                 Default (128, 128, 128) ≈ 2M voxels, ~8 MB float32.
        spacing: Voxel spacing in mm (z, y, x).
        seed:    Random seed for reproducible noise.

    Returns:
        (volume, metadata) tuple.
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape

    # ------------------------------------------------------------------
    # 1. Background — air
    # ------------------------------------------------------------------
    volume = np.full(shape, AIR_HU, dtype=np.float32)

    # Coordinate grids (z, y, x indices)
    z_idx, y_idx, x_idx = np.indices(shape, dtype=np.float32)

    # Normalised coordinates in [-1, 1] range per axis
    zn = (z_idx / nz) * 2.0 - 1.0   # -1=top, +1=bottom
    yn = (y_idx / ny) * 2.0 - 1.0   # -1=anterior, +1=posterior
    xn = (x_idx / nx) * 2.0 - 1.0   # -1=left, +1=right

    # ------------------------------------------------------------------
    # 2. Body outline — z-dependent ellipse (upper-body shape)
    # ------------------------------------------------------------------
    # Key: zn = -1 (top/neck), zn = +1 (bottom/abdomen)
    # Radii vary per slice to create an upper-body silhouette instead of
    # a constant-section cylinder.
    #
    # Z-keypoints (normalised zn):       -1.0   -0.70  -0.35  0.00  0.30  0.65  1.00
    #   rx (left-right radius):           0.30   0.44   0.68  0.78  0.74  0.65  0.52
    #   ry (anterior-posterior radius):   0.28   0.38   0.52  0.65  0.62  0.53  0.40
    #
    # This produces: narrow neck → broadening shoulders → wide chest →
    # gradually tapering abdomen.

    z_keypoints = np.array([-1.0, -0.70, -0.35, 0.00, 0.30, 0.65, 1.00], dtype=np.float32)
    rx_keypoints = np.array([0.30, 0.44, 0.68, 0.78, 0.74, 0.65, 0.52], dtype=np.float32)
    ry_keypoints = np.array([0.28, 0.38, 0.52, 0.65, 0.62, 0.53, 0.40], dtype=np.float32)

    # 1-D per-slice radii (nz,)
    zn_1d = np.linspace(-1.0, 1.0, nz, dtype=np.float32)
    body_rx_z = np.interp(zn_1d, z_keypoints, rx_keypoints).astype(np.float32)
    body_ry_z = np.interp(zn_1d, z_keypoints, ry_keypoints).astype(np.float32)

    # Broadcast to 3-D: (nz, 1, 1)
    body_rx_3d = body_rx_z[:, np.newaxis, np.newaxis]
    body_ry_3d = body_ry_z[:, np.newaxis, np.newaxis]

    body_distance = np.sqrt(
        (xn / body_rx_3d) ** 2 + (yn / body_ry_3d) ** 2
    )
    body_mask = body_distance <= 1.0

    # Apply soft tissue HU with noise
    tissue_noise = rng.normal(SOFT_TISSUE_HU, SOFT_TISSUE_STD, shape).astype(np.float32)
    volume[body_mask] = tissue_noise[body_mask]

    # Smooth the body boundary slightly
    edge_zone = (body_distance > 0.92) & (body_distance <= 1.08)
    if edge_zone.any():
        fade = 1.0 - (body_distance[edge_zone] - 0.92) / 0.16
        fade = np.clip(fade, 0.0, 1.0)
        volume[edge_zone] = (
            volume[edge_zone] * fade + AIR_HU * (1.0 - fade)
        )

    # ------------------------------------------------------------------
    # 3. Lungs — two ellipses (left and right)
    #    Lungs are present in the upper ~60% of the volume
    # ------------------------------------------------------------------
    lung_z_frac = zn.copy()  # -1 top to +1 bottom
    lung_present = lung_z_frac < 0.2  # upper ~60% (zn from -1 to +0.2)

    # Left lung
    left_lung_cx = -0.35
    left_lung_cy = 0.0
    left_lung_rx = 0.22
    left_lung_ry = 0.30

    left_lung_dist = np.sqrt(
        ((xn - left_lung_cx) / left_lung_rx) ** 2 +
        ((yn - left_lung_cy) / left_lung_ry) ** 2
    )
    left_lung_mask = (left_lung_dist <= 1.0) & lung_present & body_mask

    # Right lung (slightly larger)
    right_lung_cx = 0.35
    right_lung_cy = 0.0
    right_lung_rx = 0.24
    right_lung_ry = 0.32

    right_lung_dist = np.sqrt(
        ((xn - right_lung_cx) / right_lung_rx) ** 2 +
        ((yn - right_lung_cy) / right_lung_ry) ** 2
    )
    right_lung_mask = (right_lung_dist <= 1.0) & lung_present & body_mask

    lung_mask = left_lung_mask | right_lung_mask
    lung_hu = rng.normal(-650.0, 80.0, shape).astype(np.float32)
    # Clip lung HU to plausible range
    lung_hu = np.clip(lung_hu, -900.0, -400.0)
    volume[lung_mask] = lung_hu[lung_mask]

    # ------------------------------------------------------------------
    # 4. Spine / vertebra — circular region in posterior center
    #    Present throughout the entire volume
    # ------------------------------------------------------------------
    spine_cx = 0.0
    spine_cy = 0.40   # posterior
    spine_r = 0.10

    spine_dist = np.sqrt(
        ((xn - spine_cx) / spine_r) ** 2 +
        ((yn - spine_cy) / spine_r) ** 2
    )
    spine_mask = (spine_dist <= 1.0) & body_mask
    spine_hu = rng.normal(800.0, 100.0, shape).astype(np.float32)
    spine_hu = np.clip(spine_hu, 500.0, 1200.0)
    volume[spine_mask] = spine_hu[spine_mask]

    # ------------------------------------------------------------------
    # 5. Ribs — thin arcs in the chest wall
    #    Present in upper ~65% of volume
    # ------------------------------------------------------------------
    rib_z_present = lung_z_frac < 0.3
    # Ribs are at the body boundary, in left/right and anterior regions
    rib_ring = (body_distance > 0.78) & (body_distance <= 0.92)
    # Only in the lateral and anterior aspects (not posterior where spine is)
    rib_angular = yn < 0.35  # anterior + lateral
    rib_mask_base = rib_ring & rib_angular & rib_z_present

    # Make ribs intermittent (gaps between ribs) along z
    rib_spacing = 8  # voxels per rib+gap
    z_phase = (z_idx % rib_spacing) < (rib_spacing * 0.55)  # rib thicker than gap
    rib_mask = rib_mask_base & z_phase

    rib_hu = rng.normal(650.0, 80.0, shape).astype(np.float32)
    rib_hu = np.clip(rib_hu, 400.0, 900.0)
    volume[rib_mask] = rib_hu[rib_mask]

    # ------------------------------------------------------------------
    # 6. Heart — left-center, upper half
    # ------------------------------------------------------------------
    heart_present = lung_z_frac < 0.05  # upper ~52%
    heart_cx = -0.08
    heart_cy = 0.08
    heart_rx = 0.20
    heart_ry = 0.22

    heart_dist = np.sqrt(
        ((xn - heart_cx) / heart_rx) ** 2 +
        ((yn - heart_cy) / heart_ry) ** 2
    )
    heart_mask = (heart_dist <= 1.0) & heart_present & body_mask & ~lung_mask
    heart_hu = rng.normal(48.0, 8.0, shape).astype(np.float32)
    volume[heart_mask] = heart_hu[heart_mask]

    # ------------------------------------------------------------------
    # 7. Liver — right side, lower half
    # ------------------------------------------------------------------
    liver_present = lung_z_frac > -0.15  # lower ~57%
    liver_cx = 0.30
    liver_cy = 0.05
    liver_rx = 0.22
    liver_ry = 0.25

    liver_dist = np.sqrt(
        ((xn - liver_cx) / liver_rx) ** 2 +
        ((yn - liver_cy) / liver_ry) ** 2
    )
    liver_mask = (liver_dist <= 1.0) & liver_present & body_mask & ~lung_mask & ~spine_mask
    liver_hu = rng.normal(62.0, 8.0, shape).astype(np.float32)
    volume[liver_mask] = liver_hu[liver_mask]

    # ------------------------------------------------------------------
    # 8. Spleen — left side, lower half, smaller than liver
    # ------------------------------------------------------------------
    spleen_present = lung_z_frac > -0.05  # lower ~52%
    spleen_cx = -0.32
    spleen_cy = 0.05
    spleen_rx = 0.10
    spleen_ry = 0.14

    spleen_dist = np.sqrt(
        ((xn - spleen_cx) / spleen_rx) ** 2 +
        ((yn - spleen_cy) / spleen_ry) ** 2
    )
    spleen_mask = (spleen_dist <= 1.0) & spleen_present & body_mask & ~lung_mask & ~spine_mask
    spleen_hu = rng.normal(45.0, 6.0, shape).astype(np.float32)
    volume[spleen_mask] = spleen_hu[spleen_mask]

    # ------------------------------------------------------------------
    # 9. Kidneys — posterior, lower third, left and right
    # ------------------------------------------------------------------
    kidney_present = lung_z_frac > 0.15  # lower ~42%
    kidney_cy = 0.30  # posterior

    # Right kidney
    r_kidney_cx = 0.22
    r_kidney_r = 0.08
    r_kidney_dist = np.sqrt(
        ((xn - r_kidney_cx) / r_kidney_r) ** 2 +
        ((yn - kidney_cy) / r_kidney_r) ** 2
    )
    r_kidney_mask = (r_kidney_dist <= 1.0) & kidney_present & body_mask & ~spine_mask
    kidney_hu = rng.normal(35.0, 6.0, shape).astype(np.float32)
    volume[r_kidney_mask] = kidney_hu[r_kidney_mask]

    # Left kidney
    l_kidney_cx = -0.22
    l_kidney_dist = np.sqrt(
        ((xn - l_kidney_cx) / r_kidney_r) ** 2 +
        ((yn - kidney_cy) / r_kidney_r) ** 2
    )
    l_kidney_mask = (l_kidney_dist <= 1.0) & kidney_present & body_mask & ~spine_mask
    volume[l_kidney_mask] = kidney_hu[l_kidney_mask]

    # ------------------------------------------------------------------
    # 10. Aorta — small circle anterior to spine
    # ------------------------------------------------------------------
    aorta_cx = 0.0
    aorta_cy = 0.22  # anterior to spine
    aorta_r = 0.04
    aorta_dist = np.sqrt(
        ((xn - aorta_cx) / aorta_r) ** 2 +
        ((yn - aorta_cy) / aorta_r) ** 2
    )
    aorta_mask = (aorta_dist <= 1.0) & body_mask & ~spine_mask
    aorta_hu = rng.normal(52.0, 6.0, shape).astype(np.float32)
    volume[aorta_mask] = aorta_hu[aorta_mask]

    # ------------------------------------------------------------------
    # 11. Trachea — air-filled tube anterior to spine, upper half
    # ------------------------------------------------------------------
    trachea_present = lung_z_frac < 0.15  # upper ~57%
    trachea_cx = 0.0
    trachea_cy = 0.18
    trachea_r = 0.03
    trachea_dist = np.sqrt(
        ((xn - trachea_cx) / trachea_r) ** 2 +
        ((yn - trachea_cy) / trachea_r) ** 2
    )
    trachea_mask = (trachea_dist <= 1.0) & trachea_present & body_mask
    volume[trachea_mask] = -900.0  # air in trachea

    # ------------------------------------------------------------------
    # 12. Global smoothing — mild Gaussian to reduce hard edges
    # ------------------------------------------------------------------
    volume = gaussian_filter(volume, sigma=0.6)

    # ------------------------------------------------------------------
    # Build metadata
    # ------------------------------------------------------------------
    metadata: Dict[str, Any] = {
        "width": int(nx),
        "height": int(ny),
        "depth": int(nz),
        "spacing": list(spacing),       # (z, y, x) in mm
        "window_presets": {
            name: {"windowLevel": float(p["window_level"]),
                   "windowWidth": float(p["window_width"])}
            for name, p in WINDOW_PRESETS.items()
        },
        "body_threshold_hu": -500.0,    # voxels >= this HU belong to the body
        "description": (
            "Synthetic upper-body CT phantom. NOT a real medical image. "
            "Contains geometric approximations of body outline (z-varying: "
            "neck→shoulders→chest→abdomen), lungs, spine, ribs, heart, "
            "liver, spleen, kidneys, aorta, trachea."
        ),
    }

    return volume, metadata

def generate_procedural_ct_phantom(
    size: int = 192,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Generate a procedural CT phantom using the existing synthetic volume logic.

    This mirrors the `/simulation/phantom` procedural branch so other callers
    can reuse the same output contract without duplicating generation code.
    """
    shape = (size, size, size)
    volume, metadata = generate_upper_body_ct_phantom(shape=shape)
    metadata["source"] = "procedural"
    return volume, None, metadata


# ---------------------------------------------------------------------------
# Atlas-based CT phantom — real CT volume from disk
# ---------------------------------------------------------------------------

def generate_atlas_ct_phantom(
    case_id: str = "s0001",
    size: int = 192,
    scan_direction: str = "head_to_feet",
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Load a real CT volume + organ label map from the phantom atlas directory.

    Reads:
        models/phantom_atlas/{case_id}/ct.nii.gz
        models/phantom_atlas/{case_id}/organs_label.nii.gz (optional)

    The CT volume is resampled so its largest dimension ≤ `size`.
    The 20-channel organ label mask is converted to a single-channel
    uint8 label map (0=background, 1–20=organs).

    The z-axis direction is determined from the NIfTI affine header.
    If the natural ordering does not match `scan_direction`, the volume
    and labels are flipped along z so that:

        head_to_feet → z=0 is superior (head/chest), z increases inferiorly.
        feet_to_head → z=0 is inferior (abdomen), z increases superiorly.

    Resize is done via isotropic zoom only — no cropping, full FOV preserved.

    Args:
        case_id:        Atlas case identifier (e.g. "s0001").
        size:           Target max dimension for the resampled volume.
        scan_direction: Desired z-axis scan order — 'head_to_feet' (default)
                        or 'feet_to_head'.

    Returns:
        (ct_volume, label_volume, metadata) tuple.
        ct_volume:    float32 ndarray of shape (z, y, x) with HU values.
        label_volume: uint8 ndarray of shape (z, y, x), or None if no label file.
        metadata:     dict with width, height, depth, spacing, source, case_id,
                      label_map, window_presets, body_threshold_hu, and debug
                      fields (original_shape, scan_direction, flipped_z,
                      label_nonzero_counts, slice_label_presence, etc.).
    """
    import nibabel as nib

    # ------------------------------------------------------------------
    # Validate scan_direction
    # ------------------------------------------------------------------
    if scan_direction not in ("head_to_feet", "feet_to_head"):
        raise ValueError(
            f"Invalid scan_direction '{scan_direction}'. "
            f"Must be 'head_to_feet' or 'feet_to_head'."
        )

    # Resolve project root (2 levels up from this file, or from env)
    project_root = os.environ.get(
        "MEDSIM_PROJECT_ROOT",
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        ),
    )
    atlas_dir = os.path.join(project_root, "models", "phantom_atlas", case_id)
    ct_path = os.path.join(atlas_dir, "ct.nii.gz")
    label_path = os.path.join(atlas_dir, "organs_label.nii.gz")

    # ------------------------------------------------------------------
    # 1. Load CT
    # ------------------------------------------------------------------
    if not os.path.isfile(ct_path):
        raise FileNotFoundError(
            f"Atlas CT file not found: {ct_path}. "
            f"Please place ct.nii.gz in models/phantom_atlas/{case_id}/"
        )

    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata(dtype=np.float32)  # shape: (x, y, z) in NIfTI

    # Determine z-direction from NIfTI affine
    # ---------------------------------------------------------------
    # The affine maps data indices to world coordinates (typically RAS:
    # Right, Anterior, Superior).  The third column (index 2) gives the
    # direction of increasing data-z in world space.
    #
    #   affine[:3, 2] = [rx, ry, rz]  (world direction of data-z axis)
    #
    #   rz > 0  →  data z increases toward Superior (head).
    #              After transpose, z=0 is at the inferior (feet) end.
    #              → natural order is FEET_TO_HEAD.
    #
    #   rz < 0  →  data z increases toward Inferior (feet).
    #              After transpose, z=0 is at the superior (head) end.
    #              → natural order is HEAD_TO_FEET.
    # ---------------------------------------------------------------
    best_affine = ct_img.header.get_best_affine()
    z_dir_world = best_affine[:3, 2]  # [rx, ry, rz]
    rz = float(z_dir_world[2])

    # When |rz| is tiny the z-axis is not the principal SI axis — still
    # use the sign logic but log a warning so the user can investigate.
    if abs(rz) < 1e-6:
        import logging
        logging.getLogger(__name__).warning(
            "NIfTI z-axis has near-zero S component (rz=%.6f). "
            "Scan-direction auto-detection may be unreliable for case %s.",
            rz, case_id,
        )

    nifti_natural_is_head_to_feet = rz < 0  # True if z=0 is head

    need_flip = False
    if scan_direction == "head_to_feet" and not nifti_natural_is_head_to_feet:
        need_flip = True
    elif scan_direction == "feet_to_head" and nifti_natural_is_head_to_feet:
        need_flip = True

    # NIfTI stores data (x, y, z) — transpose to (z, y, x) for our convention
    ct_volume_zxy = np.transpose(ct_data, (2, 1, 0)).copy()
    # ct_volume_zxy.shape = (nz_orig, ny_orig, nx_orig)

    # Original spacing from NIfTI header (x, y, z zooms)
    zooms = ct_img.header.get_zooms()[:3]  # (sx, sy, sz)
    orig_spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))  # → (z, y, x)

    spatial_shape = ct_volume_zxy.shape  # (nz, ny, nx)

    # ------------------------------------------------------------------
    # 2. Resample CT to target size (isotropic zoom — NO cropping)
    # ------------------------------------------------------------------
    max_dim = max(spatial_shape)
    if max_dim > size:
        zoom_factor = size / max_dim
    else:
        zoom_factor = 1.0

    # Isotropic zoom — preserves aspect ratio, max dimension ≤ size
    ct_zoom = (zoom_factor, zoom_factor, zoom_factor)

    # Use order=1 (linear) for CT — preserves HU values
    ct_resampled = zoom(
        ct_volume_zxy, ct_zoom, order=1, mode="constant", cval=-1000.0,
    )
    ct_resampled = ct_resampled.astype(np.float32)

    # Resulting spacing (isotropic)
    new_spacing = (
        orig_spacing[0] / zoom_factor if zoom_factor > 0 else orig_spacing[0],
        orig_spacing[1] / zoom_factor if zoom_factor > 0 else orig_spacing[1],
        orig_spacing[2] / zoom_factor if zoom_factor > 0 else orig_spacing[2],
    )

    # ------------------------------------------------------------------
    # 3. Flip z-axis if needed (so that z=0 matches scan_direction)
    # ------------------------------------------------------------------
    if need_flip:
        ct_resampled = ct_resampled[::-1, :, :].copy()

    nz, ny, nx = ct_resampled.shape

    # ------------------------------------------------------------------
    # 4. Load & convert organ labels (optional)
    # ------------------------------------------------------------------
    label_volume: Optional[np.ndarray] = None

    if os.path.isfile(label_path):
        label_img = nib.load(label_path)
        label_data = label_img.get_fdata(dtype=np.float32)

        # The label file can be:
        #   (a) 4-D one-hot: (20, x, y, z) — 20 channels first
        #   (b) 4-D one-hot: (x, y, z, 20) — channels last
        #   (c) 3-D single-channel label map
        # Detect and convert.

        if label_data.ndim == 4:
            # Find the axis of length 20 (the channel axis)
            channel_axis: Optional[int] = None
            for ax in range(4):
                if label_data.shape[ax] == 20:
                    channel_axis = ax
                    break

            if channel_axis is None:
                # Try to handle non-20-channel labels gracefully
                n_channels = label_data.shape[0] if label_data.shape[0] > 1 else label_data.shape[-1]
                raise ValueError(
                    f"Label file has 4 dimensions but none of size 20: "
                    f"shape={label_data.shape}. Found {n_channels} channels."
                )

            # Move channel axis to first position → (20, x, y, z)
            label_reordered = np.moveaxis(label_data, channel_axis, 0)
            # label_reordered.shape = (20, x, y, z)  [nibabel spatial convention]

            # Transpose spatial dims from (x,y,z) to (z,y,x):
            #   (20, x, y, z) → (20, z, y, x)
            label_reordered = np.transpose(label_reordered, (0, 3, 2, 1))

            # Resample each channel with nearest-neighbor BEFORE argmax
            # to preserve label boundaries. We'll resample the raw one-hot
            # channels individually, then argmax.
            resampled_channels = []
            for ch in range(label_reordered.shape[0]):
                ch_data = label_reordered[ch].astype(np.float32)
                ch_resampled = zoom(
                    ch_data, ct_zoom, order=0, mode="constant", cval=0.0,
                )
                # Flip if needed (before argmax so label indices stay in sync)
                if need_flip:
                    ch_resampled = ch_resampled[::-1, :, :].copy()
                resampled_channels.append(ch_resampled)

            # Stack → (20, nz, ny, nx)
            label_resampled_4d = np.stack(resampled_channels, axis=0)

            # Argmax along channel axis → label index 0..19
            label_map_raw = np.argmax(label_resampled_4d, axis=0)  # (nz, ny, nx)

            # Where all channels are ~0, set to 0 (background)
            all_zero = np.all(label_resampled_4d < 0.001, axis=0)
            label_map_raw[all_zero] = -1  # temporary sentinel

            label_volume = (label_map_raw + 1).astype(np.uint8)  # shift 0→1, 1→2, ...
            label_volume[label_map_raw == -1] = 0  # background

        elif label_data.ndim == 3:
            # Already a single-channel label map
            # Transpose from (x, y, z) to (z, y, x)
            label_map_zxy = np.transpose(label_data, (2, 1, 0)).copy()
            label_map_zxy = label_map_zxy.astype(np.uint8)

            # Resample label map (order=0 = nearest-neighbor)
            label_resampled = zoom(
                label_map_zxy, ct_zoom, order=0, mode="constant", cval=0,
            )
            label_volume = label_resampled.astype(np.uint8)

            # Flip if needed
            if need_flip:
                label_volume = label_volume[::-1, :, :].copy()

        else:
            raise ValueError(
                f"Unexpected label data shape: {label_data.shape}. "
                f"Expected 3-D or 4-D array."
            )

        # Sanity check
        if label_volume.shape != ct_resampled.shape:
            raise RuntimeError(
                f"Label shape {label_volume.shape} does not match "
                f"CT shape {ct_resampled.shape} after resampling."
            )

    # ------------------------------------------------------------------
    # 5. Build label statistics (for debugging & frontend display)
    # ------------------------------------------------------------------
    label_nonzero_counts: Dict[int, int] = {}
    slice_label_presence: Dict[str, list] = {}

    if label_volume is not None:
        # Count non-zero voxels per organ label
        for label_id in range(1, 21):
            count = int(np.sum(label_volume == label_id))
            if count > 0:
                label_nonzero_counts[int(label_id)] = count

        # Z-index ranges for key organ groups
        _organ_groups = {
            "lung": [10, 11, 12, 13, 14],       # all lung lobes
            "lung_left": [10, 13],               # left lung (lower + upper)
            "lung_right": [11, 12, 14],          # right lung (lower + middle + upper)
            "liver": [9],
            "spleen": [17],
            "kidney_left": [7],
            "kidney_right": [8],
            "pancreas": [15],
            "bladder": [20],
            "trachea": [19],
            "stomach": [18],
        }
        for organ_name, label_ids in _organ_groups.items():
            # Check which z-slices contain ANY of the labels in the group
            z_presence = np.any(
                np.isin(label_volume, label_ids), axis=(1, 2),
            )
            z_indices = np.where(z_presence)[0]
            if len(z_indices) > 0:
                slice_label_presence[organ_name] = [
                    int(z_indices[0]),
                    int(z_indices[-1]),
                ]

    # ------------------------------------------------------------------
    # 6. Build metadata
    # ------------------------------------------------------------------
    metadata: Dict[str, Any] = {
        # Dimensions
        "width": int(nx),
        "height": int(ny),
        "depth": int(nz),
        "spacing": [float(new_spacing[0]), float(new_spacing[1]), float(new_spacing[2])],

        # Source
        "source": "atlas",
        "case_id": case_id,

        # Shape / spacing debug info
        "original_shape": [int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2])],
        "output_shape": [int(nz), int(ny), int(nx)],
        "original_spacing": [float(orig_spacing[0]), float(orig_spacing[1]), float(orig_spacing[2])],
        "output_spacing": [float(new_spacing[0]), float(new_spacing[1]), float(new_spacing[2])],

        # Scan direction
        "scan_axis": "z",
        "scan_direction": scan_direction,
        "flipped_z": need_flip,
        "nifti_rz": rz,  # raw S-component from affine (debug)

        # Label info
        "label_map": {int(k): v for k, v in ORGAN_LABEL_MAP.items()},
        "label_nonzero_counts": label_nonzero_counts,
        "slice_label_presence": slice_label_presence,

        # Window presets
        "window_presets": {
            name: {"windowLevel": float(p["window_level"]),
                   "windowWidth": float(p["window_width"])}
            for name, p in WINDOW_PRESETS.items()
        },

        # Misc
        "body_threshold_hu": -500.0,
        "description": (
            f"Atlas-based CT phantom from case {case_id}. "
            f"Real CT volume resampled to ~{size}³ voxels "
            f"(shape {nz}×{ny}×{nx}). "
            f"Scan direction: {scan_direction} (flipped={need_flip}). "
            f"Original spacing: {orig_spacing}; resampled spacing: {new_spacing}. "
            f"NOT for medical diagnosis."
        ),
    }

    return ct_resampled, label_volume, metadata
