"""
Lesion Generator

Generates synthetic lesions in CT volumes with realistic
Hounsfield Unit distributions, shapes, and margins.

Supported lesion types:
- Tumor: Soft tissue density with heterogeneity
- Nodule: Small rounded opacity, typically in lung
- Cyst: Fluid-filled with near-water density
- Calcification: High-density calcium deposits
- Metastasis: Secondary malignant growth with irregular margins
"""

import logging
import os
import io
import base64
from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure
from app.core.config import settings

# When DEBUG_LESION is set, matplotlib PNGs are saved to this directory
_DEBUG_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "debug_output",
)

# Optional matplotlib for debug PNG generation within the generator
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

logger = logging.getLogger(__name__)


class LesionGenerator:
    """
    Synthetic lesion generator for CT medical imaging.

    Generates realistic lesion volumes with configurable:
    - Shape (spherical, ellipsoidal, irregular, lobulated, spiculated)
    - HU density (mean and standard deviation)
    - Margin characteristics (sharp vs. diffuse)
    - Internal structure (calcification, necrosis, heterogeneity)
    """

    # Default HU ranges for different lesion types
    LESION_HU_DEFAULTS = {
        "tumor": {"mean": 40, "std": 20, "description": "Soft tissue tumor"},
        "nodule": {"mean": -100, "std": 50, "description": "Pulmonary nodule"},
        "cyst": {"mean": 10, "std": 5, "description": "Fluid-filled cyst"},
        "calcification": {"mean": 300, "std": 100, "description": "Calcified lesion"},
        "metastasis": {"mean": 50, "std": 25, "description": "Metastatic lesion"},
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed or settings.SIMULATION_DEFAULT_SEED)

    def generate_preview(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fast preview of a lesion configuration.

        Returns summary statistics and a small voxel preview
        without generating the full volume.
        """
        center = (
            config.get("center_x", 0),
            config.get("center_y", 0),
            config.get("center_z", 0),
        )
        radii = (
            config.get("radius_x", 10),
            config.get("radius_y", 10),
            config.get("radius_z", 10),
        )

        # Generate small preview volume (32×32×32)
        preview_size = 32
        preview = self._generate_lesion_volume(
            shape=(preview_size, preview_size, preview_size),
            center=(preview_size // 2,) * 3,
            radii=(radii[0] * 0.3, radii[1] * 0.3, radii[2] * 0.3),
            hu_mean=config.get("hu_mean", 40),
            hu_std=config.get("hu_std", 20),
            margin_sharpness=config.get("margin_sharpness", 0.8),
            shape_type=config.get("shape", "spherical"),
            spiculation=config.get("spiculation_degree", 0.0),
        )

        return {
            "voxel_count": int(np.sum(preview != 0)),
            "hu_min": float(np.min(preview)),
            "hu_max": float(np.max(preview)),
            "hu_mean": float(np.mean(preview)),
            "hu_std": float(np.std(preview)),
            "volume_mm3": float(np.sum(preview != 0) * settings.SIMULATION_VOXEL_SIZE ** 3),
            "preview_shape": list(preview.shape),
            "center": list(center),
            "radii_mm": list(radii),
        }

    def generate_lesion(
        self,
        volume_shape: Tuple[int, int, int],
        config: Dict[str, Any],
        spacing: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """
        Generate a lesion within a volume of the given shape.

        Args:
            volume_shape: Shape of the target volume (z, y, x)
            config: Lesion configuration dictionary
            spacing: Voxel spacing (z, y, x) in mm. Used to convert mm radii
                     to voxel counts. Falls back to SIMULATION_VOXEL_SIZE if None.

        Returns:
            numpy array of HU values matching volume_shape
        """
        lesion_type = config.get("lesion_type", "tumor")
        hu_defaults = self.LESION_HU_DEFAULTS.get(lesion_type, self.LESION_HU_DEFAULTS["tumor"])

        center = (
            max(0.0, min(config.get("center_z", float(volume_shape[0] // 2)), float(volume_shape[0] - 1))),
            max(0.0, min(config.get("center_y", float(volume_shape[1] // 2)), float(volume_shape[1] - 1))),
            max(0.0, min(config.get("center_x", float(volume_shape[2] // 2)), float(volume_shape[2] - 1))),
        )
        # Convert mm radii to voxel counts using actual spacing
        if spacing is not None:
            vz, vy, vx = spacing
        else:
            vz = vy = vx = settings.SIMULATION_VOXEL_SIZE
        radii = (
            config.get("radius_z", 10) / vz,
            config.get("radius_y", 10) / vy,
            config.get("radius_x", 10) / vx,
        )

        hu_mean = config.get("hu_mean", hu_defaults["mean"])
        hu_std = config.get("hu_std", hu_defaults["std"])

        # ── DEBUG: Log config-to-params resolution ──
        _resolved_shape = config.get("shape", "spherical")
        _resolved_spic = config.get("spiculation_degree", 0.0)
        logger.debug(
            "==== generate_lesion CONFIG DUMP ====\n"
            "  config dict keys: %s\n"
            "  spacing (z,y,x):  %s\n"
            "  --- resolved params ---\n"
            "  center (z,y,x):    (%.1f, %.1f, %.1f)\n"
            "  radii_mm (z,y,x):  (%.1f, %.1f, %.1f)\n"
            "  radii_voxel(z,y,x):(%.2f, %.2f, %.2f)\n"
            "  shape (from config): [%s]\n"
            "  spiculation_degree:  %.3f\n"
            "  hu_mean=%.1f  hu_std=%.1f\n"
            "  margin_sharpness=%.3f",
            ", ".join(config.keys()),
            str(spacing),
            center[0], center[1], center[2],
            config.get("radius_z", 10), config.get("radius_y", 10), config.get("radius_x", 10),
            radii[0], radii[1], radii[2],
            _resolved_shape,
            _resolved_spic,
            hu_mean, hu_std,
            config.get("margin_sharpness", 0.8),
        )

        return self._generate_lesion_volume(
            shape=volume_shape,
            center=center,
            radii=radii,
            hu_mean=hu_mean,
            hu_std=hu_std,
            margin_sharpness=config.get("margin_sharpness", 0.8),
            shape_type=config.get("shape", "spherical"),
            spiculation=config.get("spiculation_degree", 0.0),
        )

    def _debug_save_lesion_mask_png(
        self,
        lesion_mask: np.ndarray,
        distance: np.ndarray,
        label: str,
    ) -> None:
        """Save a diagnostic PNG of the lesion mask middle slice (from within generator)."""
        if not _HAS_MPL:
            return
        try:
            os.makedirs(_DEBUG_OUTPUT_DIR, exist_ok=True)
            nonzero = np.argwhere(lesion_mask)
            if len(nonzero) == 0:
                return
            cz = int(np.median(nonzero[:, 0]))

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Mask
            im0 = axes[0].imshow(lesion_mask[cz, :, :].astype(np.uint8) * 255, cmap="gray", aspect="equal")
            axes[0].set_title(f"Lesion Mask (axial z={cz})")
            plt.colorbar(im0, ax=axes[0], shrink=0.75)

            # Distance field at same slice (shows shape deformation)
            im1 = axes[1].imshow(distance[cz, :, :], cmap="viridis", aspect="equal")
            axes[1].set_title(f"Distance Field (axial z={cz})")
            plt.colorbar(im1, ax=axes[1], shrink=0.75)

            # Contour overlay: mask boundary on distance
            dist_at_z = distance[cz, :, :]
            mask_at_z = lesion_mask[cz, :, :]
            overlay = np.zeros((*dist_at_z.shape, 4), dtype=float)
            # Normalize distance for display
            d_min, d_max = dist_at_z.min(), dist_at_z.max()
            if d_max > d_min:
                overlay[:, :, 0] = (dist_at_z - d_min) / (d_max - d_min)  # R channel
            else:
                overlay[:, :, 0] = 0
            overlay[:, :, 1] = 0.3  # some green
            overlay[:, :, 2] = 0.5  # some blue
            overlay[:, :, 3] = 0.6  # alpha
            # Mark mask contour
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(mask_at_z)
            boundary = mask_at_z.astype(bool) & (~eroded)
            overlay[boundary, :] = [1.0, 0.0, 0.0, 1.0]  # red boundary
            axes[2].imshow(overlay)
            axes[2].set_title(f"Distance + Mask Boundary (axial z={cz})")

            plt.tight_layout()
            path = os.path.join(_DEBUG_OUTPUT_DIR, f"{label}_generator_mask_debug.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Generator DEBUG PNG saved: %s", path)
        except Exception as e:
            logger.warning("Failed to save generator debug PNG: %s", e)

    def _debug_log_deformation(
        self,
        distance_before: np.ndarray,
        distance_after: np.ndarray,
        shape_type: str,
    ) -> None:
        """Log deformation diagnostic statistics."""
        delta = distance_before - distance_after
        logger.debug(
            "==== SHAPE DEFORMATION [%s] ====\n"
            "  deformation:  min=%+.6f  max=%+.6f  mean=%+.6f  std=%.6f\n"
            "  |deformation|>0.01: %d voxels (%.2f%% of volume)\n"
            "  distance_before: min=%.4f  max=%.4f  mean=%.4f\n"
            "  distance_after:  min=%.4f  max=%.4f  mean=%.4f\n"
            "  distance_after<1.0: %d voxels (mask pre-sigmoid)",
            shape_type,
            float(np.min(delta)), float(np.max(delta)),
            float(np.mean(delta)), float(np.std(delta)),
            int(np.sum(np.abs(delta) > 0.01)),
            100.0 * np.sum(np.abs(delta) > 0.01) / max(delta.size, 1),
            float(np.min(distance_before)), float(np.max(distance_before)),
            float(np.mean(distance_before)),
            float(np.min(distance_after)), float(np.max(distance_after)),
            float(np.mean(distance_after)),
            int(np.sum(distance_after < 1.0)),
        )

    def _generate_lesion_volume(
        self,
        shape: Tuple[int, int, int],
        center: Tuple[float, float, float],
        radii: Tuple[float, float, float],
        hu_mean: float,
        hu_std: float,
        margin_sharpness: float,
        shape_type: str = "spherical",
        spiculation: float = 0.0,
    ) -> np.ndarray:
        """Generate the lesion voxel grid with appropriate shape and HU values."""
        z, y, x = np.indices(shape, dtype=float)
        cz, cy, cx = center
        rz, ry, rx = radii

        # ── DEBUG: Entry diagnostics ──
        logger.debug(
            "==== _generate_lesion_volume ENTRY ====\n"
            "  shape_type:        [%s]\n"
            "  volume_shape:      %s\n"
            "  center (z,y,x):    (%.2f, %.2f, %.2f)\n"
            "  radii_voxel(z,y,x):(%.2f, %.2f, %.2f)\n"
            "  hu_mean=%.1f  hu_std=%.1f\n"
            "  margin_sharpness=%.3f  spiculation=%.3f",
            shape_type, str(shape),
            cz, cy, cx,
            rz, ry, rx,
            hu_mean, hu_std,
            margin_sharpness, spiculation,
        )

        # Normalized distance from center
        distance = np.sqrt(
            ((z - cz) / rz) ** 2 +
            ((y - cy) / ry) ** 2 +
            ((x - cx) / rx) ** 2
        )

        # ── Distance before deformation (for later comparison) ──
        distance_before = distance.copy()

        # Apply shape modifications
        if shape_type == "lobulated":
            distance = self._apply_lobulation(distance, z, y, x, cz, cy, cx, rz, ry, rx)
            self._debug_log_deformation(distance_before, distance, "lobulated")
        elif shape_type == "spiculated":
            distance = self._apply_spiculation(distance, z, y, x, cz, cy, cx, rz, ry, rx, spiculation)
            self._debug_log_deformation(distance_before, distance, "spiculated")
        elif shape_type == "irregular":
            distance = self._apply_irregularity(distance, z, y, x, cz, cy, cx)
            self._debug_log_deformation(distance_before, distance, "irregular")
        else:
            # spherical — no deformation, log that shape_type was "spherical"
            self._debug_log_deformation(distance_before, distance, "spherical (no-op)")

        # Soft margin using sigmoid
        margin_width = 1.0 - margin_sharpness
        if margin_width > 0.01:
            mask = 1.0 / (1.0 + np.exp((distance - 1.0) / (margin_width * 0.2)))
        else:
            mask = (distance <= 1.0).astype(float)

        # Generate HU values with heterogeneity
        hu_volume = np.zeros(shape, dtype=np.float32)
        lesion_voxels = mask > 0.01

        # Base HU with noise
        hu_values = self.rng.normal(hu_mean, hu_std, shape)
        hu_volume[lesion_voxels] = hu_values[lesion_voxels]

        # Apply smooth transition at margins
        hu_volume = hu_volume * mask

        # Add internal structure (optional calcification/necrosis)
        # TODO: Add calcification and necrosis patterns

        # ── DEBUG: lesion generation diagnostics ──
        _debug_mask = mask > 0.01
        _debug_mask_count = int(np.sum(_debug_mask))
        _debug_total_voxels = shape[0] * shape[1] * shape[2]
        if _debug_mask_count > 0:
            _debug_lesion_hu = hu_volume[_debug_mask]
            logger.debug(
                "==== LESION DEBUG ====\n"
                "  voxels:      %d\n"
                "  ratio:       %.6f (%.4f%%)\n"
                "  hu_mean:     %.2f\n"
                "  hu_min:      %.2f\n"
                "  hu_max:      %.2f\n"
                "  hu_std:      %.2f\n"
                "  center:      (%.1f, %.1f, %.1f)\n"
                "  radii_voxel: (%.1f, %.1f, %.1f)\n"
                "  radius_mm (approx): (%.1f, %.1f, %.1f)\n"
                "  shape:       %s\n"
                "  margin_sharpness: %.2f\n"
                "  shape_type:  %s"
                "  spiculation: %.3f",
                _debug_mask_count,
                _debug_mask_count / max(_debug_total_voxels, 1),
                _debug_mask_count / max(_debug_total_voxels, 1) * 100,
                float(np.mean(_debug_lesion_hu)),
                float(np.min(_debug_lesion_hu)),
                float(np.max(_debug_lesion_hu)),
                float(np.std(_debug_lesion_hu)),
                center[0], center[1], center[2],
                radii[0], radii[1], radii[2],
                radii[0] * settings.SIMULATION_VOXEL_SIZE,
                radii[1] * settings.SIMULATION_VOXEL_SIZE,
                radii[2] * settings.SIMULATION_VOXEL_SIZE,
                shape,
                margin_sharpness,
                shape_type,
                spiculation,
            )
        else:
            logger.debug(
                "==== LESION DEBUG ==== (voxels=0)\n"
                "  radii_voxel=(%.1f, %.1f, %.1f)\n"
                "  center=(%.1f, %.1f, %.1f)\n"
                "  shape=%s\n"
                "  distance_min=%.4f  distance_max=%.4f  distance_mean=%.4f\n"
                "  mask_sum=%d (threshold > 0.01)",
                radii[0], radii[1], radii[2],
                center[0], center[1], center[2],
                shape,
                float(np.min(distance)), float(np.max(distance)), float(np.mean(distance)),
                _debug_mask_count,
            )

        # ── DEBUG: Save lesion mask PNG from within generator ──
        if _debug_mask_count > 0 and _debug_mask.any():
            self._debug_save_lesion_mask_png(
                lesion_mask=_debug_mask,
                distance=distance,
                label=f"gen_{shape_type}_cz{int(cz)}",
            )

        return hu_volume

    def _apply_lobulation(
        self,
        distance: np.ndarray,
        z: np.ndarray, y: np.ndarray, x: np.ndarray,
        cz: float, cy: float, cx: float,
        rz: float, ry: float, rx: float,
    ) -> np.ndarray:
        """Add lobulated contours to the lesion shape."""
        # Multiple overlapping sinusoidal deformations with stronger amplitude
        for i in range(3):
            angle = self.rng.uniform(0, 2 * np.pi, 3)
            freq = self.rng.uniform(2, 4, 3)
            amp = self.rng.uniform(0.2, 0.5)  # was 0.1~0.3 — stronger undulation
            deformation = amp * (
                np.sin(freq[0] * (z - cz) / rz + angle[0]) *
                np.sin(freq[1] * (y - cy) / ry + angle[1]) *
                np.sin(freq[2] * (x - cx) / rx + angle[2])
            )
            distance = distance / (1.0 + deformation)
        return distance

    def _apply_spiculation(
        self,
        distance: np.ndarray,
        z: np.ndarray, y: np.ndarray, x: np.ndarray,
        cz: float, cy: float, cx: float,
        rz: float, ry: float, rx: float,
        spiculation_degree: float,
    ) -> np.ndarray:
        """Add spiculated (spiky) margins to simulate malignancy.

        Uses surface-weighted deformation to create thin outward protrusions
        instead of the old multiplicative approach that caused uniform expansion.
        Each spike is a narrow angular lobe (power 8) that only pushes outward
        near the lesion boundary (distance ≈ 1.0), leaving the interior intact.
        """
        theta = np.arctan2(y - cy, x - cx)
        phi = np.arctan2(z - cz, np.sqrt((x - cx) ** 2 + (y - cy) ** 2))

        # Number and strength of spikes scales with spiculation_degree
        n_spikes = 8 + int(spiculation_degree * 8)  # 8-16 spikes
        spike_angles = self.rng.uniform(0, 2 * np.pi, n_spikes)
        # Base amp gives spike length 0 ~ spike_length_max in normalized distance
        spike_amps = self.rng.uniform(0.15, 0.35, n_spikes) * spiculation_degree

        # Build angular spiculation field (narrow lobes with power 8)
        spiculation_field = np.zeros_like(distance)
        for sa, amp in zip(spike_angles, spike_amps):
            angular_dist = np.cos(theta - sa) * np.cos(phi)
            # Power 8 gives very narrow angular lobes — no overlap between spikes
            spiculation_field += amp * np.maximum(angular_dist, 0) ** 8

        # Surface-weighted deformation: only affect voxels near the boundary.
        # Gaussian centered at distance=1.0 with width that controls spike length.
        spike_length = 0.2 + 0.3 * spiculation_degree  # 0.2 (subtle) to 0.5 (long)
        surface_weight = np.exp(-((distance - 1.0) / spike_length) ** 2)

        # Subtract (not divide) to push outward ONLY near the surface.
        # Voxels far inside (distance << 1) are untouched.
        # Voxels far outside (distance >> 1) are untouched.
        return distance - spiculation_field * surface_weight

    def _apply_irregularity(
        self,
        distance: np.ndarray,
        z: np.ndarray, y: np.ndarray, x: np.ndarray,
        cz: float, cy: float, cx: float,
    ) -> np.ndarray:
        """Add Perlin-like irregular deformation to the lesion shape.

        Larger raw noise + smaller sigma + stronger multiplier ensures
        visible boundary irregularity (was smoothing to near-zero effect).
        """
        noise = self.rng.uniform(-0.8, 0.8, distance.shape)  # was ±0.2
        noise = gaussian_filter(noise, sigma=1.5)  # was sigma=3
        return distance * (1.0 + noise * 1.0)  # was *0.5
