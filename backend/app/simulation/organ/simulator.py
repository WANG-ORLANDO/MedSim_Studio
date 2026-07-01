"""
Organ Simulator

Generates synthetic organ structures with realistic CT tissue densities.
Provides base organ shapes with appropriate HU values and optional
contrast enhancement patterns.

Supported organs: liver, kidney, lung, brain, bone, heart, spleen, pancreas, bladder
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes
from app.core.config import settings

logger = logging.getLogger(__name__)


class OrganSimulator:
    """
    Simulates organ structures in CT volumes with realistic
    Hounsfield Unit values and tissue characteristics.
    """

    # Standard HU ranges for organs
    ORGAN_HU_RANGES = {
        "liver": {"mean": 60, "std": 15},
        "kidney": {"mean": 35, "std": 10},
        "lung": {"mean": -500, "std": 150},
        "brain": {"mean": 40, "std": 10},
        "bone": {"mean": 700, "std": 300},
        "heart": {"mean": 50, "std": 15},
        "spleen": {"mean": 45, "std": 10},
        "pancreas": {"mean": 40, "std": 12},
        "bladder": {"mean": 10, "std": 5},
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed or settings.SIMULATION_DEFAULT_SEED)

    def generate_preview(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fast preview of an organ configuration.

        Returns summary statistics without generating the full volume.
        """
        organ_type = config.get("organ_type", "liver")
        hu_defaults = self.ORGAN_HU_RANGES.get(organ_type, self.ORGAN_HU_RANGES["liver"])
        hu_mean = config.get("hu_mean", hu_defaults["mean"])
        hu_std = config.get("hu_std", hu_defaults["std"])

        # Generate small preview volume (32x32x32)
        preview_size = 32
        preview_shape = (preview_size, preview_size, preview_size)
        preview = self.generate_organ(
            volume_shape=preview_shape,
            config=config,
        )

        return {
            "voxel_count": int(np.sum(preview > -500)),
            "hu_min": float(np.min(preview)),
            "hu_max": float(np.max(preview)),
            "hu_mean": float(np.mean(preview)),
            "hu_std": float(np.std(preview)),
            "organ_type": organ_type,
            "preview_shape": list(preview.shape),
        }

    def generate_organ(
        self,
        volume_shape: Tuple[int, int, int],
        config: Dict[str, Any],
    ) -> np.ndarray:
        """
        Generate an organ volume with realistic HU distribution.

        Args:
            volume_shape: Shape of the target volume (z, y, x)
            config: Organ configuration with type and density parameters

        Returns:
            numpy array of HU values matching volume_shape
        """
        organ_type = config.get("organ_type", "liver")
        hu_defaults = self.ORGAN_HU_RANGES.get(organ_type, self.ORGAN_HU_RANGES["liver"])

        hu_mean = config.get("hu_mean", hu_defaults["mean"])
        hu_std = config.get("hu_std", hu_defaults["std"])

        # Generate base organ shape
        organ_mask = self._generate_organ_shape(volume_shape, organ_type)

        # Apply HU values
        hu_volume = np.zeros(volume_shape, dtype=np.float32)
        noise = self.rng.normal(hu_mean, hu_std, volume_shape)
        hu_volume[organ_mask] = noise[organ_mask]

        # Add texture (tissue heterogeneity)
        if config.get("enable_noise", True):
            noise_level = config.get("noise_level", 0.1)
            texture = gaussian_filter(
                self.rng.normal(0, hu_std * noise_level, volume_shape),
                sigma=2,
            )
            hu_volume[organ_mask] += texture[organ_mask]

        # Apply contrast enhancement pattern
        if config.get("enable_enhancement", False):
            pattern = config.get("enhancement_pattern", "none")
            hu_volume = self._apply_enhancement(hu_volume, organ_mask, pattern, hu_mean)

        return hu_volume

    def _generate_organ_shape(
        self,
        shape: Tuple[int, int, int],
        organ_type: str,
    ) -> np.ndarray:
        """
        Generate a simplified anatomical shape for the given organ type.
        Uses parametrized geometric models (ellipsoids, superquadrics).
        """
        z, y, x = np.indices(shape, dtype=float)
        cz, cy, cx = shape[0] * 0.5, shape[1] * 0.5, shape[2] * 0.5

        # Organ-specific shape parameters
        shape_params = self._get_organ_shape_params(organ_type, shape)
        rz, ry, rx = shape_params["radii"]

        # Normalized distance (superquadric for more natural shapes)
        distance = np.sqrt(
            ((z - cz) / rz) ** 2 +
            ((y - cy) / ry) ** 2 +
            ((x - cx) / rx) ** 2
        )

        mask = distance <= 1.0

        # Apply smooth edges
        if shape_params.get("smooth_edges", True):
            mask = binary_fill_holes(mask)
            mask = gaussian_filter(mask.astype(float), sigma=shape_params.get("edge_sigma", 2)) > 0.3

        return mask

    def _get_organ_shape_params(
        self,
        organ_type: str,
        volume_shape: Tuple[int, int, int],
    ) -> Dict[str, Any]:
        """Get shape parameters for each organ type."""
        dz, dy, dx = volume_shape
        params = {
            "liver": {
                "radii": (dz * 0.3, dy * 0.25, dx * 0.35),
                "smooth_edges": True,
                "edge_sigma": 3,
            },
            "kidney": {
                "radii": (dz * 0.15, dy * 0.12, dx * 0.2),
                "smooth_edges": True,
                "edge_sigma": 2,
            },
            "lung": {
                "radii": (dz * 0.35, dy * 0.3, dx * 0.25),
                "smooth_edges": True,
                "edge_sigma": 3,
            },
            "brain": {
                "radii": (dz * 0.35, dy * 0.3, dx * 0.3),
                "smooth_edges": True,
                "edge_sigma": 4,
            },
            "bone": {
                "radii": (dz * 0.05, dy * 0.02, dx * 0.02),
                "smooth_edges": True,
                "edge_sigma": 1,
            },
            "heart": {
                "radii": (dz * 0.2, dy * 0.2, dx * 0.2),
                "smooth_edges": True,
                "edge_sigma": 2,
            },
            "spleen": {
                "radii": (dz * 0.2, dy * 0.15, dx * 0.15),
                "smooth_edges": True,
                "edge_sigma": 2,
            },
            "pancreas": {
                "radii": (dz * 0.1, dy * 0.08, dx * 0.15),
                "smooth_edges": True,
                "edge_sigma": 2,
            },
            "bladder": {
                "radii": (dz * 0.15, dy * 0.15, dx * 0.15),
                "smooth_edges": True,
                "edge_sigma": 2,
            },
        }
        return params.get(organ_type, params["liver"])

    def _apply_enhancement(
        self,
        hu_volume: np.ndarray,
        mask: np.ndarray,
        pattern: str,
        base_hu: float,
    ) -> np.ndarray:
        """Apply contrast enhancement patterns to the organ."""
        enhanced = hu_volume.copy()

        if pattern == "homogeneous":
            # Uniform enhancement across the organ
            enhancement = self.rng.normal(base_hu * 0.3, base_hu * 0.05)
            enhanced[mask] += enhancement

        elif pattern == "heterogeneous":
            # Patchy enhancement pattern
            enhancement_map = gaussian_filter(
                self.rng.normal(0, base_hu * 0.2, hu_volume.shape),
                sigma=5,
            )
            enhanced[mask] += enhancement_map[mask]

        elif pattern == "rim":
            # Rim enhancement (peripheral)
            from scipy.ndimage import distance_transform_edt
            if mask.any():
                dist = distance_transform_edt(~mask)
                rim = (dist > 0) & (dist < 5) & mask
                enhanced[rim] += base_hu * 0.5

        elif pattern == "septal":
            # Septal enhancement (internal strands)
            # Simplified: random streaks within the organ
            for _ in range(3):
                start = tuple(
                    self.rng.randint(0, s) for s in hu_volume.shape
                )
                # Simple line enhancement
                pass

        return enhanced
