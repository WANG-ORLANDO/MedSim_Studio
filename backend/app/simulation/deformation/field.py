"""
Deformation Field Computation

Computes deformation fields for simulating anatomical variations,
organ motion, and non-rigid registration. Supports multiple
deformation models including B-spline and Demons.

Used for:
- Simulating patient motion
- Generating training data for registration networks
- Anatomical variation augmentation
"""

import logging
from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

logger = logging.getLogger(__name__)


class DeformationField:
    """
    Computes and applies deformation fields to medical images.

    Supports rigid, affine, and non-rigid (B-spline) deformations
    for simulating anatomical variations.
    """

    def __init__(self, shape: Tuple[int, int, int]):
        """
        Initialize deformation field for a given volume shape.

        Args:
            shape: Shape of the target volume (z, y, x)
        """
        self.shape = shape
        self.field = None

    def generate_bspline_field(
        self,
        control_point_spacing: int = 32,
        magnitude: float = 10.0,
        smoothing_sigma: float = 3.0,
    ) -> np.ndarray:
        """
        Generate a B-spline deformation field.

        Args:
            control_point_spacing: Spacing between control points in voxels
            magnitude: Maximum deformation magnitude in voxels
            smoothing_sigma: Smoothing kernel sigma

        Returns:
            Deformation field of shape (3, *shape) where 3 = (dz, dy, dx)
        """
        rng = np.random.default_rng()

        # Create coarse control point grid
        grid_shape = tuple(
            max(3, s // control_point_spacing + 2) for s in self.shape
        )

        # Random displacements at control points
        control_field = rng.uniform(-magnitude, magnitude, (3, *grid_shape))

        # Smooth the control field
        smoothed = np.zeros_like(control_field)
        for i in range(3):
            smoothed[i] = gaussian_filter(control_field[i], sigma=smoothing_sigma)

        # Upsample to full resolution using linear interpolation
        # (B-spline approximation via interpolation)
        field = np.zeros((3, *self.shape), dtype=np.float32)

        z, y, x = np.indices(self.shape, dtype=float)
        grid_z = np.linspace(0, grid_shape[0] - 1, self.shape[0])
        grid_y = np.linspace(0, grid_shape[1] - 1, self.shape[1])
        grid_x = np.linspace(0, grid_shape[2] - 1, self.shape[2])

        for i in range(3):
            # Trilinear interpolation of control field
            field[i] = self._trilinear_interpolation(
                smoothed[i], grid_z, grid_y, grid_x
            )

        self.field = field
        return field

    def apply_deformation(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply the deformation field to a volume.

        Args:
            volume: Input volume to deform

        Returns:
            Deformed volume
        """
        if self.field is None:
            raise ValueError("Deformation field not generated. Call generate_bspline_field first.")

        # Create sampling grid
        z, y, x = np.indices(self.shape, dtype=float)

        # Apply displacement
        sample_z = z + self.field[0]
        sample_y = y + self.field[1]
        sample_x = x + self.field[2]

        # Clip to volume bounds
        for coord, max_val in [(sample_z, self.shape[0] - 1),
                                 (sample_y, self.shape[1] - 1),
                                 (sample_x, self.shape[2] - 1)]:
            np.clip(coord, 0, max_val, out=coord)

        # Interpolate
        deformed = map_coordinates(
            volume,
            [sample_z, sample_y, sample_x],
            order=1,
            mode='nearest',
        )

        return deformed.reshape(self.shape)

    @staticmethod
    def _trilinear_interpolation(
        volume: np.ndarray,
        z_grid: np.ndarray,
        y_grid: np.ndarray,
        x_grid: np.ndarray,
    ) -> np.ndarray:
        """Simple trilinear interpolation for upsampling."""
        from scipy.interpolate import RegularGridInterpolator

        shape = volume.shape
        points = (np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        interpolator = RegularGridInterpolator(points, volume, bounds_error=False, fill_value=0)

        zz, yy, xx = np.meshgrid(z_grid, y_grid, x_grid, indexing='ij')
        result = interpolator(np.stack([zz, yy, xx], axis=-1))

        return result

    def compute_jacobian_determinant(self) -> np.ndarray:
        """
        Compute the Jacobian determinant of the deformation field.
        Used to detect regions of expansion (>1) and contraction (<1).
        """
        if self.field is None:
            raise ValueError("Deformation field not generated.")

        # Compute spatial gradients of the deformation field
        J = np.zeros((*self.shape, 3, 3), dtype=np.float32)

        spacing = (1.0, 1.0, 1.0)
        for i in range(3):
            for j in range(3):
                # Central difference gradient
                slice_before = np.zeros(self.shape)
                slice_after = np.zeros(self.shape)

                if j == 0:  # z direction
                    slice_before[1:] = self.field[i][:-1]
                    slice_after[:-1] = self.field[i][1:]
                elif j == 1:  # y direction
                    slice_before[:, 1:] = self.field[i][:, :-1]
                    slice_after[:, :-1] = self.field[i][:, 1:]
                else:  # x direction
                    slice_before[:, :, 1:] = self.field[i][:, :, :-1]
                    slice_after[:, :, :-1] = self.field[i][:, :, 1:]

                J[..., i, j] = (slice_after - slice_before) / (2 * spacing[j])

        # Add identity matrix
        for i in range(3):
            J[..., i, i] += 1.0

        # Compute determinant
        det = (
            J[..., 0, 0] * (J[..., 1, 1] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 1]) -
            J[..., 0, 1] * (J[..., 1, 0] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 0]) +
            J[..., 0, 2] * (J[..., 1, 0] * J[..., 2, 1] - J[..., 1, 1] * J[..., 2, 0])
        )

        return det
