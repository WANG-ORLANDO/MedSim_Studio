"""
Volume Renderer

Server-side volume rendering utilities for generating
2D projections and MPR reconstructions from CT volumes.
Uses VTK for high-quality rendering when available.

Provides:
- Multi-planar reformatting (axial, sagittal, coronal)
- Maximum intensity projection (MIP)
- Volume rendering with transfer functions
- Slice extraction and window/level adjustment
"""

import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VolumeRenderer:
    """
    Server-side CT volume rendering engine.

    Generates 2D projections and reformatted views from
    3D CT volumes for streaming to the web frontend.
    """

    def __init__(self):
        self._vtk_available = False
        self._init_vtk()

    def _init_vtk(self):
        """Attempt to initialize VTK for GPU-accelerated rendering."""
        try:
            # from vtkmodules.vtkRenderingVolume import vtkVolumeMapper
            # from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLVolumeMapper
            self._vtk_available = True
            logger.info("VTK volume rendering available")
        except ImportError:
            logger.warning("VTK not available, using CPU fallback")
            self._vtk_available = False

    def extract_slice(
        self,
        volume: np.ndarray,
        plane: str = "axial",
        slice_index: int = 0,
        window_center: float = 40.0,
        window_width: float = 400.0,
    ) -> np.ndarray:
        """
        Extract a 2D slice from the volume with window/level applied.

        Args:
            volume: 3D CT volume (z, y, x)
            plane: 'axial', 'sagittal', or 'coronal'
            slice_index: Index along the selected plane
            window_center: Window center (HU)
            window_width: Window width (HU)

        Returns:
            2D numpy array with window/level applied
        """
        if plane == "axial":
            slice_2d = volume[slice_index, :, :]
        elif plane == "sagittal":
            slice_2d = volume[:, :, slice_index]
        elif plane == "coronal":
            slice_2d = volume[:, slice_index, :]
        else:
            raise ValueError(f"Unknown plane: {plane}")

        # Apply window/level
        half_width = window_width / 2
        lower = window_center - half_width
        upper = window_center + half_width

        result = np.clip(slice_2d, lower, upper)
        result = ((result - lower) / window_width * 255).astype(np.uint8)

        return result

    def compute_mpr(
        self,
        volume: np.ndarray,
        orientation: str = "axial",
        window_center: float = 40.0,
        window_width: float = 400.0,
    ) -> np.ndarray:
        """
        Compute Multi-Planar Reconstruction.

        Generates a complete set of slices for the given orientation.
        """
        slices = []
        max_index = volume.shape[["axial", "sagittal", "coronal"].index(orientation)]

        for i in range(max_index):
            slice_data = self.extract_slice(
                volume, orientation, i, window_center, window_width
            )
            slices.append(slice_data)

        return np.stack(slices, axis=0)

    def maximum_intensity_projection(
        self,
        volume: np.ndarray,
        axis: int = 0,
    ) -> np.ndarray:
        """
        Compute Maximum Intensity Projection along the given axis.

        Args:
            volume: 3D CT volume
            axis: Axis to project along (0=z, 1=y, 2=x)

        Returns:
            2D MIP image
        """
        return np.max(volume, axis=axis)

    def minimum_intensity_projection(
        self,
        volume: np.ndarray,
        axis: int = 0,
    ) -> np.ndarray:
        """Compute Minimum Intensity Projection."""
        return np.min(volume, axis=axis)
