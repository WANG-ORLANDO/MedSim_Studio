"""
CT parameter simulation core for MVP preview generation.

This module applies image-domain approximations of CT acquisition and
reconstruction parameters to a HU volume using only NumPy/SciPy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, laplace, zoom


Spacing3D = Tuple[float, float, float]

DEFAULT_MAS_BY_DOSE_LEVEL: Dict[str, int] = {
    "low": 50,
    "standard": 150,
    "high": 300,
}


def _center_slice_stats(volume: np.ndarray) -> Dict[str, float]:
    center_idx = int(volume.shape[0] // 2)
    center_slice = volume[center_idx]
    return {
        "slice_index": center_idx,
        "min": float(np.min(center_slice)),
        "max": float(np.max(center_slice)),
        "mean": float(np.mean(center_slice)),
        "std": float(np.std(center_slice)),
    }


def _clipped_float32(volume: np.ndarray) -> np.ndarray:
    return np.clip(volume, -1024.0, 3071.0).astype(np.float32, copy=False)


def _resolve_params(params: Dict[str, Any]) -> Dict[str, Any]:
    dose_level = params["dose_level"]
    resolved_mas = int(params.get("mAs") or DEFAULT_MAS_BY_DOSE_LEVEL[dose_level])
    return {
        "slice_thickness_mm": float(params["slice_thickness_mm"]),
        "dose_level": dose_level,
        "mAs": resolved_mas,
        "dose_level_reference_mAs": int(DEFAULT_MAS_BY_DOSE_LEVEL[dose_level]),
        "kVp": int(params["kVp"]),
        "pitch": float(params["pitch"]),
        "fov_mm": float(params["fov_mm"]),
        "matrix_size": int(params["matrix_size"]),
        "kernel": params["kernel"],
        "contrast_phase": params["contrast_phase"],
    }


def _apply_slice_thickness(
    volume: np.ndarray,
    spacing: Spacing3D,
    slice_thickness_mm: float,
) -> tuple[np.ndarray, Dict[str, Any]]:
    z_spacing = max(float(spacing[0]), 1e-6)
    if slice_thickness_mm <= z_spacing + 1e-6:
        return volume, {
            "effective_slice_thickness_mm": float(max(slice_thickness_mm, z_spacing)),
            "z_sigma_voxels": 0.0,
        }

    sigma_voxels = max((slice_thickness_mm / z_spacing - 1.0) * 0.5, 0.0)
    blurred = gaussian_filter1d(volume, sigma=sigma_voxels, axis=0, mode="nearest")
    return blurred.astype(np.float32, copy=False), {
        "effective_slice_thickness_mm": float(slice_thickness_mm),
        "z_sigma_voxels": float(sigma_voxels),
    }


def _apply_dose_noise(
    volume: np.ndarray,
    resolved_params: Dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, Dict[str, Any]]:
    reference_mas = 150.0
    actual_mas = max(float(resolved_params["mAs"]), 1.0)
    sigma = 12.0 * np.sqrt(reference_mas / actual_mas)

    dose_level = resolved_params["dose_level"]
    level_scale = {"low": 1.2, "standard": 1.0, "high": 0.85}[dose_level]
    sigma *= level_scale

    noise = rng.normal(0.0, sigma, size=volume.shape).astype(np.float32)
    return (volume + noise).astype(np.float32, copy=False), {
        "noise_sigma_hu": float(sigma),
        "noise_reference_mAs": reference_mas,
        "dose_level_scale": float(level_scale),
    }


def _apply_kvp_transform(
    volume: np.ndarray,
    kVp: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    contrast_scale = {
        80: 1.18,
        100: 1.08,
        120: 1.00,
        140: 0.93,
    }[kVp]

    centered = volume.copy()
    high_hu_mask = centered > 150.0
    centered *= contrast_scale
    centered[high_hu_mask] *= 1.0 + (contrast_scale - 1.0) * 0.35

    return centered.astype(np.float32, copy=False), {
        "contrast_scale": float(contrast_scale),
        "high_hu_boost_applied": bool(kVp < 120),
    }


def _apply_pitch_effect(
    volume: np.ndarray,
    pitch: float,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if pitch <= 1.0:
        return volume, {"z_sigma_voxels": 0.0}

    sigma_voxels = (pitch - 1.0) * 0.8
    degraded = gaussian_filter1d(volume, sigma=sigma_voxels, axis=0, mode="nearest")
    return degraded.astype(np.float32, copy=False), {
        "z_sigma_voxels": float(sigma_voxels),
    }


def _resize_xy(volume: np.ndarray, scale: float, order: int = 1) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return volume.astype(np.float32, copy=False)
    return zoom(volume, (1.0, scale, scale), order=order, mode="nearest").astype(np.float32, copy=False)


def _crop_or_pad_to_shape(volume: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    _, ny, nx = volume.shape
    target_y, target_x = target_shape

    if ny > target_y:
        start_y = (ny - target_y) // 2
        volume = volume[:, start_y:start_y + target_y, :]
    elif ny < target_y:
        pad_before = (target_y - ny) // 2
        pad_after = target_y - ny - pad_before
        volume = np.pad(volume, ((0, 0), (pad_before, pad_after), (0, 0)), mode="constant", constant_values=-1000.0)

    if nx > target_x:
        start_x = (nx - target_x) // 2
        volume = volume[:, :, start_x:start_x + target_x]
    elif nx < target_x:
        pad_before = (target_x - nx) // 2
        pad_after = target_x - nx - pad_before
        volume = np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)), mode="constant", constant_values=-1000.0)

    return volume.astype(np.float32, copy=False)


def _apply_fov_effect(
    volume: np.ndarray,
    spacing: Spacing3D,
    fov_mm: float,
    warnings: list[str],
) -> tuple[np.ndarray, Spacing3D, Dict[str, Any]]:
    ny, nx = volume.shape[1], volume.shape[2]
    current_fov_y = float(spacing[1] * ny)
    current_fov_x = float(spacing[2] * nx)
    reference_fov = max(current_fov_y, current_fov_x)

    if abs(fov_mm - reference_fov) < 1.0:
        return volume, spacing, {
            "mode": "identity",
            "reference_fov_mm": float(reference_fov),
        }

    zoom_factor = reference_fov / max(fov_mm, 1.0)
    if zoom_factor > 1.0:
        scaled = _resize_xy(volume, zoom_factor, order=1)
        result = _crop_or_pad_to_shape(scaled, (ny, nx))
        min_body_fraction = min(1.0 / zoom_factor, 1.0)
        if min_body_fraction < 0.45:
            warnings.append(
                f"Requested fov_mm={fov_mm} is small relative to current FOV ~{reference_fov:.1f} mm; anatomy may be cropped."
            )
        mode = "crop_and_resize"
    else:
        scaled = _resize_xy(volume, zoom_factor, order=1)
        result = _crop_or_pad_to_shape(scaled, (ny, nx))
        mode = "pad_and_resize"

    new_spacing = (
        float(spacing[0]),
        float(fov_mm / ny),
        float(fov_mm / nx),
    )
    return result.astype(np.float32, copy=False), new_spacing, {
        "mode": mode,
        "reference_fov_mm": float(reference_fov),
        "applied_zoom_factor": float(zoom_factor),
    }


def _apply_matrix_effect(
    volume: np.ndarray,
    matrix_size: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if matrix_size >= 512:
        return volume, {
            "downsample_factor": 1.0,
            "effective_matrix_size": int(matrix_size),
        }

    scale = matrix_size / 512.0
    down = _resize_xy(volume, scale, order=1)
    restored = zoom(
        down,
        (1.0, volume.shape[1] / max(down.shape[1], 1), volume.shape[2] / max(down.shape[2], 1)),
        order=1,
        mode="nearest",
    )
    restored = _crop_or_pad_to_shape(restored.astype(np.float32, copy=False), (volume.shape[1], volume.shape[2]))
    return restored, {
        "downsample_factor": float(scale),
        "effective_matrix_size": int(matrix_size),
    }


def _apply_kernel_effect(
    volume: np.ndarray,
    kernel: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if kernel in ("smooth", "soft"):
        sigma = 1.1 if kernel == "smooth" else 0.8
        filtered = gaussian_filter(volume, sigma=(0.0, sigma, sigma), mode="nearest")
        return filtered.astype(np.float32, copy=False), {"mode": "gaussian_blur", "sigma_xy": float(sigma)}

    if kernel == "standard":
        filtered = gaussian_filter(volume, sigma=(0.0, 0.45, 0.45), mode="nearest")
        return filtered.astype(np.float32, copy=False), {"mode": "mild_blur", "sigma_xy": 0.45}

    if kernel == "lung":
        blurred = gaussian_filter(volume, sigma=(0.0, 0.5, 0.5), mode="nearest")
        sharpened = volume + 0.55 * (volume - blurred)
        return _clipped_float32(sharpened), {"mode": "unsharp_mask", "amount": 0.55, "pre_blur_sigma_xy": 0.5}

    if kernel in ("bone", "sharp"):
        blurred = gaussian_filter(volume, sigma=(0.0, 0.7, 0.7), mode="nearest")
        sharpened = volume + 0.9 * (volume - blurred)
        if kernel == "sharp":
            sharpened = sharpened - 0.08 * laplace(blurred, mode="nearest")
        return _clipped_float32(sharpened), {
            "mode": "strong_unsharp_mask",
            "amount": 0.9,
            "pre_blur_sigma_xy": 0.7,
            "laplacian_boost": bool(kernel == "sharp"),
        }

    return volume, {"mode": "identity"}


def _enhance_mask_region(volume: np.ndarray, mask: np.ndarray, delta_hu: float) -> np.ndarray:
    if not np.any(mask):
        return volume
    result = volume.copy()
    result[mask] += delta_hu
    softened = gaussian_filter(mask.astype(np.float32), sigma=(0.8, 0.8, 0.8), mode="nearest") > 0.05
    result[softened] = 0.7 * result[softened] + 0.3 * gaussian_filter(result, sigma=0.6, mode="nearest")[softened]
    return result.astype(np.float32, copy=False)


def _apply_contrast_phase(
    volume: np.ndarray,
    contrast_phase: str,
    label_volume: Optional[np.ndarray],
) -> tuple[np.ndarray, Dict[str, Any]]:
    if contrast_phase == "noncontrast":
        return volume, {"mode": "identity", "used_label_volume": bool(label_volume is not None)}

    result = volume.copy()
    phase_deltas = {
        "arterial": {9: 18.0, 7: 28.0, 8: 28.0, 17: 22.0, "generic": 12.0},
        "venous": {9: 26.0, 7: 18.0, 8: 18.0, 17: 20.0, "generic": 10.0},
        "delayed": {9: 14.0, 7: 12.0, 8: 12.0, 17: 10.0, "generic": 6.0},
    }[contrast_phase]

    label_based = False
    if label_volume is not None and label_volume.shape == volume.shape:
        for label_id in (9, 7, 8, 17):
            label_mask = label_volume == label_id
            if np.any(label_mask):
                result = _enhance_mask_region(result, label_mask, phase_deltas[label_id])
                label_based = True

    if not label_based:
        soft_tissue_mask = (volume > 20.0) & (volume < 120.0)
        vascular_mask = (volume >= 120.0) & (volume < 300.0)
        result[soft_tissue_mask] += phase_deltas["generic"]
        result[vascular_mask] += phase_deltas["generic"] * 0.5

    return result.astype(np.float32, copy=False), {
        "mode": "empirical_hu_boost",
        "used_label_volume": bool(label_based),
        "phase": contrast_phase,
    }


def simulate_ct_scan_params(
    volume: np.ndarray,
    spacing: Spacing3D,
    params: Dict[str, Any],
    label_volume: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Simulate CT scan parameter effects on a HU volume.

    Returns a dictionary with simulated volume, metadata, params_json,
    and lightweight preview statistics.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D (z, y, x), got shape={volume.shape}")
    if label_volume is not None and label_volume.shape != volume.shape:
        raise ValueError(
            f"label_volume shape {label_volume.shape} does not match volume shape {volume.shape}"
        )

    working = volume.astype(np.float32, copy=True)
    input_spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    warnings: list[str] = []
    resolved_params = _resolve_params(params)
    algorithm_steps: list[Dict[str, Any]] = []

    rng_seed = int(
        resolved_params["mAs"] * 17
        + resolved_params["kVp"] * 13
        + int(resolved_params["slice_thickness_mm"] * 100)
        + int(resolved_params["pitch"] * 100)
    )
    rng = np.random.default_rng(rng_seed)

    hu_before = [float(np.min(working)), float(np.max(working))]
    preview_before = _center_slice_stats(working)

    working, step_meta = _apply_slice_thickness(working, input_spacing, resolved_params["slice_thickness_mm"])
    algorithm_steps.append({"name": "slice_thickness", **step_meta})

    working, step_meta = _apply_dose_noise(working, resolved_params, rng)
    algorithm_steps.append({"name": "dose_noise", **step_meta})

    working, step_meta = _apply_kvp_transform(working, resolved_params["kVp"])
    algorithm_steps.append({"name": "kvp_transform", **step_meta})

    working, step_meta = _apply_pitch_effect(working, resolved_params["pitch"])
    algorithm_steps.append({"name": "pitch_degradation", **step_meta})

    working, output_spacing, step_meta = _apply_fov_effect(working, input_spacing, resolved_params["fov_mm"], warnings)
    algorithm_steps.append({"name": "fov_adjustment", **step_meta})

    working, step_meta = _apply_matrix_effect(working, resolved_params["matrix_size"])
    algorithm_steps.append({"name": "matrix_resolution", **step_meta})

    working, step_meta = _apply_kernel_effect(working, resolved_params["kernel"])
    algorithm_steps.append({"name": "reconstruction_kernel", **step_meta})

    working, step_meta = _apply_contrast_phase(working, resolved_params["contrast_phase"], label_volume)
    algorithm_steps.append({"name": "contrast_phase", **step_meta})

    working = _clipped_float32(working)

    hu_after = [float(np.min(working)), float(np.max(working))]
    preview_after = _center_slice_stats(working)

    approximation_notes = [
        "Image-domain approximation only; not a physics-based CT forward model.",
        "Dose and mAs use Gaussian noise scaling rather than quantum noise reconstruction.",
        "kVp effect uses HU contrast remapping, not spectrum-dependent attenuation.",
        "Pitch, slice thickness, FOV, and matrix effects are approximated with resampling and smoothing.",
        "Contrast phase enhancement is empirical and may use coarse organ labels when available.",
    ]

    standardized_output_notes = [
        "axis_order = zyx",
        "dtype = float32",
        "spacing order = z,y,x",
        "volume data is stored in top-level simulated_volume_base64",
        "standardized_case is intended for downstream artifact/lesion modules",
    ]

    metadata = {
        "shape": [int(v) for v in working.shape],
        "spacing": [float(v) for v in output_spacing],
        "hu_range": hu_after,
        "effective_slice_thickness_mm": float(resolved_params["slice_thickness_mm"]),
        "algorithm_notes": approximation_notes,
        "warnings": warnings,
        "notes": standardized_output_notes,
    }

    params_json = {
        "requested_params": params,
        "resolved_params": resolved_params,
        "algorithm_steps": algorithm_steps,
        "approximation_notes": approximation_notes,
        "warnings": warnings,
        "notes": standardized_output_notes,
        "input_shape": [int(v) for v in volume.shape],
        "output_shape": [int(v) for v in working.shape],
        "input_spacing": [float(v) for v in input_spacing],
        "output_spacing": [float(v) for v in output_spacing],
        "hu_range_before": hu_before,
        "hu_range_after": hu_after,
    }

    return {
        "simulated_volume": working,
        "simulated_spacing": output_spacing,
        "params_json": params_json,
        "metadata": metadata,
        "preview_stats": {
            "original_center_slice_stats": preview_before,
            "simulated_center_slice_stats": preview_after,
        },
    }

