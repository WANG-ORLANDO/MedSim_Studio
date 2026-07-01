"""伪影修复质量评估 — PSNR / SSIM / NMSE / MAE(HU)"""

import numpy as np
from typing import Dict, Any, Optional, List


def compute_psnr(original: np.ndarray, restored: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB)

    Args:
        original: 参考图像 (H,W) 或 (Z,H,W)
        restored: 修复图像
    Returns:
        PSNR 值 (dB)，越大越好
    """
    mse = np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2)
    if mse == 0:
        return 999.99  # cap at high value instead of inf (JSON-safe)
    return float(10.0 * np.log10(np.max(np.abs(original)) ** 2 / mse))


def compute_ssim(original: np.ndarray, restored: np.ndarray) -> float:
    """Structural Similarity Index (逐切片平均)

    Args:
        original: 参考图像
        restored: 修复图像
    Returns:
        SSIM 值 (0~1)，越大越好
    """
    from skimage.metrics import structural_similarity as ssim_func

    if original.ndim == 2:
        data_range = original.max() - original.min()
        if data_range == 0:
            return 1.0
        return float(ssim_func(original, restored, data_range=data_range))

    scores = []
    for z in range(original.shape[0]):
        data_range = original[z].max() - original[z].min()
        if data_range == 0:
            scores.append(1.0)
            continue
        try:
            s = ssim_func(original[z], restored[z], data_range=data_range)
            scores.append(s)
        except Exception:
            pass
    return float(np.mean(scores)) if scores else 0.0


def compute_nmse(original: np.ndarray, restored: np.ndarray) -> float:
    """Normalized Mean Square Error

    Args:
        original: 参考图像
        restored: 修复图像
    Returns:
        NMSE 值，越小越好 (0 表示完全一致)
    """
    orig = original.astype(np.float64)
    rest = restored.astype(np.float64)
    norm = np.sum(orig ** 2)
    if norm == 0:
        return float('inf')
    return float(np.sum((orig - rest) ** 2) / norm)


def compute_mae_hu(original: np.ndarray, restored: np.ndarray) -> float:
    """Mean Absolute Error in HU (亨氏单位)

    Args:
        original: 参考 HU 图像
        restored: 修复 HU 图像
    Returns:
        MAE (HU)，越小越好
    """
    return float(np.mean(np.abs(original.astype(np.float64) - restored.astype(np.float64))))


def compute_all_metrics(
    original: np.ndarray,
    restored: np.ndarray,
    artifact: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """计算全部质量指标

    Args:
        original: 干净参考图像 (2D 或 3D)
        restored: 修复后图像
        artifact: 含伪影图像 (可选，用于计算相对改善)
    Returns:
        {psnr, ssim, nmse, mae_hu} 指标字典
    """
    metrics = {
        "psnr": compute_psnr(original, restored),
        "ssim": compute_ssim(original, restored),
        "nmse": compute_nmse(original, restored),
        "mae_hu": compute_mae_hu(original, restored),
    }

    if artifact is not None:
        artifact_metrics = {
            "psnr_before": compute_psnr(original, artifact),
            "ssim_before": compute_ssim(original, artifact),
        }
        metrics["psnr_improvement"] = metrics["psnr"] - artifact_metrics["psnr_before"]
        metrics["ssim_improvement"] = metrics["ssim"] - artifact_metrics["ssim_before"]
        metrics.update(artifact_metrics)

    return metrics


def format_quality_report(metrics: Dict[str, float]) -> str:
    """格式化质量报告为可读字符串"""
    lines = [
        "=" * 50,
        "  Artifact Restoration Quality Report",
        "=" * 50,
    ]

    if "psnr_before" in metrics:
        lines.append(f"  PSNR (before):     {metrics['psnr_before']:.2f} dB")
        lines.append(f"  PSNR (after):      {metrics['psnr']:.2f} dB")
        lines.append(f"  PSNR improvement:  +{metrics.get('psnr_improvement', 0):.2f} dB")
        lines.append("")
        lines.append(f"  SSIM (before):     {metrics['ssim_before']:.4f}")
        lines.append(f"  SSIM (after):      {metrics['ssim']:.4f}")
        lines.append(f"  SSIM improvement:  +{metrics.get('ssim_improvement', 0):.4f}")
    else:
        lines.append(f"  PSNR:   {metrics['psnr']:.2f} dB")
        lines.append(f"  SSIM:   {metrics['ssim']:.4f}")

    lines.append("")
    lines.append(f"  NMSE:   {metrics['nmse']:.6f}")
    lines.append(f"  MAE:    {metrics['mae_hu']:.2f} HU")
    lines.append("=" * 50)

    return "\n".join(lines)
