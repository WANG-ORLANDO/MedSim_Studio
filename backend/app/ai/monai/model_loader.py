"""
MONAI Model Registry & Manager

Lazily loads and caches MONAI segmentation models.
Gracefully handles missing PyTorch/MONAI dependencies — the app
starts without them, and inference raises a clear error.

Three architectures:
  - UNet:     3D U-Net for multi-organ segmentation
  - SegResNet: 3D SegResNet for lesion segmentation
  - Swin UNETR: Transformer-based whole-body segmentation
"""

import logging
from typing import Dict, Optional, List
from app.core.config import settings

logger = logging.getLogger(__name__)

# Label indices matched to /api/v1/segmentation/labels
ORGAN_LABEL_MAP: Dict[str, int] = {
    "background": 0,
    "liver": 1,
    "kidney": 2,
    "lung": 3,
    "spleen": 4,
    "pancreas": 5,
    "bladder": 6,
    "bone": 7,
    "lesion_tumor": 8,
    "lesion_metastasis": 9,
}

NUM_CLASSES = len(ORGAN_LABEL_MAP)  # 10


class ModelNotAvailableError(RuntimeError):
    """Raised when a model cannot be loaded (dependencies missing or weights not found)."""
    pass


class SegmentationModelManager:
    """
    Singleton-style manager for MONAI segmentation models.

    Models are lazily constructed and cached on first request.
    If torch or MONAI are not installed, clear error messages guide the user.
    """

    _instance: Optional["SegmentationModelManager"] = None
    _models: Dict[str, object] = {}
    _device: str = "cpu"

    def __new__(cls) -> "SegmentationModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._device = settings.AI_DEVICE if hasattr(settings, "AI_DEVICE") else "cpu"

            # Attempt to detect CUDA
            if cls._device == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("AI_DEVICE=cuda but CUDA not available; falling back to cpu")
                        cls._device = "cpu"
                except ImportError:
                    logger.warning("AI_DEVICE=cuda but torch not installed; falling back to cpu")
                    cls._device = "cpu"

            logger.info("SegmentationModelManager initialized (device=%s)", cls._device)

        return cls._instance

    @property
    def device(self) -> str:
        return self._device

    def load_model(self, name: str) -> object:
        """
        Load (or retrieve from cache) a segmentation model by name.

        Args:
            name: One of "unet", "segresnet", "swin_unetr"

        Returns:
            PyTorch nn.Module in eval mode

        Raises:
            ModelNotAvailableError: If dependencies missing or model unknown
        """
        if name in self._models:
            return self._models[name]

        # Lazy import — torch/monai may not be installed
        try:
            import torch
        except ImportError:
            raise ModelNotAvailableError(
                "PyTorch is not installed. Run: pip install torch==2.1.2"
            )

        try:
            import monai
        except ImportError:
            raise ModelNotAvailableError(
                "MONAI is not installed. Run: pip install monai==1.3.0"
            )

        from monai.networks.nets import UNet, SegResNet, SwinUNETR

        name_lower = name.lower()

        logger.info("[MODEL] Building %s model architecture...", name)
        import time as _time
        _t0 = _time.time()

        if name_lower == "unet":
            model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=NUM_CLASSES,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
            )
        elif name_lower == "segresnet":
            model = SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=NUM_CLASSES,
                init_filters=8,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                dropout_prob=0.2,
            )
        elif name_lower == "swin_unetr":
            model = SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=1,
                out_channels=NUM_CLASSES,
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=False,
            )
        else:
            raise ModelNotAvailableError(
                f"Unknown model '{name}'. Available: unet, segresnet, swin_unetr"
            )

        # Try to load pretrained weights
        weights_path = self._resolve_weights_path(name_lower)
        if weights_path:
            try:
                import torch
                state_dict = torch.load(weights_path, map_location=self._device)
                model.load_state_dict(state_dict)
                logger.info("Loaded weights for model '%s' from %s", name, weights_path)
            except Exception as e:
                logger.warning(
                    "Failed to load weights for model '%s' from %s: %s. "
                    "Running with random weights — segmentation results will be random.",
                    name, weights_path, e,
                )
        else:
            logger.warning(
                "No pretrained weights found for model '%s'. "
                "Place .pth files in %s/ to enable real segmentation. "
                "Results will be random without weights.",
                name, settings.AI_MODEL_PATH,
            )

        logger.info("[MODEL] Built model in %.1fs, moving to %s...", _time.time() - _t0, self._device)
        _t1 = _time.time()

        model.to(self._device)
        logger.info("[MODEL] model.to(%s) took %.1fs", self._device, _time.time() - _t1)

        model.eval()
        self._models[name_lower] = model

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            "[MODEL] '%s' ready on %s: %s parameters (%.1f MB)",
            name, self._device, param_count,
            param_count * 4 / (1024 * 1024),
        )

        return model

    def _resolve_weights_path(self, name: str) -> Optional[str]:
        """Check AI_MODEL_PATH for a matching .pth file."""
        import os
        model_dir = settings.AI_MODEL_PATH
        if not model_dir or not os.path.isdir(model_dir):
            return None

        candidates = [
            os.path.join(model_dir, f"{name}.pth"),
            os.path.join(model_dir, f"{name}.pt"),
            os.path.join(model_dir, f"{name}_weights.pth"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def unload_all(self) -> None:
        """Unload all cached models and free GPU memory."""
        try:
            import torch
            for name, model in self._models.items():
                if hasattr(model, "cpu"):
                    model.cpu()
                del model
            torch.cuda.empty_cache()
        except ImportError:
            pass
        self._models.clear()
        logger.info("All segmentation models unloaded")

    def is_model_loaded(self, name: str) -> bool:
        """Check if a model is already loaded in cache."""
        return name.lower() in self._models

    def get_loaded_models(self) -> List[str]:
        """Return list of currently loaded model names."""
        return list(self._models.keys())
