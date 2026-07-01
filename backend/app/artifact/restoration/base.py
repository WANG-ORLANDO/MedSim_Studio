"""伪影修复基类 — 所有修复器共享接口"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class BaseRestorer:
    """伪影修复器基类

    所有修复器必须实现:
    - restore(volume, spacing, params) → (restored_volume, metadata)
    """

    def restore(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, ...],
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    def get_default_params(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_restorer_type() -> str:
        return "base"
