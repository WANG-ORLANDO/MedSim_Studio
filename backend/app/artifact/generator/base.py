"""
伪影生成器基类

定义所有伪影生成器的统一接口：接收 CT 体积 -> 返回含伪影体积 + 伪影掩码。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class BaseArtifactGenerator(ABC):
    """伪影生成器抽象基类"""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        生成伪影。

        Args:
            volume:  输入 CT 体积 (z, y, x), float32, HU 值
            spacing: 体素间距 (z, y, x) mm
            params:  伪影参数字典

        Returns:
            (artifact_volume, artifact_mask, metadata) 三元组:
            - artifact_volume: 叠加伪影后的 CT 体积 (shape 同输入)
            - artifact_mask:   伪影影响区域掩码 (0/1, shape 同输入)
            - metadata:        生成参数记录
        """
        ...

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """返回该伪影类型的默认参数"""
        ...

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """验证参数合法性"""
        ...

    def get_artifact_type(self) -> str:
        """返回伪影类型标识"""
        return self.__class__.__name__.replace("Generator", "").lower()
