"""单元测试：验证数据集类和分类模型的正确性"""

import numpy as np
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from app.artifact.classifier.dataset import (
    ArtifactClassificationDataset,
    build_dataset_from_volume,
    slice_to_windowed,
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_TO_IDX,
)
from app.artifact.classifier.model import (
    ArtifactClassifier,
    create_classifier,
)


class TestDataset:
    """数据集测试"""

    def test_slice_to_windowed(self):
        hu = np.array([[-1000, 0, 40, 200, 3000]], dtype=np.float32)
        result = slice_to_windowed(hu, window_level=40, window_width=400)
        assert result.dtype == np.uint8
        assert result.shape == hu.shape
        # -1000 → 0 (低于窗口下限)
        assert result[0, 0] == 0
        # 40 → ~127 (窗位中心)
        assert 100 < result[0, 2] < 160
        # 3000 → 255 (高于窗口上限)
        assert result[0, 4] == 255

    def test_dataset_creation(self):
        images = [np.random.randint(0, 256, (512, 512), dtype=np.uint8) for _ in range(10)]
        labels = [[1, 0, 0, 0, 0, 0, 0, 0] for _ in range(10)]
        dataset = ArtifactClassificationDataset(images, labels)
        assert len(dataset) == 10
        img_tensor, label_tensor = dataset[0]
        assert img_tensor.shape == (3, 512, 512)
        assert label_tensor.shape == (8,)
        assert img_tensor.dtype == torch.float32

    def test_dataset_with_transform(self):
        images = [np.random.randint(0, 256, (256, 256), dtype=np.uint8) for _ in range(5)]
        labels = [[0, 1, 0, 0, 0, 0, 0, 0] for _ in range(5)]

        try:
            import albumentations as A
            transform = A.Compose([A.Rotate(limit=10, p=1.0)])
            dataset = ArtifactClassificationDataset(images, labels, transform=transform)
            img_tensor, label_tensor = dataset[0]
            assert img_tensor.shape == (3, 256, 256)
        except ImportError:
            pass

    def test_build_dataset_from_volume(self):
        volume = np.random.randn(30, 512, 512).astype(np.float32) * 500
        label = [0, 0, 1, 0, 0, 0, 0, 0]
        images, labels = build_dataset_from_volume(volume, label, slice_indices=[10, 15, 20])
        assert len(images) == 3
        assert len(labels) == 3
        assert all(l == label for l in labels)
        assert images[0].shape == (512, 512)
        assert images[0].dtype == np.uint8

    def test_build_dataset_auto_indices(self):
        volume = np.random.randn(60, 256, 256).astype(np.float32) * 200
        label = [1, 0, 0, 0, 0, 0, 0, 0]
        images, labels = build_dataset_from_volume(volume, label)
        # 默认取中间 1/3: 20..39 = 20 张
        assert len(images) == 20

    def test_class_constants(self):
        assert NUM_CLASSES == 8
        assert len(CLASS_NAMES) == 8
        assert CLASS_NAMES[0] == "clean"
        assert CLASS_NAMES[7] == "mixed"
        assert CLASS_TO_IDX["metal"] == 1
        assert CLASS_TO_IDX["ring"] == 4


class TestModel:
    """模型测试"""

    def test_create_classifier(self):
        model = create_classifier(num_classes=8, pretrained=False, device="cpu")
        assert isinstance(model, ArtifactClassifier)
        model.eval()

    def test_forward_pass(self):
        model = create_classifier(num_classes=8, pretrained=False, device="cpu")
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 8)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_predict(self):
        model = create_classifier(num_classes=8, pretrained=False, device="cpu")
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            preds = model.predict(x, threshold=0.5)
        assert preds.shape == (4, 8)
        assert set(preds.unique().tolist()).issubset({0.0, 1.0})

    def test_save_load(self):
        import tempfile
        model = create_classifier(num_classes=8, pretrained=False, device="cpu")
        model.eval()

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            from app.artifact.classifier.model import save_classifier, load_classifier
            save_classifier(model, path)
            loaded = load_classifier(path, device="cpu")
            loaded.eval()

            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out1 = model(x)
                out2 = loaded(x)
            assert torch.allclose(out1, out2, atol=1e-5)
        finally:
            os.unlink(path)

    def test_different_backbones(self):
        for backbone in ["efficientnet_b0", "efficientnet_b3"]:
            model = create_classifier(
                num_classes=8, backbone=backbone, pretrained=False, device="cpu"
            )
            model.eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
