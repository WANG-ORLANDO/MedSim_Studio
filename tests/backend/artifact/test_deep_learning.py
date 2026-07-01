"""单元测试：RED-CNN 深度学习修复模型"""

import numpy as np
import torch
import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend"))

from app.artifact.restoration.deep_learning import REDCNN, REDCNNRestorer, REDCNNTrainer


def _make_test_slice(size=64):
    """生成简单测试切片"""
    img = np.full((size, size), 40.0, dtype=np.float32)
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    circle = ((y - cy) ** 2 + (x - cx) ** 2) <= (size // 4) ** 2
    img[circle] = 400.0
    return img


class TestREDCNNArchitecture:
    """模型架构测试"""

    def test_output_shape(self):
        model = REDCNN(in_channels=1, out_channels=1, num_features=96, num_layers=5)
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_different_sizes(self):
        model = REDCNN(num_features=32, num_layers=3)
        for size in [32, 48, 64, 128]:
            x = torch.randn(1, 1, size, size)
            out = model(x)
            assert out.shape == (1, 1, size, size)

    def test_residual_connection(self):
        """残差连接: 对于零输入，输出应接近零"""
        model = REDCNN(num_features=32, num_layers=3)
        model.eval()
        x = torch.zeros(1, 1, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.abs().max() < 0.2, f"Zero input should give near-zero output, got {out.abs().max():.4f}"

    def test_parameter_count(self):
        model = REDCNN(num_features=96, num_layers=5)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be reasonable for a small model
        assert 500_000 < n_params < 10_000_000, f"Unexpected param count: {n_params}"


class TestREDCNNInference:
    """推理测试"""

    def test_restore_slice(self):
        restorer = REDCNNRestorer(device="cpu")
        slice_2d = _make_test_slice(64)
        result = restorer.restore_slice(slice_2d)
        assert result.shape == slice_2d.shape
        assert result.dtype == np.float32

    def test_restore_volume(self):
        restorer = REDCNNRestorer(device="cpu")
        volume = np.stack([_make_test_slice(64) for _ in range(8)])
        result = restorer.restore_volume(volume)
        assert result.shape == volume.shape
        assert result.dtype == np.float32

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pth")
            # Save with matching params
            model = REDCNN(num_features=96, num_layers=5)
            torch.save({"model_state_dict": model.state_dict()}, model_path)
            # Load
            restorer = REDCNNRestorer(model_path=model_path, device="cpu")
            slice_2d = _make_test_slice(32)
            result = restorer.restore_slice(slice_2d)
            assert result.shape == slice_2d.shape

    def test_no_model_file(self):
        """无模型文件时应使用随机权重"""
        restorer = REDCNNRestorer(model_path="/nonexistent/path.pth", device="cpu")
        slice_2d = _make_test_slice(64)
        result = restorer.restore_slice(slice_2d)
        assert result.shape == slice_2d.shape


class TestREDCNNTrainer:
    """训练器测试"""

    def test_short_training(self):
        """短训练测试 (2 epochs, 小数据)"""
        clean_vols = [np.full((4, 32, 32), 40.0, dtype=np.float32)]
        artifact_vols = [clean_vols[0] + np.random.default_rng(42).normal(0, 20, (4, 32, 32)).astype(np.float32)]

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = REDCNNTrainer(model=REDCNN(num_features=16, num_layers=2), device="cpu")
            history = trainer.train(
                clean_volumes=clean_vols,
                artifact_volumes=artifact_vols,
                epochs=2,
                batch_size=4,
                lr=1e-3,
                output_dir=tmpdir,
            )
            assert len(history["train_loss"]) == 2
            assert len(history["val_loss"]) == 2
            assert os.path.exists(os.path.join(tmpdir, "best_model.pth"))
            assert os.path.exists(os.path.join(tmpdir, "final_model.pth"))

    def test_training_reduces_loss(self):
        """训练应使 loss 下降"""
        clean_vols = [np.full((8, 32, 32), 40.0, dtype=np.float32)]
        rng = np.random.default_rng(42)
        artifact_vols = [clean_vols[0] + rng.normal(0, 30, (8, 32, 32)).astype(np.float32)]

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = REDCNNTrainer(model=REDCNN(num_features=16, num_layers=2), device="cpu")
            history = trainer.train(
                clean_volumes=clean_vols,
                artifact_volumes=artifact_vols,
                epochs=10,
                batch_size=8,
                lr=1e-3,
                output_dir=tmpdir,
            )
            # Loss should decrease overall
            assert history["val_loss"][-1] <= history["val_loss"][0] * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
