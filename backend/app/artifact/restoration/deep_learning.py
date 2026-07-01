"""RED-CNN 深度学习修复模型 — Residual Encoder-Decoder CNN"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple


class REDCNN(nn.Module):
    """Residual Encoder-Decoder CNN for CT artifact removal

    Architecture (from B组开发流程.md §7.3.2):
        Input: artifact CT slice (1×H×W)
        ├─ Conv + ReLU (num_features channels)
        ├─ Encoder: 5× (Conv + ReLU)
        ├─ Decoder: 5× (Conv + ReLU) + skip connections
        ├─ Conv (1 channel)
        └─ Residual: output = input - residual
        Output: restored CT slice (1×H×W)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 num_features: int = 96, num_layers: int = 5):
        super().__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_layers)
        ])

        # Decoder
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_layers)
        ])

        self.conv_last = nn.Conv2d(num_features, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_first(x)

        # Encoder + save skip connections
        skip_connections = []
        for encoder in self.encoders:
            out = encoder(out)
            skip_connections.append(out)

        # Decoder + skip connections
        for i, decoder in enumerate(self.decoders):
            out = decoder(out)
            out = out + skip_connections[-(i + 1)]

        out = self.conv_last(out)
        return residual - out  # Residual learning


class REDCNNRestorer:
    """RED-CNN 推理接口"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = REDCNN().to(device)

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()

    @torch.no_grad()
    def restore_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """修复单张 CT 切片

        Args:
            slice_2d: (H, W) float32 HU 值切片

        Returns:
            (H, W) float32 修复后切片
        """
        h, w = slice_2d.shape
        tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).float().to(self.device)
        output = self.model(tensor)
        return output.squeeze(0).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def restore_volume(self, volume: np.ndarray) -> np.ndarray:
        """修复整个 CT 体积

        Args:
            volume: (z, y, x) float32 HU 值体积

        Returns:
            (z, y, x) float32 修复后体积
        """
        restored = np.zeros_like(volume)
        for z in range(volume.shape[0]):
            restored[z] = self.restore_slice(volume[z])
        return restored


class REDCNNTrainer:
    """RED-CNN 训练器"""

    def __init__(self, model: Optional[REDCNN] = None, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = (model or REDCNN()).to(device)

    def train(
        self,
        clean_volumes: list,
        artifact_volumes: list,
        epochs: int = 50,
        batch_size: int = 16,
        lr: float = 1e-3,
        val_split: float = 0.15,
        output_dir: str = "/app/models/artifact_restorer",
    ) -> Dict[str, Any]:
        """训练 RED-CNN

        Args:
            clean_volumes: 干净体积列表 [(z,y,x), ...]
            artifact_volumes: 含伪影体积列表 (与 clean 一一配对)
            epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
            val_split: 验证集比例
            output_dir: 模型保存目录

        Returns:
            训练历史 dict
        """
        os.makedirs(output_dir, exist_ok=True)

        # 提取切片对
        clean_slices, artifact_slices = [], []
        for clean, art in zip(clean_volumes, artifact_volumes):
            for z in range(clean.shape[0]):
                clean_slices.append(clean[z])
                artifact_slices.append(art[z])

        n = len(clean_slices)
        n_val = max(1, int(n * val_split))
        n_train = n - n_val

        # 构建 Tensor
        train_clean = torch.from_numpy(np.stack(clean_slices[:n_train])).unsqueeze(1).float().to(self.device)
        train_art = torch.from_numpy(np.stack(artifact_slices[:n_train])).unsqueeze(1).float().to(self.device)
        val_clean = torch.from_numpy(np.stack(clean_slices[n_train:])).unsqueeze(1).float().to(self.device)
        val_art = torch.from_numpy(np.stack(artifact_slices[n_train:])).unsqueeze(1).float().to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            # Mini-batch training
            indices = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n_train, batch_size):
                idx = indices[i:i + batch_size]
                art_batch = train_art[idx]
                clean_batch = train_clean[idx]

                optimizer.zero_grad()
                output = self.model(art_batch)
                loss = criterion(output, clean_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(val_art)
                val_loss = criterion(val_output, val_clean).item()
                val_psnr = self._compute_psnr(val_clean, val_output)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_psnr"].append(val_psnr)

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs} train_loss={avg_train_loss:.6f} val_loss={val_loss:.6f} val_psnr={val_psnr:.2f}dB")

            # Save best model
            if epoch == 1 or val_loss < min(history["val_loss"][:-1], default=float('inf')):
                self._save(os.path.join(output_dir, "best_model.pth"), epoch, val_loss)

        self._save(os.path.join(output_dir, "final_model.pth"), epochs, history["val_loss"][-1])
        print(f"Training complete. Models saved to {output_dir}")
        return history

    def _compute_psnr(self, clean: torch.Tensor, restored: torch.Tensor) -> float:
        mse = nn.functional.mse_loss(restored, clean).item()
        if mse == 0:
            return 999.99
        return 10.0 * np.log10(1.0 / mse)

    def _save(self, path: str, epoch: int, val_loss: float):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }, path)
