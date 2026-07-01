"""伪影分类训练脚本 — BCE + Cosine LR + AMP + Early Stopping"""

import os
import sys
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from app.artifact.classifier.dataset import (
    ArtifactClassificationDataset,
    CLASS_NAMES,
    NUM_CLASSES,
    get_train_transforms,
)
from app.artifact.classifier.model import ArtifactClassifier, create_classifier, save_classifier
from app.artifact.generator import get_generator, list_artifact_types

logger = logging.getLogger(__name__)

TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 5,
    "early_stopping_patience": 10,
    "mixed_precision": True,
    "val_split": 0.15,
    "num_workers": 2,
    "image_size": 224,
}


def generate_training_data(
    num_volumes_per_class: int = 20,
    volume_size: int = 64,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """使用伪影生成器批量生成训练数据

    Returns:
        (volumes, label_vectors) 元组
    """
    rng = np.random.default_rng(seed)
    artifact_types = list_artifact_types()
    all_volumes = []
    all_labels = []

    # === 类别 0: Clean（无伪影）===
    for _ in range(num_volumes_per_class):
        vol = np.full((volume_size, volume_size, volume_size), 40.0, dtype=np.float32)
        cz, cy, cx = volume_size // 2, volume_size // 2, volume_size // 2
        radius = volume_size // 4
        z, y, x = np.ogrid[:volume_size, :volume_size, :volume_size]
        sphere = ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2
        vol[sphere] = 400.0
        # 添加骨骼结构
        vol[vol > 300] = 800.0
        all_volumes.append(vol)
        all_labels.append([1, 0, 0, 0, 0, 0, 0, 0])

    # === 类别 1-6: 单一伪影 ===
    class_mapping = {
        "metal": 1, "motion": 2, "noise": 3,
        "ring": 4, "streak": 5, "beam_hardening": 6,
    }
    for art_type, class_idx in class_mapping.items():
        for _ in range(num_volumes_per_class):
            base_vol = np.full((volume_size, volume_size, volume_size), 40.0, dtype=np.float32)
            cz, cy, cx = volume_size // 2, volume_size // 2, volume_size // 2
            radius = volume_size // 4
            z, y, x = np.ogrid[:volume_size, :volume_size, :volume_size]
            sphere = ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2
            base_vol[sphere] = 400.0

            try:
                gen = get_generator(art_type)
                params = gen.get_default_params()
                vol_out, _, _ = gen.generate(base_vol, (1.0, 1.0, 1.0), params)
                all_volumes.append(vol_out)
                label = [0] * NUM_CLASSES
                label[class_idx] = 1
                all_labels.append(label)
            except Exception as e:
                logger.warning(f"Failed to generate {art_type}: {e}")

    # === 类别 7: Mixed（多伪影组合）===
    for _ in range(num_volumes_per_class):
        base_vol = np.full((volume_size, volume_size, volume_size), 40.0, dtype=np.float32)
        cz, cy, cx = volume_size // 2, volume_size // 2, volume_size // 2
        radius = volume_size // 4
        z, y, x = np.ogrid[:volume_size, :volume_size, :volume_size]
        sphere = ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2
        base_vol[sphere] = 400.0

        # 随机选择 2-3 种伪影叠加
        n_artifacts = rng.integers(2, 4)
        selected = rng.choice(list(class_mapping.keys()), size=n_artifacts, replace=False)
        current = base_vol.copy()
        label = [0] * NUM_CLASSES
        label[7] = 1
        for art_type in selected:
            try:
                gen = get_generator(art_type)
                params = gen.get_default_params()
                current, _, _ = gen.generate(current, (1.0, 1.0, 1.0), params)
            except Exception:
                pass
        all_volumes.append(current)
        all_labels.append(label)

    logger.info(f"Generated {len(all_volumes)} volumes ({num_volumes_per_class} per class)")
    return all_volumes, all_labels


def build_dataloaders(
    volumes: List[np.ndarray],
    label_vectors: List[List[int]],
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    """构建训练/验证 DataLoader"""
    all_images = []
    all_labels = []
    for vol, labels in zip(volumes, label_vectors):
        from app.artifact.classifier.dataset import build_dataset_from_volume
        imgs, lbls = build_dataset_from_volume(vol, labels)
        all_images.extend(imgs)
        all_labels.extend(lbls)

    logger.info(f"Total samples: {len(all_images)}")

    dataset = ArtifactClassificationDataset(
        all_images, all_labels, transform=get_train_transforms(),
    )

    val_size = max(1, int(len(dataset) * config["val_split"]))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    return train_loader, val_loader


def get_cosine_schedule_with_warmup(
    optimizer, warmup_steps: int, total_steps: int,
):
    """Cosine 学习率调度 + Warmup"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """计算分类指标"""
    preds = (outputs >= threshold).float()
    correct = (preds == targets).float()
    accuracy = correct.mean().item()

    # Per-class metrics
    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        if targets[:, i].sum() > 0:
            per_class_acc[name] = (preds[:, i] == targets[:, i]).float().mean().item()

    # F1
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy,
        "macro_f1": f1.mean().item(),
        "per_class_acc": per_class_acc,
    }


def train_one_epoch(
    model: ArtifactClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    device: str,
    use_amp: bool,
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(device_type=device):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item() * images.size(0)
        all_outputs.append(outputs.detach().cpu())
        all_targets.append(labels.cpu())

    avg_loss = total_loss / len(loader.dataset)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_outputs, all_targets)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def validate(
    model: ArtifactClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """验证"""
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        all_outputs.append(outputs.cpu())
        all_targets.append(labels.cpu())

    avg_loss = total_loss / len(loader.dataset)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_outputs, all_targets)
    metrics["loss"] = avg_loss
    return metrics


def train(
    config: Optional[Dict[str, Any]] = None,
    output_dir: str = "/app/models/artifact_classifier",
    num_volumes_per_class: int = 20,
    pretrained: bool = True,
) -> Dict[str, Any]:
    """完整训练流程

    Args:
        config: 训练配置，None 使用默认
        output_dir: 模型保存目录
        num_volumes_per_class: 每类生成的体积数

    Returns:
        训练历史
    """
    if config is None:
        config = TRAIN_CONFIG.copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device}")
    logger.info(f"Config: {config}")

    os.makedirs(output_dir, exist_ok=True)

    # 生成训练数据
    logger.info("Generating training data...")
    volumes, label_vectors = generate_training_data(
        num_volumes_per_class=num_volumes_per_class,
        volume_size=64,
    )

    # 构建 DataLoader
    train_loader, val_loader = build_dataloaders(volumes, label_vectors, config)
    logger.info(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # 创建模型
    model = create_classifier(num_classes=NUM_CLASSES, pretrained=pretrained, device=device)

    # 损失函数 — BCEWithLogitsLoss（模型内部已有 Sigmoid，所以用 BCELoss）
    criterion = nn.BCELoss()

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # 学习率调度
    total_steps = config["epochs"] * len(train_loader)
    warmup_steps = config["warmup_epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # AMP
    use_amp = config["mixed_precision"] and device == "cuda"
    scaler = GradScaler() if use_amp else None

    # 训练循环
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, use_amp,
        )
        val_metrics = validate(model, val_loader, criterion, device)
        dt = time.time() - t0

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        logger.info(
            f"Epoch {epoch}/{config['epochs']} ({dt:.1f}s) — "
            f"train_loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f} f1={train_metrics['macro_f1']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.3f} f1={val_metrics['macro_f1']:.3f}"
        )

        # Early stopping + 保存最佳模型
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            save_classifier(model, os.path.join(output_dir, "best_model.pth"))
            logger.info(f"  → Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # 保存最终模型
    save_classifier(model, os.path.join(output_dir, "final_model.pth"))

    # 保存训练历史
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Models saved to {output_dir}")

    return history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-volumes", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="/app/models/artifact_classifier")
    parser.add_argument("--no-pretrained", action="store_true", help="Skip pretrained weights download")
    args = parser.parse_args()

    config = TRAIN_CONFIG.copy()
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr

    train(config=config, output_dir=args.output_dir, num_volumes_per_class=args.num_volumes, pretrained=not args.no_pretrained)
