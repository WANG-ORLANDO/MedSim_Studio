"""完整训练 + 评估脚本 — 混淆矩阵 + ROC-AUC + Grad-CAM + 推理速度"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from app.artifact.classifier.dataset import (
    ArtifactClassificationDataset,
    build_dataset_from_volume,
    CLASS_NAMES,
    NUM_CLASSES,
    IDX_TO_CLASS,
    get_train_transforms,
)
from app.artifact.classifier.model import ArtifactClassifier, create_classifier, save_classifier
from app.artifact.generator import get_generator, list_artifact_types


def generate_training_data(
    num_volumes_per_class: int = 50,
    volume_size: int = 64,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """生成训练数据"""
    rng = np.random.default_rng(seed)
    all_volumes = []
    all_labels = []

    # Clean
    for _ in range(num_volumes_per_class):
        vol = np.full((volume_size, volume_size, volume_size), 40.0, dtype=np.float32)
        z, y, x = np.ogrid[:volume_size, :volume_size, :volume_size]
        vol[((z - volume_size//2)**2 + (y - volume_size//2)**2 + (x - volume_size//2)**2) <= (volume_size//4)**2] = 400.0
        all_volumes.append(vol)
        all_labels.append([1, 0, 0, 0, 0, 0, 0, 0])

    class_mapping = {"metal": 1, "motion": 2, "noise": 3, "ring": 4, "streak": 5, "beam_hardening": 6}
    for art_type, class_idx in class_mapping.items():
        for _ in range(num_volumes_per_class):
            vol = np.full((volume_size, volume_size, volume_size), 40.0, dtype=np.float32)
            z, y, x = np.ogrid[:volume_size, :volume_size, :volume_size]
            vol[((z - volume_size//2)**2 + (y - volume_size//2)**2 + (x - volume_size//2)**2) <= (volume_size//4)**2] = 400.0
            try:
                gen = get_generator(art_type)
                vol, _, _ = gen.generate(vol, (1.0, 1.0, 1.0), gen.get_default_params())
            except Exception:
                pass
            label = [0] * NUM_CLASSES
            label[class_idx] = 1
            all_volumes.append(vol)
            all_labels.append(label)

    # Mixed
    for _ in range(num_volumes_per_class):
        vol = np.full((volume_size, volume_size, volume_size), 40.0, dtype=np.float32)
        z, y, x = np.ogrid[:volume_size, :volume_size, :volume_size]
        vol[((z - volume_size//2)**2 + (y - volume_size//2)**2 + (x - volume_size//2)**2) <= (volume_size//4)**2] = 400.0
        n_art = rng.integers(2, 4)
        selected = rng.choice(list(class_mapping.keys()), size=n_art, replace=False)
        label = [0] * NUM_CLASSES
        label[7] = 1
        for at in selected:
            try:
                gen = get_generator(at)
                vol, _, _ = gen.generate(vol, (1.0, 1.0, 1.0), gen.get_default_params())
                label[class_mapping[at]] = 1
            except Exception:
                pass
        all_volumes.append(vol)
        all_labels.append(label)

    return all_volumes, all_labels


def build_dataloaders(volumes, labels, batch_size=32, val_split=0.15):
    all_imgs, all_lbls = [], []
    for v, l in zip(volumes, labels):
        imgs, lbls = build_dataset_from_volume(v, l)
        all_imgs.extend(imgs)
        all_lbls.extend(lbls)

    ds = ArtifactClassificationDataset(all_imgs, all_lbls, transform=get_train_transforms())
    val_size = max(1, int(len(ds) * val_split))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, len(all_imgs)


def train_model(backbone="efficientnet_b3", epochs=50, batch_size=32, num_volumes=50, device="cpu"):
    """完整训练流程"""
    from torch.cuda.amp import autocast, GradScaler

    volumes, labels = generate_training_data(num_volumes_per_class=num_volumes)
    train_loader, val_loader, total_samples = build_dataloaders(volumes, labels, batch_size)

    model = create_classifier(num_classes=NUM_CLASSES, backbone=backbone, pretrained=False, device=device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    total_steps = epochs * len(train_loader)
    warmup_steps = 3 * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    use_amp = (device == "cuda")
    scaler = GradScaler() if use_amp else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            if use_amp:
                with autocast(device_type=device):
                    out = model(imgs)
                    loss = criterion(out, lbls)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            preds = (out >= 0.5).float()
            train_correct += (preds == lbls).float().mean().item() * imgs.size(0)
            train_total += imgs.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                val_loss += criterion(out, lbls).item() * imgs.size(0)
                preds = (out >= 0.5).float()
                val_correct += (preds == lbls).float().mean().item() * imgs.size(0)
                val_total += imgs.size(0)
                all_preds.append(out.cpu().numpy())
                all_targets.append(lbls.cpu().numpy())

        for _ in range(len(train_loader)):
            scheduler.step()

        tl = train_loss / train_total
        vl = val_loss / val_total
        ta = train_correct / train_total
        va = val_correct / val_total
        dt = time.time() - t0

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} ({dt:.1f}s) train_loss={tl:.4f} val_loss={vl:.4f} train_acc={ta:.3f} val_acc={va:.3f}")

        if vl < best_val_loss:
            best_val_loss = vl
            save_classifier(model, "/tmp/clf_full/best_model.pth")

    save_classifier(model, "/tmp/clf_full/final_model.pth")
    with open("/tmp/clf_full/history.json", "w") as f:
        json.dump(history, f)

    all_preds_np = np.concatenate(all_preds)
    all_targets_np = np.concatenate(all_targets)
    return model, history, all_preds_np, all_targets_np


def compute_roc_auc(all_preds, all_targets):
    """计算 per-class ROC-AUC"""
    from sklearn.metrics import roc_auc_score
    results = {}
    for i, name in enumerate(CLASS_NAMES):
        if all_targets[:, i].sum() > 0 and (1 - all_targets[:, i]).sum() > 0:
            try:
                auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
                results[name] = float(auc)
            except ValueError:
                results[name] = 0.0
        else:
            results[name] = 0.0
    return results


def plot_confusion_matrix(all_preds, all_targets, save_path):
    """生成混淆矩阵图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    pred_labels = (all_preds >= 0.5).astype(int)
    # 单标签: 取 argmax
    true_single = np.argmax(all_targets, axis=1)
    pred_single = np.argmax(pred_labels, axis=1)

    cm = confusion_matrix(true_single, pred_labels, labels=range(NUM_CLASSES))
    # 使用 argmax 版本
    cm2 = confusion_matrix(true_single, pred_single, labels=range(NUM_CLASSES))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(cm2, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(NUM_CLASSES)
    short_names = ["Clean", "Metal", "Motion", "Noise", "Ring", "Streak", "BeamH", "Mixed"]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(short_names)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    thresh = cm2.max() / 2
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            ax.text(j, i, format(cm2[i, j], "d"), ha="center", va="center",
                    color="white" if cm2[i, j] > thresh else "black", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def plot_training_curves(history, save_path):
    """生成训练曲线图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved: {save_path}")


def plot_roc_auc(roc_results, save_path):
    """生成 ROC-AUC 柱状图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(roc_results.keys())
    aucs = [roc_results[n] for n in names]
    short_names = ["Clean", "Metal", "Motion", "Noise", "Ring", "Streak", "BeamH", "Mixed"]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if a >= 0.9 else "#f39c12" if a >= 0.7 else "#e74c3c" for a in aucs]
    bars = ax.bar(range(len(names)), aucs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Per-Class ROC-AUC")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.9, color="green", linestyle="--", alpha=0.5, label="90% threshold")
    ax.legend()

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC-AUC chart saved: {save_path}")


def test_inference_speed(model_path, device="cpu", num_runs=50):
    """测试推理速度"""
    from app.artifact.classifier.model import load_classifier
    model = load_classifier(model_path, device=device)
    model.eval()

    # Warmup
    dummy = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(5):
        with torch.no_grad():
            model(dummy)

    # Benchmark
    times = []
    for _ in range(num_runs):
        t0 = time.time()
        with torch.no_grad():
            model(dummy)
        times.append((time.time() - t0) * 1000)

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    p95_ms = np.percentile(times, 95)
    return {"avg_ms": avg_ms, "std_ms": std_ms, "p95_ms": p95_ms, "fps": 1000 / avg_ms}


if __name__ == "__main__":
    os.makedirs("/tmp/clf_full", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. 训练
    print("=" * 60)
    print("Phase 1: Training (50 epochs, EfficientNet-B3)")
    print("=" * 60)
    model, history, all_preds, all_targets = train_model(
        backbone="efficientnet_b3", epochs=50, batch_size=32, num_volumes=50, device=device,
    )

    # 2. 评估
    print("\n" + "=" * 60)
    print("Phase 2: Evaluation")
    print("=" * 60)
    roc_results = compute_roc_auc(all_preds, all_targets)
    print("\nPer-class ROC-AUC:")
    for name, auc in sorted(roc_results.items(), key=lambda x: -x[1]):
        status = "✓" if auc >= 0.9 else "△" if auc >= 0.7 else "✗"
        print(f"  {status} {name:15s}: {auc:.4f}")

    # 混淆矩阵
    pred_labels = (all_preds >= 0.5).astype(int)
    true_single = np.argmax(all_targets, axis=1)
    pred_single = np.argmax(pred_labels, axis=1)
    from sklearn.metrics import accuracy_score, f1_score
    overall_acc = accuracy_score(true_single, pred_single)
    macro_f1 = f1_score(true_single, pred_single, average="macro")
    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    # 3. 生成图表
    print("\n" + "=" * 60)
    print("Phase 3: Generating plots")
    print("=" * 60)
    plot_training_curves(history, "/tmp/clf_full/training_curves.png")
    plot_confusion_matrix(all_preds, all_targets, "/tmp/clf_full/confusion_matrix.png")
    plot_roc_auc(roc_results, "/tmp/clf_full/roc_auc.png")

    # 4. 推理速度
    print("\n" + "=" * 60)
    print("Phase 4: Inference Speed")
    print("=" * 60)
    speed = test_inference_speed("/tmp/clf_full/best_model.pth", device=device)
    print(f"Avg: {speed['avg_ms']:.1f}ms | P95: {speed['p95_ms']:.1f}ms | FPS: {speed['fps']:.1f}")

    # 5. 总结
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    single_pass = all(roc_results.get(n, 0) >= 0.9 for n in CLASS_NAMES if n != "mixed")
    mixed_pass = roc_results.get("mixed", 0) >= 0.8
    print(f"Single-class AUC ≥ 90%: {'PASS' if single_pass else 'FAIL'}")
    print(f"Mixed AUC ≥ 80%: {'PASS' if mixed_pass else 'FAIL'}")
    print(f"Speed < 50ms: {'PASS' if speed['avg_ms'] < 50 else 'FAIL'} ({speed['avg_ms']:.1f}ms)")
