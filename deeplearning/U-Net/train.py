"""
改进的训练脚本
主要特点：
  1. 使用改进的Unet模型（ResNet + CBAM + FiLM）
  2. 优化的组合损失函数（GradientWeightedL1 + MS-SSIM + FrequencyLoss）
  3. AdamW优化器 + ReduceLROnPlateau学习率策略
  4. torch.compile 加速
  5. 梯度裁剪和混合精度训练
  6. 详细的训练日志和可视化
  7. 早停机制
"""

import os
import sys
import time
import torch
from datetime import datetime
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

from dataset import MyDataset
from predict import validate
from model import PathLossUnetModel
from losses1 import CombinedLoss


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch, epochs):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    loss_details = {"gwl1": 0, "ssim": 0, "freq": 0}
    step_nums = len(dataloader)

    for step, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # 混合精度训练
        with autocast(device.type):
            outputs = model(images)["out"]
            loss, details = criterion(outputs, masks)

        # 反向传播
        scaler.scale(loss).backward()
        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k in loss_details:
            loss_details[k] += details[k]

        # 每30个batch打印一次进度，以及最后一个step
        if step % 30 == 0 or step == step_nums - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\rEpoch [{epoch}/{epochs}] Step [{step+1}/{step_nums}] "
                  f"Loss: {loss.item():.4f} "
                  f"(GWL1:{details['gwl1']:.4f} SSIM:{details['ssim']:.4f} "
                  f"Freq:{details['freq']:.4f}) "
                  f"LR: {current_lr:.2e}",
                  end="")

    n = step_nums
    return total_loss / n, {k: v / n for k, v in loss_details.items()}


def main():
    # ==================== 配置参数 ====================
    data_path = "../data"
    batch_size = 8
    epochs = 60
    lr_init = 2e-5          # 与当前续训一致，从上次断点的LR继续
    weight_decay = 1e-4
    num_workers = 12
    resume = True
    early_stop_patience = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    save_dir = "./save"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "train.pth")

    # ==================== 数据加载 ====================
    print("\n加载数据集...")
    full_dataset = MyDataset(data_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(88)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )
    print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

    # ==================== 模型构建 ====================
    print("\n构建模型...")
    model = PathLossUnetModel(
        in_channels=5,
        out_channels=1,
        base_channels=64,
        freq_embed_dim=32
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # ==================== 优化器和调度器 ====================
    optimizer = AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)

    # ReduceLROnPlateau：验证loss无改善则LR乘以0.7
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=0,
        min_lr=1e-7
    )

    # 混合精度训练
    scaler = GradScaler()

    # 损失函数
    criterion = CombinedLoss(
        w_gwl1=0.5,      # GradientWeightedL1，替代原L1，边缘/焦散区域加权惩罚
        w_ssim=0.25,      # MS-SSIM，维护整体结构感知
        w_freq=0.2,      # FrequencyLoss，恢复干涉条纹等高频细节
        edge_alpha=5.0   # 边缘权重放大倍数，可在3~8之间调节
    ).to(device)

    # ==================== 断点续训 ====================
    best_loss = float('inf')
    epoch_start = 0
    no_improve = 0

    if resume and os.path.exists(model_path):
        print(f"\n加载检查点: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        # 自动兼容：若checkpoint由compile后的模型保存，key带_orig_mod.前缀，自动剥离
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict):
            print("  检测到 _orig_mod. 前缀，自动剥离...")
            state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']
        best_loss = checkpoint.get('val_loss', float('inf'))
        print(f"已训练 {epoch_start} 轮, 最佳验证损失={best_loss:.4f}")
        for param_group in optimizer.param_groups:
           param_group['lr'] = 2e-6  # 改成你想要的lr


    # torch.compile 加速：必须在 load_state_dict 之后调用，否则key前缀不匹配
    print("正在编译模型 (torch.compile)...")
    model = torch.compile(model)

    # ==================== 训练循环 ====================
    print(f"\n开始训练... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    for epoch in range(epoch_start + 1, epochs + 1):
        t0 = time.time()

        train_loss, train_details = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch, epochs
        )

        # 验证
        val_loss = validate(model, val_loader, criterion, device, save_dir, epoch)

        # ReduceLROnPlateau：每个epoch验证后调用，基于val_loss决定是否降LR
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch:3d} | "
              f"Train: {train_loss:.4f} "
              f"(GWL1:{train_details['gwl1']:.4f} SSIM:{train_details['ssim']:.4f} "
              f"Freq:{train_details['freq']:.4f}) | "
              f"Val: {val_loss:.4f} | LR: {current_lr:.2e} | {elapsed:.1f}s")

        # 保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, model_path)
            print(f"  ✓ 已保存最佳模型 (val_loss={best_loss:.4f})\n")
        else:
            no_improve += 1
            print(f"  · 无改善 {no_improve}/{early_stop_patience}\n")

        # 早停
        if no_improve >= early_stop_patience:
            print(f"早停触发于第 {epoch} 轮")
            break

    print(f"\n训练完成！最佳验证损失: {best_loss:.4f}")

    # ==================== 最终验证 ====================
    print("\n加载最佳模型进行最终验证...")
    best_checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    final_val_loss = validate(model, val_loader, criterion, device, save_dir, "best")
    print(f"最终验证损失: {final_val_loss:.4f}")


if __name__ == "__main__":
    main()
