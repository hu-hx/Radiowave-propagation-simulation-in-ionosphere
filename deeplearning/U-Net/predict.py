import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.amp import autocast, GradScaler

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# PL归一化参数（与dataset.py保持一致）
PL_min, PL_max = 80, 240


def validate(model, dataloader, criterion, device, save_dir, epoch):
    """验证模型并保存对比图"""
    model.eval()
    total_loss = 0
    step_nums = len(dataloader)

    # 创建验证结果目录
    epoch_dir = os.path.join(save_dir, f"val_epoch{epoch if isinstance(epoch, str) else f'{epoch:03d}'}")
    os.makedirs(epoch_dir, exist_ok=True)
    saved_count = 0
    MAX_SAVE = 30

    with torch.no_grad():
        for step, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            with autocast(device.type):
                outputs = model(images)["out"]
                loss, _ = criterion(outputs, masks)

            total_loss += loss.item()
            print(f"\r验证中... [{step+1}/{step_nums}]", end="")

            # 保存对比图
            for i in range(images.size(0)):
                if saved_count >= MAX_SAVE:
                    break
                img_path = os.path.join(epoch_dir, f"val_{saved_count+1:02d}.png")
                _save_comparison(outputs[i], masks[i], img_path,
                                title=f"Epoch {epoch}")
                saved_count += 1

    print(f"\n  已保存 {saved_count} 张验证对比图 → {epoch_dir}")
    return total_loss / step_nums


def _save_comparison(pred, gt, path, title=""):
    """保存预测vs真实对比图（直接显示PL）"""
    # 反归一化：从[0,1]映射回dB
    pred_db = pred.squeeze().cpu().numpy() * (PL_max - PL_min) + PL_min
    gt_db = gt.squeeze().cpu().numpy() * (PL_max - PL_min) + PL_min

    # 计算误差
    error = np.abs(pred_db - gt_db)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14)

    # 预测结果
    im1 = axes[0].pcolormesh(pred_db, shading='auto', cmap='jet_r')
    axes[0].set_title("predict PL (dB)")
    im1.set_clim(80, 240)
    plt.colorbar(im1, ax=axes[0])

    # 真实标签
    im2 = axes[1].pcolormesh(gt_db, shading='auto', cmap='jet_r')
    axes[1].set_title("real PL (dB)")
    im2.set_clim(80, 240)
    plt.colorbar(im2, ax=axes[1])

    # 误差图
    im3 = axes[2].pcolormesh(error, shading='auto', cmap='hot')
    axes[2].set_title(f"error (MAE={error.mean():.2f} dB)")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
