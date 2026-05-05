"""
生成验证集对比图脚本
基于 train.py 和 predict.py 编写，专门用于生成120张验证集对比图
不修改任何现有文件
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 与 dataset.py 保持一致的归一化参数
PL_min, PL_max = 80, 240

# 导入本地模块
from dataset import MyDataset
from model import PathLossUnetModel


def save_comparison(pred, gt, path, title=""):
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
    axes[0].set_title("预测传播损失 (dB)")
    im1.set_clim(80, 240)
    plt.colorbar(im1, ax=axes[0])

    # 真实标签
    im2 = axes[1].pcolormesh(gt_db, shading='auto', cmap='jet_r')
    axes[1].set_title("真实传播损失 (dB)")
    im2.set_clim(80, 240)
    plt.colorbar(im2, ax=axes[1])

    # 误差图
    im3 = axes[2].pcolormesh(error, shading='auto', cmap='hot')
    axes[2].set_title(f"误差图 (MAE={error.mean():.2f} dB)")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def load_model_and_data(device, data_path="../data", batch_size=8):
    """加载模型和验证数据集"""
    # 加载完整数据集
    print("加载数据集...")
    full_dataset = MyDataset(data_path)

    # 使用与训练相同的随机种子划分验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(88)  # 与 train.py 相同的种子
    )

    # 验证集 DataLoader（不 shuffle）
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=12,
        pin_memory=True, persistent_workers=True
    )
    print(f"验证集大小: {len(val_dataset)} 个样本")

    # 构建模型
    print("构建模型...")
    model = PathLossUnetModel(
        in_channels=5,
        out_channels=1,
        base_channels=64,
        freq_embed_dim=32
    ).to(device)

    # 加载训练好的权重
    model_path = "./train.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重文件不存在: {model_path}")

    print(f"加载模型权重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # 处理 torch.compile 前缀
    if any(k.startswith('_orig_mod.') for k in state_dict):
        print("  检测到 _orig_mod. 前缀，自动剥离...")
        state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    return model, val_loader


def generate_validation_plots(num_images=120, output_dir="./val_plots_120"):
    """生成指定数量的验证集对比图"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 加载模型和验证集
    model, val_loader = load_model_and_data(device)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    saved_count = 0
    max_save = num_images

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            if saved_count >= max_save:
                break

            images = images.to(device)
            masks = masks.to(device)

            with autocast(device.type):
                outputs = model(images)["out"]

            # 保存当前批次中的图片
            for i in range(images.size(0)):
                if saved_count >= max_save:
                    break

                img_path = os.path.join(output_dir, f"val_{saved_count+1:04d}.png")
                save_comparison(
                    outputs[i], masks[i], img_path,
                    title=f"验证样本 {saved_count+1}"
                )
                saved_count += 1

            print(f"\r已生成 {saved_count}/{max_save} 张图片...", end="")

    print(f"\n完成！共生成 {saved_count} 张验证对比图，保存在 {output_dir}")


if __name__ == "__main__":
    generate_validation_plots(num_images=120, output_dir="./val_plots_120")