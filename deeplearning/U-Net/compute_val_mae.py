"""
计算验证集平均绝对误差（MAE）的脚本
参考train.py和predict.py，计算所有验证集样本的MAE（单位：dB）
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入项目模块
from dataset import MyDataset
from model import PathLossUnetModel
from losses1 import CombinedLoss

# PL归一化参数（与dataset.py保持一致）
PL_min, PL_max = 80, 240

def load_model_and_data(model_path, data_path, batch_size=8, num_workers=12):
    """加载模型和验证集"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载完整数据集
    print(f"\n加载数据集: {data_path}")
    full_dataset = MyDataset(data_path)

    # 按照train.py相同的随机种子划分验证集（80/20）
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(88)  # 固定随机种子88
    )
    print(f"总样本数: {len(full_dataset)}, 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # 验证集DataLoader
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )

    # 构建模型
    print("\n构建模型...")
    model = PathLossUnetModel(
        in_channels=5,
        out_channels=1,
        base_channels=64,
        freq_embed_dim=32
    ).to(device)

    # 加载检查点
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行train.py训练模型并保存最佳模型")
        sys.exit(1)

    print(f"加载模型检查点: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # 处理可能的_orig_mod前缀（torch.compile兼容）
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict):
        print("  检测到 _orig_mod. 前缀，自动剥离...")
        state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    print("模型加载完成")

    # 损失函数（用于验证损失计算，但本脚本主要计算MAE）
    criterion = CombinedLoss(
        w_gwl1=0.5,
        w_ssim=0.25,
        w_freq=0.2,
        edge_alpha=5.0
    ).to(device)

    return device, model, val_loader, criterion

def compute_mae(model, val_loader, device):
    """计算验证集平均绝对误差（MAE，单位：dB）"""
    model.eval()
    total_mae = 0.0
    total_samples = 0

    # 存储每个样本的误差，用于进一步分析
    all_errors = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)["out"]

            # 反归一化到dB单位
            pred_db = outputs.cpu().numpy() * (PL_max - PL_min) + PL_min
            gt_db = masks.cpu().numpy() * (PL_max - PL_min) + PL_min

            # 计算绝对误差（形状: [batch_size, 1, height, width]）
            error = np.abs(pred_db - gt_db)

            # 计算每个样本的平均误差（在空间维度上平均）
            batch_mae = error.mean(axis=(1, 2, 3))  # 形状: [batch_size]

            # 累积
            total_mae += batch_mae.sum()
            total_samples += len(batch_mae)
            all_errors.extend(batch_mae)

            # 进度显示
            if (batch_idx + 1) % 10 == 0:
                print(f"\r处理批次 [{batch_idx+1}/{len(val_loader)}], 当前批次平均MAE: {batch_mae.mean():.3f} dB", end="")

    print()  # 换行

    if total_samples == 0:
        return 0.0, all_errors

    overall_mae = total_mae / total_samples
    return overall_mae, all_errors

def main():
    # ==================== 配置参数 ====================
    model_path = "./train.pth"      # 最佳模型路径
    data_path = "../data"                # 数据目录
    batch_size = 8                       # 与train.py保持一致
    num_workers = 12                     # 与train.py保持一致

    print("=" * 60)
    print("验证集平均绝对误差（MAE）计算")
    print("=" * 60)

    # 加载模型和验证集
    device, model, val_loader, criterion = load_model_and_data(
        model_path, data_path, batch_size, num_workers
    )

    # 计算MAE
    print("\n计算验证集平均绝对误差...")
    overall_mae, all_errors = compute_mae(model, val_loader, device)

    # 输出结果
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    print(f"验证集样本总数: {len(all_errors)}")
    print(f"平均绝对误差 (MAE): {overall_mae:.4f} dB")
    print(f"MAE 范围: [{min(all_errors):.4f}, {max(all_errors):.4f}] dB")
    print(f"MAE 标准差: {np.std(all_errors):.4f} dB")

    # 将MAE结果保存到文件
    os.makedirs("./eval_results", exist_ok=True)
    results_file = "./eval_results/val_mae_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("验证集平均绝对误差（MAE）计算结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"验证集样本总数: {len(all_errors)}\n")
        f.write(f"平均绝对误差 (MAE): {overall_mae:.4f} dB\n")
        f.write(f"MAE 范围: [{min(all_errors):.4f}, {max(all_errors):.4f}] dB\n")
        f.write(f"MAE 标准差: {np.std(all_errors):.4f} dB\n")
        f.write("\n详细误差列表（每个样本的MAE，单位dB）：\n")
        for i, err in enumerate(all_errors):
            f.write(f"样本 {i+1}: {err:.4f} dB\n")

    print(f"\nMAE计算结果已保存到文件: {results_file}")

    # 可选：保存误差分布图
    save_plot = True
    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('绝对误差 (dB)')
        plt.ylabel('样本数量')
        plt.title(f'验证集绝对误差分布 (整体MAE = {overall_mae:.4f} dB)')
        plt.grid(True, alpha=0.3)

        # 保存图像
        os.makedirs("./eval_results", exist_ok=True)
        plot_path = "./eval_results/val_mae_distribution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n误差分布图已保存: {plot_path}")

    print("\n计算完成！")

if __name__ == "__main__":
    main()