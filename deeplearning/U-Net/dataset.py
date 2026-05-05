"""
数据集加载模块
支持电子密度(Ne)和传播损失(PL)数据的加载和预处理
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class dat_transform(object):
    """
    将.dat文件转换为归一化的Tensor
    改进点：
      - Ne使用对数缩放，保留低电子密度细节
      - 添加坐标通道，帮助网络感知空间位置
      - 频率作为独立通道
      - PL使用线性缩放到[0,1]
    """
    def __init__(self, dtype=np.float32, size=(512, 512)):
        self.dtype = dtype
        self.size = size

        # 对数缩放参数
        self.log_Ne_min = 3.0  # log10(1e3)
        self.log_Ne_max = 6.0  # log10(1e6)
        self.max_freq = 31.0

        # PL数据归一化参数（值域80-240 dB）
        self.PL_min = 80.0
        self.PL_max = 240.0

    def __call__(self, path, flag, freq=None,theta0=None,z0=None,beta=None):
        img_array = np.loadtxt(path, dtype=self.dtype)
        tensor = torch.from_numpy(img_array)

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # (1, H, W)

        tensor = tensor.unsqueeze(0)  # (1, 1, H, W)
        tensor = F.interpolate(
            tensor, size=self.size,
            mode='bilinear', align_corners=False
        )
        tensor = tensor.squeeze(0)  # (1, H, W)

        if flag == "PL":
            # PL数据线性归一化到[0,1]
            tensor = (tensor - self.PL_min) / (self.PL_max - self.PL_min)
            tensor = tensor.clamp(0.0, 1.0)
            return tensor

        elif flag == "Ne":
            # 1. 对数缩放Ne
            tensor = tensor.clamp(min=1.0)
            tensor_log = torch.log10(tensor)
            tensor_ne = (tensor_log - self.log_Ne_min) / (self.log_Ne_max - self.log_Ne_min)
            tensor_ne = tensor_ne.clamp(0.0, 1.0)

            # 2. 频率通道
            tensor_freq = torch.full_like(tensor_ne, freq / self.max_freq)

            # 3. 天线增益通道
            z_max = 400 # km
            x_max = 3000 # km
            theta0, beta = np.deg2rad(theta0), np.deg2rad(beta)
            z0 = z0 / 1e3
            _, h, w = tensor_ne.shape
            # 使用torch而非numpy，保持数据类型一致
            x = torch.linspace(x_max / w, x_max, w, dtype=torch.float32)
            z = torch.linspace(0.1, z_max, h, dtype=torch.float32)
            Z, X = torch.meshgrid(z, x, indexing='ij')  # 使用torch.meshgrid
            Angle_Map = torch.atan2(Z - z0, X.clamp(min=1.0))

            theta0_tensor = torch.tensor(theta0, dtype=torch.float32)
            beta_tensor = torch.tensor(beta, dtype=torch.float32)
            Gain_Map = torch.exp(-torch.log(torch.tensor(2.0)) * ((Angle_Map - theta0_tensor) / (beta_tensor / 2))** 2)
            Gain_Map = Gain_Map.unsqueeze(0)  # (1, H, W)


            # 4. 坐标通道
            y_coords = torch.linspace(0, 1, h).view(1, h, 1).expand(1, h, w)
            x_coords = torch.linspace(0, 1, w).view(1, 1, w).expand(1, h, w)

            # 拼接：[Ne_log, Freq,Gain_Map, Y, X] → 5通道
            tensor = torch.cat([tensor_ne, tensor_freq,Gain_Map, y_coords, x_coords], dim=0)

            return tensor

        else:
            raise ValueError(f"数据标签不符合规范：{flag}")



class MyDataset(Dataset):
    """电离层传播数据集"""
    def __init__(self, path, transform=dat_transform()):
        super(MyDataset, self).__init__()
        self.path = path
        self.path_Ne = os.path.join(path, "Ne")
        self.path_PL = os.path.join(path, "PL")

        self.images_Ne = sorted([f for f in os.listdir(self.path_Ne) if f.endswith('.dat')])
        self.images_PL = sorted([f for f in os.listdir(self.path_PL) if f.endswith('.dat')])

        self.transform = transform

        assert len(self.images_PL) > 0, f"PL目录为空: {self.path_PL}"
        print(f"数据集加载完成: Ne={len(self.images_Ne)}个文件, PL={len(self.images_PL)}个样本")

    def __getitem__(self, idx):
        pl_filename = self.images_PL[idx]

        # 文件名格式：{freq}_{theta0}_{z0}_{beta}_{date}.dat
        parts = pl_filename.split("_")
        freq = int(parts[0])
        theta0 = int(parts[1])
        z0 = int(parts[2])
        beta = int(parts[3])
        date = parts[-1]

        image_PL_path = os.path.join(self.path_PL, pl_filename)
        image_PL = self.transform(image_PL_path, "PL")

        image_Ne_path = os.path.join(self.path_Ne, date)
        image_input = self.transform(image_Ne_path, "Ne", freq,theta0,z0,beta)

        return image_input, image_PL

    def __len__(self):
        return len(self.images_PL)


if __name__ == "__main__":


    dataset = MyDataset("../data")
    idx = 34
    image_input, image_PL = dataset[idx]

    print(f"输入形状: {image_input.shape}")
    print(f"  Ne_log通道 值域: [{image_input[0].min():.3f}, {image_input[0].max():.3f}]")
    print(f"  Freq通道   值域: [{image_input[1].min():.3f}, {image_input[1].max():.3f}]")
    print(f"  天线增益通道   值域: [{image_input[2].min():.3f}, {image_input[2].max():.3f}]")
    print(f"  Y_coord通道 值域: [{image_input[3].min():.3f}, {image_input[3].max():.3f}]")
    print(f"  X_coord通道 值域: [{image_input[4].min():.3f}, {image_input[3].max():.3f}]")
    print(f"标签形状: {image_PL.shape}, 值域: [{image_PL.min():.3f}, {image_PL.max():.3f}]")

    fig, axes = plt.subplots(2, 2, figsize=(18, 4))
    axes[0,0].pcolormesh(image_input[0].numpy(), shading='auto', cmap='jet')
    axes[0,0].set_title("Ne（对数归一化）")
    plt.colorbar(axes[0,0].collections[0], ax=axes[0,0])

    axes[0,1].pcolormesh(image_input[1].numpy(), shading='auto', cmap='gray')
    axes[0,1].set_title("频率通道")
    plt.colorbar(axes[0,1].collections[0], ax=axes[0,1])

    axes[1,0].pcolormesh(image_input[2].numpy(), shading='auto', cmap='gray')
    axes[1,0].set_title("天线增益通道")
    plt.colorbar(axes[1,0].collections[0], ax=axes[1,0])

    # PL需要反归一化显示
    PL_min, PL_max = 80, 240
    axes[1,1].pcolormesh(image_PL[0].numpy() * (PL_max - PL_min) + PL_min, shading='auto', cmap='jet_r')
    axes[1,1].set_title("PL (dB)")
    axes[1,1].collections[0].set_clim(80, 240)
    plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])

    plt.tight_layout()

    plt.savefig('../../'+str(idx)+'.png', dpi=300, bbox_inches='tight')
    plt.show()

