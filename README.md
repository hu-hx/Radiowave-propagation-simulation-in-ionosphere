# Radiowave Propagation Simulation
2D Raidowave propagation simulation based on the parabolic equation with Split-step Fourier transform(SSFT) and Intelligent propagation model

基于抛物方程法（PE）和分步傅里叶变换（SSFT）的二维电波传播仿真，结合深度学习智能预测模型。

## 项目简介

本项目实现了电波在对流层和电离层中的二维传播仿真，涵盖传统数值方法和基于深度学习的智能传播模型。主要应用场景包括大气波导效应分析、电离层电波传播预测以及射线追踪与抛物方程法的对比研究。

## 项目结构

```
Radiowave-propagation-simulation/
├── PE_SSFT/                    # MATLAB 抛物方程核心算法
│   ├── PE_SSFT_2D.m            # 电离层电波传播仿真（窄角/宽角抛物方程）
│   ├── PE_SSFT_2D_atmoduct.m   # 大气波导传播仿真
│   ├── SSHFT.m                 # 快速正弦变换
│   └── SSHFT_mex.cpp           # MEX 加速的正弦变换实现
│
├── atmospheric_duct/           # Python 大气波导仿真及 GUI
│   ├── PE_SSFT_2D_atmoduct.py  # 波导传播核心算法（Feit-Fleck 宽角近似）
│   ├── atmo_duct_GUI.py        # PyQt6 图形界面，支持参数配置与结果可视化
│   └── png_to_ico.py           # 图标转换工具
│
├── deeplearning/               # 深度学习智能传播模型
│   ├── make_dataset/           # 数据集生成（MATLAB 脚本）
│   │   ├── make_dataset_Ne.m   # 电离层电子密度数据集
│   │   └── make_dataset_PL.m   # 传播损耗数据集
│   └── U-Net/                  # U-Net 深度学习模型
│       ├── model.py            # ResNet残差块 + CBAM注意力 + FiLM频率调制
│       ├── train.py            # 训练脚本（混合精度 + 早停 + 梯度裁剪）
│       ├── predict.py          # 预测与验证
│       ├── dataset.py          # 数据集加载
│       └── losses1.py          # 组合损失函数（GradientWeightedL1 + MS-SSIM + FrequencyLoss）
│
└── ray/                        # 射线追踪
    ├── ray_test1.m             # 基于 PHARLAP 的电离层射线追踪示例
    ├── PE_SSFT_2D_ray.m        # PE 与射线追踪的对比验证
    └── pharlap_4.5.1.zip       # PHARLAP 射线追踪工具箱
```

## 核心算法

### 抛物方程法（PE + SSFT）

- **窄角近似**：Taylor 近似抛物方程，适用于小角度传播
- **宽角近似**：Feit-Fleck 近似，适用于大角度场景
- **步进求解**：采用快速正弦变换（DST）进行频域步进，支持 MEX 加速
- **边界条件**：上边界采用平滑吸收层，下边界支持 PEC 和阻抗边界

### 大气波导仿真

支持多种波导类型的折射率剖面建模：
- 标准大气
- 蒸发波导
- 表面波导
- 抬升波导

### 深度学习模型

基于改进的 U-Net 架构：
- ResNet 残差块 + CBAM（通道-空间注意力机制）
- FiLM 层实现频率参数的特征调制
- 多尺度特征融合，保留高频细节
- 组合损失函数：梯度加权 L1 + 多尺度 SSIM + 频域损失

## 依赖环境

### MATLAB 部分
- MATLAB R2020a 或更高版本
- Signal Processing Toolbox（用于 DST 等变换）

### Python 部分
- Python 3.8+
- NumPy, SciPy, Matplotlib
- PyQt6（大气波导 GUI）
- PyTorch 2.0+（深度学习模块）

## 使用说明

### MATLAB 仿真
```matlab
% 电离层电波传播
cd PE_SSFT
PE_SSFT_2D

% 大气波导传播
PE_SSFT_2D_atmoduct
```

### Python 大气波导仿真
```bash
cd atmospheric_duct
python PE_SSFT_2D_atmoduct.py

# 启动图形界面
python atmo_duct_GUI.py
```

### 深度学习模型
```bash
# 1. 使用 MATLAB 脚本生成数据集
cd deeplearning/make_dataset
# 运行 make_dataset_Ne.m 和 make_dataset_PL.m

# 2. 训练模型
cd ../U-Net
python train.py

# 3. 预测
python predict.py
```

## 作者

huhaixiang

## License

本项目仅供学术研究使用。
