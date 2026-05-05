"""
Unet模型
架构特点：
  1. ResNet残差块 + CBAM注意力机制
  2. FiLM层实现频率参数的特征调制
  3. 多尺度特征融合
  4. 深层特征提取保留高频细节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== CBAM 注意力模块 =====================
class ChannelAttention(nn.Module):
    """通道注意力：关注哪些特征通道更重要"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力：关注图像中哪些位置更重要"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    """CBAM：通道注意力 + 空间注意力"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)  # 通道加权
        x = x * self.sa(x)  # 空间加权
        return x


# ===================== FiLM 层 =====================
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation
    根据频率参数动态调制特征图：out = gamma * x + beta
    """
    def __init__(self, freq_dim, feature_channels):
        super(FiLMLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(freq_dim, feature_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels * 2, feature_channels * 2)
        )

    def forward(self, x, freq_embedding):
        # freq_embedding: (B, freq_dim)
        # x: (B, C, H, W)
        params = self.fc(freq_embedding)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=1)  # 各 (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


# ===================== 残差块 =====================
class ResidualBlock(nn.Module):
    """带CBAM注意力的残差块"""
    def __init__(self, channels, use_cbam=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels) if use_cbam else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += residual
        return self.relu(out)


# ===================== 编码器块 =====================
class EncoderBlock(nn.Module):
    """编码器：下采样 + 残差块 + FiLM调制"""
    def __init__(self, in_channels, out_channels, freq_dim, num_res_blocks=2):
        super(EncoderBlock, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(out_channels) for _ in range(num_res_blocks)
        ])
        self.film = FiLMLayer(freq_dim, out_channels)

    def forward(self, x, freq_emb):
        x = self.down(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.film(x, freq_emb)  # 频率调制
        return x


# ===================== 解码器块 =====================
class DecoderBlock(nn.Module):
    """解码器：上采样 + 跳跃连接 + 残差块 + FiLM调制"""
    def __init__(self, in_channels, skip_channels, out_channels, freq_dim, num_res_blocks=2):  # 修复1: freq_dim后加逗号
        super(DecoderBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 跳跃连接后的通道数
        self.conv_merge = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(out_channels) for _ in range(num_res_blocks)
        ])
        self.film = FiLMLayer(freq_dim, out_channels)

    def forward(self, x, skip, freq_emb):  # 修复2: 中文逗号→英文逗号
        x = self.up(x)
        # 处理尺寸不匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_merge(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.film(x, freq_emb)
        return x


# ===================== 主模型 =====================
class PathLossUnetModel(nn.Module):
    """
    输入：5通道 (Ne_log, Freq, Gain_Map, Y_coord, X_coord)
    输出：1通道 (归一化的PL预测)
    """
    def __init__(self, in_channels=5, out_channels=1, base_channels=64, freq_embed_dim=32):
        super(PathLossUnetModel, self).__init__()

        # 频率嵌入层：将归一化频率映射到高维空间
        self.freq_embedding = nn.Sequential(
            nn.Linear(1, freq_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(freq_embed_dim, freq_embed_dim)
        )

        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # 编码器路径（4层下采样）
        self.enc1 = EncoderBlock(base_channels,      base_channels * 2,  freq_embed_dim)  # 256
        self.enc2 = EncoderBlock(base_channels * 2,  base_channels * 4,  freq_embed_dim)  # 128
        self.enc3 = EncoderBlock(base_channels * 4,  base_channels * 8,  freq_embed_dim)  # 64
        self.enc4 = EncoderBlock(base_channels * 8,  base_channels * 16, freq_embed_dim)  # 32

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 16),
            ResidualBlock(base_channels * 16),
            CBAM(base_channels * 16)
        )

        # 解码器路径
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8,  base_channels * 8,  freq_embed_dim)
        self.dec3 = DecoderBlock(base_channels * 8,  base_channels * 4,  base_channels * 4,  freq_embed_dim)
        self.dec2 = DecoderBlock(base_channels * 4,  base_channels * 2,  base_channels * 2,  freq_embed_dim)
        self.dec1 = DecoderBlock(base_channels * 2,  base_channels,       base_channels,      freq_embed_dim)

        # 最终输出
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, out_channels, 1),
            nn.Sigmoid()  # 输出归一化到[0,1]
        )

    def forward(self, x):
        """
        x: (B, 5, H, W) - [Ne_log, Freq, Gain_pattern, Y_coord, X_coord]
        返回: {"out": (B, 1, H, W)}
        """
        x = x.float()
        # 提取频率通道并生成嵌入
        freq_channel = x[:, 1:2, 0, 0]  # (B, 1) - 频率值在整个图像中是常数
        freq_emb = self.freq_embedding(freq_channel)  # (B, freq_embed_dim)

        # 初始特征提取
        x0 = self.init_conv(x)  # (B, 64, 512, 512)

        # 编码器
        x1 = self.enc1(x0, freq_emb)  # (B, 128, 256, 256)
        x2 = self.enc2(x1, freq_emb)  # (B, 256, 128, 128)
        x3 = self.enc3(x2, freq_emb)  # (B, 512, 64, 64)
        x4 = self.enc4(x3, freq_emb)  # (B, 1024, 32, 32)

        # 瓶颈
        x_bottle = self.bottleneck(x4)  # (B, 1024, 32, 32)

        # 解码器（带跳跃连接，修复4: 中文逗号→英文逗号）
        x = self.dec4(x_bottle, x3, freq_emb)  # (B, 512, 64, 64)
        x = self.dec3(x,        x2, freq_emb)  # (B, 256, 128, 128)
        x = self.dec2(x,        x1, freq_emb)  # (B, 128, 256, 256)
        x = self.dec1(x,        x0, freq_emb)  # (B, 64, 512, 512)

        # 最终输出
        out = self.final_conv(x)  # (B, 1, 512, 512)

        return {"out": out}


if __name__ == "__main__":
    # 测试模型
    model = PathLossUnetModel(in_channels=5, out_channels=1, base_channels=64)
    x = torch.randn(2, 5, 512, 512)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out['out'].shape}")
    print(f"输出值域: [{out['out'].min():.3f}, {out['out'].max():.3f}]")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
