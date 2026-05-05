"""
优化的组合损失函数
包含：GradientWeightedL1 + MS-SSIM + 频率损失
针对电离层传播图像的边缘误差和高频细节优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSSSIMLoss(nn.Module):
    """
    多尺度结构相似性损失
    保持图像的整体结构和纹理特征
    """
    def __init__(self, scales=4, weights=None):
        super(MSSSIMLoss, self).__init__()
        self.weights = weights or [0.1, 0.2, 0.3, 0.4]
        self.scales = scales

    def _ssim_at_scale(self, pred, target, window_size=11):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

    def forward(self, pred, target):
        loss = 0.0
        p, t = pred, target
        for i, w in enumerate(self.weights):
            loss += w * self._ssim_at_scale(p, t)
            if i < self.scales - 1:
                p = F.avg_pool2d(p, 2)
                t = F.avg_pool2d(t, 2)
        return loss


class GradientWeightedL1Loss(nn.Module):
    """
    梯度加权 L1 损失：以 target 的梯度图作为空间权重
    边缘/焦散区域权重更高，对这些区域施加更强的惩罚
    alpha: 边缘区域相对于平坦区域的最大权重倍数
    """
    def __init__(self, alpha=5.0):
        super(GradientWeightedL1Loss, self).__init__()
        self.alpha = alpha
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _edge_weight(self, target):
        N, C, H, W = target.shape
        t = target.view(N * C, 1, H, W)
        gx = F.conv2d(t, self.sobel_x, padding=1)
        gy = F.conv2d(t, self.sobel_y, padding=1)
        grad_mag = (gx ** 2 + gy ** 2 + 1e-6).sqrt().view(N, C, H, W)
        # 归一化到 [1, alpha]：平坦区域权重=1，边缘区域权重最高=alpha
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)
        return 1.0 + (self.alpha - 1.0) * grad_mag

    def forward(self, pred, target):
        weight = self._edge_weight(target).detach()  # 权重不参与梯度计算
        return (weight * (pred - target).abs()).mean()


class FrequencyLoss(nn.Module):
    """
    频域损失：在频域中约束高频成分
    帮助恢复细节纹理（干涉条纹、菲涅尔区）
    """
    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def forward(self, pred, target):
        pred_float = pred.float()
        target_float = target.float()

        pred_fft = torch.fft.rfft2(pred_float, norm='ortho')
        target_fft = torch.fft.rfft2(target_float, norm='ortho')

        loss = F.l1_loss(pred_fft.real, target_fft.real) + \
               F.l1_loss(pred_fft.imag, target_fft.imag)
        return loss


class CombinedLoss(nn.Module):
    """
    优化的组合损失函数
    = w_gwl1 * GradientWeightedL1 + w_ssim * MS-SSIM + w_freq * FrequencyLoss

    GradientWeightedL1 替代原始 L1，对边缘/焦散区域施加更强惩罚
    MS-SSIM 维护整体结构感知
    FrequencyLoss 恢复干涉条纹等高频细节

    返回: (total_loss, detail_dict)
    """
    def __init__(self, w_gwl1=1.0, w_ssim=0.5, w_freq=0.4, edge_alpha=5.0):
        super(CombinedLoss, self).__init__()
        self.w_gwl1 = w_gwl1
        self.w_ssim = w_ssim
        self.w_freq = w_freq

        self.gwl1_loss = GradientWeightedL1Loss(alpha=edge_alpha)
        self.ssim_loss = MSSSIMLoss()
        self.freq_loss = FrequencyLoss()

    def forward(self, pred, target):
        gwl1 = self.gwl1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        freq = self.freq_loss(pred, target)

        total = self.w_gwl1 * gwl1 + self.w_ssim * ssim + self.w_freq * freq

        details = {
            "gwl1": gwl1.item(),
            "ssim": ssim.item(),
            "freq": freq.item()
        }

        return total, details


if __name__ == "__main__":
    # 测试损失函数
    pred = torch.randn(2, 1, 512, 512)
    target = torch.randn(2, 1, 512, 512)

    criterion = CombinedLoss()
    loss, details = criterion(pred, target)

    print(f"总损失:          {loss.item():.4f}")
    print(f"GradWeightedL1: {details['gwl1']:.4f}")
    print(f"SSIM损失:        {details['ssim']:.4f}")
    print(f"频域损失:        {details['freq']:.4f}")
