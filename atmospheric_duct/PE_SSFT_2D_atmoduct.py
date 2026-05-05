# 利用分步傅里叶变换(SSFT)求解大气波导问题
# by huhaixiang  2026.4.24

# 极化     水平极化
# 初始条件 通过天线方向图(高斯)和口径场的傅里叶变换求解(窄角)，使用了进一步推导的解析式
# 上边界   平滑的吸收层
# 下边界   PEC边界
# 步进算法 Feit-Fleck近似宽角抛物

# from gpt

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dst, idst
plt.rcParams['font.sans-serif'] = ['SimHei']      # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False        # 正常显示负号


def Mmodel(z, M0, C0, width, zb, z_thick, C1, C2, Md, evap, impend):


    # 标准大气
    M1 = M0 + C0 * z

    # 蒸发波导
    z0 = 0.15           # m，粗糙长度/经验常数
    M2 = M0 + C0 * (z - width * np.log((z + z0) / z0))

    # 悬空波导
    M3 = np.zeros_like(z)

    idx1 = z <= zb
    idx2 = (z > zb) & (z <= zb + z_thick)
    idx3 = z > zb + z_thick

    # 波导层以下：正常大气 基础层
    M3[idx1] = M0 + C1 * z[idx1]

    # 波导层内：M下降，形成陷获层
    M3[idx2] = M0 + C1 * zb - Md * (z[idx2] - zb) / z_thick

    # 波导层以上：恢复正常正梯度
    M3[idx3] = M0 + C1 * zb - Md + C2 * (z[idx3] - zb - z_thick)

    if evap and impend:
        M = M3 + M2 - M1
    elif impend:
        M = M3
    elif evap:
        M = M2
    else:
        M = M1

    # 将M转换为等效折射率
    n = 1 + M * 1e-6

    return n


def PE_atmoduct(f, theta0, z0, hr, beta, x_max, dx, z_max, dz_lam, step,
                M0,C0,width,zb,z_thick,C1,C2,Md,evap,impend
                ):
    t_start = time.time()

    # 参数设置


    theta0 = np.deg2rad(theta0)         # 仰角
    beta = np.deg2rad(beta)                # 波束宽度
    c = 3e8                             # 光速
    R = 6378.137
    f_Mhz = f / 1e6
    lam = c / f                         # 波长
    k0 = 2 * np.pi / lam                # 波数
    dz = lam / dz_lam
    d = 0.75                            # 加上吸收层之后原最大高度的占比

    # 网格设置

    Nx = int(np.floor(x_max / dx)) + 1
    x = np.arange(Nx) * dx
    x_km = x / 1e3

    z_max_wid = z_max / d                # 加入吸收层对原高度进行扩展
    Nz = int(np.floor(z_max_wid / dz))
    z = (np.arange(1, Nz + 1) * dz).reshape(-1, 1)  # DST要求z从dz开始到Nz*dz
    z_km = z / 1e3

    z_down = np.arange(0, Nz, step)
    z_down = z_down[((z_down + 1) * dz) <= z_max + dz * step]  # z_down记录的是降采样且不包含吸收层的部分
    Nz_down = len(z_down)

    # 频域变量
    zn = np.arange(1, Nz + 1).reshape(-1, 1)
    pz = zn * np.pi / (Nz + 1) / dz

    # 电波函数u
    u = np.zeros((Nz_down, Nx), dtype=complex)

    # 求解初始场

    p0 = k0 * np.sin(theta0)
    w0 = np.sqrt(2 * np.log(2)) / (k0 * np.sin(beta / 2))
    u_curr = np.exp(-((z - z0) ** 2) / w0 ** 2) * np.exp(1j * p0 * (z - z0)) \
        - np.exp(-((z + z0) ** 2) / w0 ** 2) * np.exp(-1j * p0 * (z + z0))
    u_curr = u_curr / (w0 * np.sqrt(np.pi))

    # 吸收上边界的缩放因子
    d_rev = 1 / (1 - d)
    w = 0.5 + 0.5 * np.cos((d_rev * np.pi * (z - d * z_max_wid)) / z_max_wid)
    w[z < d * z_max_wid] = 1

    # 折射指数

    n = Mmodel(z,M0,C0,width,zb,z_thick,C1,C2,Md,evap,impend)

    # SSHFT步进混合傅里叶算法

    k1 = np.exp(1j * k0 * dx * (n - 2))
    k2 = np.exp(1j * dx * np.sqrt(k0 ** 2 - pz ** 2 + 0j))

    for j in range(Nx - 1):
        dst_u = dst(u_curr, type=1, axis=0, norm='ortho')
        u[:, j] = u_curr[z_down, 0]
        u_curr = k1 * idst(k2 * dst_u, type=1, axis=0, norm='ortho') * w

    u[:, -1] = u_curr[z_down, 0]
    # u = SSHFT(u,u_curr,eps_r,sigma,lam,n,k0,w,z0,x,dx,Nz,Nx,dz,pz,z_down)

    # 提取修正折射率
    M_profile = (n - 1) * 1e6
    M_profile_down = M_profile[z_down, 0]

    # 根据z_down重新设置网格

    z = z[z_down]
    z_km = z / 1e3
    Nz = len(z)

    # 传播损耗计算

    print('正在计算传播损耗...')

    u = np.abs(u)

    r_km = np.sqrt(x_km.reshape(1, -1) ** 2 + (z_km - z0 / 1e3) ** 2)
    # r_km[r_km <= 1e-3] = 1e-3

    # 自由空间传播损耗 L0
    # L0 = 32.45 + 20lg f(MHz) + 20lg r(km)
    L0 = 32.45 + 20 * np.log10(f_Mhz) + 20 * np.log10(r_km)

    # 传播因子 F
    # F = 20lg|E/E0| = 20lg(sqrt(r) * |u(x,z)|)
    F = 20 * np.log10(np.sqrt(r_km) * np.abs(u) + 1e-5)

    # 传播损耗
    # PL = L0 - F
    PL = L0 - F
    hr_idx = np.argmin(np.abs(z[:, 0] - hr))
    PL_hr = PL[hr_idx, :]

    # 图像的绘制

    print('正在绘制图像...')

    plt.figure(1)
    plt.pcolormesh(x_km, z[:, 0], F, shading='auto',cmap='jet')
    plt.colorbar()
    plt.xlabel('距离/km')
    plt.ylabel('高度/m')
    plt.title(f'传播因子图 freq={f / 1e9:.0f}GHz, h={z0:.0f}m')



    plt.figure(2)
    plt.plot(x_km, PL_hr)
    plt.xlabel('距离/km')
    plt.ylabel('PL/m')
    plt.title(f'传播损耗图 接收天线高度{hr}m freq={f / 1e9:.0f}GHz, h={z0:.0f}m')

    plt.figure(3)
    plt.plot(M_profile_down, z[:, 0])
    plt.xlabel('修正折射率 M (M-unit)')
    plt.ylabel('高度 / m')
    plt.title('修正折射率 M 剖面图')
    plt.grid(True)

    print(f'{time.time() - t_start:.3f} s')
    plt.show()


if __name__ == '__main__':

    # 天线参数
    f = 5e9  # 频率
    theta0 = 0  # 仰角
    z0 = 15  # 天线架设高度
    hr = 10  # 接收天线高度
    beta = 3  # 波束宽度


    # 网格参数
    x_max = 250e3
    dx = 100
    z_max = 400
    dz_lam = 2       # dz * dz_lam = lam
    step = 10        # 降采样参数  由于计算步长限制导致dz太小，所以计算之后提高dz减小内存开销

    # 波导参数
    M0 = 338.5  # M 修正折射率
    C0 = 0.117  # 大气梯度 M-unit/m
    w = 30  # m，蒸发波导高度参数
    zb = 200  # m，悬空波导底高度
    z_thick = 100  # m，悬空波导层厚度
    C1 = 0.117  # M-unit/m，下层正常M梯度
    C2 = 0.117  # M-unit/m，上层正常M梯度
    Md = 30  # M亏损量，M-unit
    evap = True
    impend = True

    PE_atmoduct(f, theta0, z0, hr, beta, x_max, dx, z_max, dz_lam, step,
                M0,C0,w,zb,z_thick,C1,C2,Md,evap,impend
                )
