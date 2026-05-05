function u = SSHFT(u,u_curr,eps_r,sigma,lambda,n,k0,w,z0,x,dx,Nz,Nx,dz,pz,z_down)
% 针对水平极化波的分布混合傅里叶算法

% setenv('MW_MINGW64_LOC', "D:\APP\MATLAB\bin\win64\mingw64\x86_64-15.2.0-release-win32-seh-ucrt-rt_v13-rev1\mingw64")
% mex -v CXXFLAGS='$CXXFLAGS -std=c++14' SSHFT_mex.cpp

% =========================================================================
% 优化说明（相比原始版本）：
%
%   1. 零填充FFT技巧：对向量 xr 构造 [0; xr; 0...0] 并做 FFT，则
%         real(FFT)(2:N+1) = DCT-1(xr)
%        -imag(FFT)(2:N+1) = DST-1(xr)
%      一次 FFT 同时得到 DST 和 DCT，消除了 flipud 和对称/反对称扩展构造。
%
%   2. 正变换（两个输入相同）：4次FFT → 2次FFT（对 real/imag 各调用1次）。
%      逆变换（T1、T2 不同输入）：4次FFT不变，但结构简化、内存访问减少。
%      每步迭代总 FFT 次数：8 → 6，理论加速约 25%（FFT 为主要耗时）。
%
%   3. 合并后置不变因子：将 medium_phase、w、(2/pi) 在循环外预乘为 mpw，
%      每步减少 2 次大向量逐元乘法。
%
%   4. 预计算 pz_sq、FFT缓冲区（pad_r/pad_i）及固定索引 idx，避免循环内
%      重复分配临时数组。
%
%   5. 进度条每 200 步刷新一次，减少 I/O 阻塞（原为每步）。
% =========================================================================

fprintf('正在求解SSFT...\n');

% 计算阻抗系数
eps_r_earth = eps_r + 60i*sigma*lambda; % 地表的复相对介电常数
alpha = atan(z0 ./ x(2:end));
beta_imp = 1i*k0*sqrt(eps_r_earth - sin(alpha).^2);

% 连续混合傅立叶变换（CMFT）步进：
% U(p)=∫u(z)[beta*sin(pz)-p*cos(pz)]dz
% u(x+dx,z)=exp(ik0(n-1)dx) * CMFT^{-1}{exp(i dx(sqrt(k0^2-p^2)-k0))U}

dp = pi / ((Nz + 1) * dz);
prop_p = exp(1i * dx * (sqrt(k0^2 - pz.^2) - k0));
medium_phase = exp(1i * k0 * dx * (n - 1));

% ---- 优化①：预计算循环内所有不变量 ----
pz_sq   = pz .^ 2;                     % 避免每步重复平方
mpw     = (2/pi) * medium_phase .* w;  % 合并三个固定因子，每步省2次大向量乘法


% ---- 优化②：预分配 FFT 零填充缓冲区，循环内复用 ----
% pad_r/pad_i 初始化为全零，循环中只更新 idx 区间，其余位置始终为零
N2_fft = 2 * (Nz + 1);                % FFT 点数（与原始 sine_sum 一致）
pad_r  = zeros(N2_fft, 1);
pad_i  = zeros(N2_fft, 1);
idx    = 2 : Nz+1;                    % 有效填充区索引

for ix = 1:Nx-1
    bj = beta_imp(ix);

    % ================================================================
    % 正变换：同一输入 u_curr，2次FFT同时得到 DST 和 DCT
    % 原理：FFT([0; xr; 0...0]) 的实部=DCT-1(xr)，负虚部=DST-1(xr)
    %       无需 flipud，无需构造对称/反对称扩展，且无额外归一化因子
    % ================================================================
    pad_r(idx) = real(u_curr);
    Fr = fft(pad_r);
    Fre = Fr(idx);

    pad_i(idx) = imag(u_curr);
    Fi = fft(pad_i);
    Fie = Fi(idx);

    % DST(u_curr) = -imag(FFT)，DCT(u_curr) = real(FFT)，复数形式组合
    Us = dz * (-imag(Fre) + 1j * (-imag(Fie)));   % ≡ dz * sine_sum(u_curr)
    Uc = dz * ( real(Fre) + 1j * ( real(Fie)));   % ≡ dz * cosine_sum(u_curr)

    % ================================================================
    % CMFT正变换 + 谱域步进，式(2.89)
    % ================================================================
    U  = bj .* Us - pz .* Uc;

    den  = bj^2 + pz_sq;          % 优化③：pz_sq已预算
    pU   = (prop_p ./ den) .* U;  % 合并除法，减少一次中间向量
    T1   = bj  * pU;              % (bj/den) · prop_p · U
    T2   = pz .* pU;              % (pz/den) · prop_p · U

    % ================================================================
    % 逆变换：DST(T1) 和 DCT(T2)，各需 2次FFT（共4次）
    % 同样利用零填充FFT技巧，避免 flipud 和对称扩展
    % ================================================================
    pad_r(idx) = real(T1);
    Fr1 = fft(pad_r);
    Fr1e = Fr1(idx);
    pad_i(idx) = imag(T1);
    Fi1 = fft(pad_i);
    Fi1e = Fi1(idx);

    pad_r(idx) = real(T2);
    Fr2 = fft(pad_r);
    Fr2e = Fr2(idx);
    pad_i(idx) = imag(T2);
    Fi2 = fft(pad_i);
    Fi2e = Fi2(idx);

    ss = dp * (-imag(Fr1e) + 1j * (-imag(Fi1e)));  % ≡ dp * sine_sum(T1)
    sc = dp * ( real(Fr2e) + 1j * ( real(Fi2e)));  % ≡ dp * cosine_sum(T2)

   

    % ================================================================
    % 保存当前步场值，更新 u_curr
    % 优化④：mpw 将 medium_phase、w、(2/pi) 合并，每步减少2次大向量乘法
    % ================================================================
    u(:,ix) = u_curr(z_down);
    u_curr  = mpw .* (ss - sc);

    % 进度条每200步刷新一次
    if mod(ix, 200) == 0 || ix == Nx-1
        pct = ix / (Nx-1);
        nb  = round(pct * 40);
        fprintf('\r|%s%s| %5.1f%% | %d/%d', ...
            repmat('█',1,nb), repmat('░',1,40-nb), pct*100, ix, Nx-1);
    end
end
u(:,end) = u_curr(z_down);
fprintf('\n')

end