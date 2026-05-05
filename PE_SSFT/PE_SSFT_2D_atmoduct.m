% 利用分步傅里叶变换(SSFT)求解大气波导问题
% by huhaixiang  2026.4.24

% 极化     水平极化
% 初始条件 通过天线方向图(高斯)和口径场的傅里叶变换求解(窄角)，使用了进一步推导的解析式
% 上边界   平滑的吸收层
% 下边界   阻抗边界
% 步进算法 Feit‐Fleck近似宽角抛物

% from gpt

clear;clc;
tic
%% 参数设置

% 基础参数
f = 5e9;                             % 频率 
theta0 = deg2rad(0);                 % 仰角
z0 = 15;                             % 天线架设高度
hr = 10;                             % 接收天线高度
beta = deg2rad(3);                   % 波束宽度

c = 3e8;                             % 光速
R = 6378.137;

f_Mhz = f/1e6;
lambda = c / f;                      % 波长
k0 = 2 * pi / lambda;                % 波数

% 地面电磁参数
eps_r = 80; % 相对介电常数
sigma = 4;  % 电导率


% 网格参数
x_max = 250e3;
dx = 100;
z_max = 400;
dz = lambda/2;                     

% 降采样参数  由于计算步长限制导致dz太小，所以计算之后提高dz减小内存开销
step = 10; 


% 方法选择
SSFT_methods = 2;


d = 0.75;  

%% 网格设置 
                          
Nx = floor(x_max/dx)+1;
x = (0:Nx-1)*dx;
x_km = x/1e3;

z_max_wid = z_max/d;                % 加入吸收层对原高度进行扩展      
Nz = floor(z_max_wid/dz);           
z = (1:Nz).' * dz;                  % DST要求z从dz开始到Nz*dz (边界0在网格由DST隐含)
z_km = z/1e3;

z_down = 1:step:Nz;
z_down = z_down(z_down*dz <= z_max+dz*step);  % z_down记录的是降采样且不包含吸收层的部分
Nz_down = length(z_down);

% 频域变量
zn  = (1:Nz).';
pz = zn*pi / (Nz+1) / dz;            

% 电波函数u 
u = zeros(Nz_down,Nx);


%% 求解初始场

p0 = k0*sin(theta0);
w = sqrt(2 * log(2)) / (k0 * sin(beta / 2));
u_curr = exp(-(z - z0).^2 / w^2) .* exp( 1j * p0 .* (z - z0)) ...
    - exp(-(z + z0).^2 / w^2) .* exp(-1j * p0 .* (z + z0));
u_curr = u_curr/(w*sqrt(pi));

%% 吸收上边界的缩放因子
d_rev = 1/(1-d);
w = 0.5+0.5*cos((d_rev*pi*(z-d*z_max_wid))/z_max_wid);
w( z < d*z_max_wid ) = 1;

%% 折射指数  

n = Mmodel(z);


%% SSHFT步进混合傅里叶算法

k1 = exp(1j*k0*dx*(n-2));
k2 = exp(1j*dx*sqrt(k0^2-pz.^2));
for j = 1:Nx-1 
    dst_u = dst(u_curr);
    u(:,j) = u_curr(z_down); 
    u_curr = k1 .* idst( k2.*dst_u ) .* w ; 
end
u(:,end) = u_curr(z_down);
%u = SSHFT(u,u_curr,eps_r,sigma,lambda,n,k0,w,z0,x,dx,Nz,Nx,dz,pz,z_down);


%% 根据z_down重新设置网格

z = z(z_down);
z_km = z/1e3;
Nz = length(z);

%% 传播损耗计算 

fprintf('正在计算传播损耗...\n');

u = abs(u);

r_km = sqrt(x_km.^2 + (z_km - z0/1e3).^2);
%r_km(r_km <= 1e-3) = 1e-3;

% 自由空间传播损耗 L0
% L0 = 32.45 + 20lg f(MHz) + 20lg r(km)
L0 = 32.45 + 20*log10(f_Mhz) + 20*log10(r_km);

% 传播因子 F
% F = 20lg|E/E0| = 20lg(sqrt(r) * |u(x,z)|)
F = 20*log10(sqrt(r_km) .* abs(u)+1e-5);



% 实际传播损耗
% PL = L0 - F
PL = L0 - F;
[~, hr_idx] = min(abs(z - hr));
PL_hr = PL(hr_idx,:);

%% 图像的绘制

fprintf('正在绘制图像...\n');


figure(1)
pcolor(x_km,z,F)
shading flat;
colormap(jet)
%clim([-100 0]);    % 颜色范围锁定
colorbar;
xlabel('距离/km');
ylabel('高度/m');
title(sprintf('传播因子图 freq=%.0fGHz, h=%.0fm', f/1e9, z0)); 

% figure(2)
% plot(x_km,PL_hr)
% xlabel('距离/km');
% ylabel('PL/m');
% title(sprintf('传播损耗图 接收天线高度%dm freq=%.0fGHz, h=%.0fm',hr, f/1e9, z0));


toc



function n = Mmodel(z)


% M 修正折射率
M0 = 338.5;

% 标准大气
C0 = 0.117;          % M-unit/m
M1 = M0 + C0*z;

% 蒸发波导
z0 = 0.15;           % m，粗糙长度/经验常数
d = 30;              % m，蒸发波导高度参数
C0 = 0.117;          % M-unit/m

M2 = M0 + C0*(z - d*log((z + z0)/z0));

% 表面波导
zb = 200;            % m，波导底高度
z_thick = 100;       % m，波导层厚度

C1 = 0.117;          % M-unit/m，下层正常M梯度
C2 = 0.117;          % M-unit/m，上层正常M梯度
Md = 30;             % M亏损量，M-unit

M3 = zeros(size(z));

idx1 = z <= zb;
idx2 = z > zb & z <= zb + z_thick;
idx3 = z > zb + z_thick;

% 波导层以下：正常大气 基础层
M3(idx1) = M0 + C1 .* z(idx1);

% 波导层内：M下降，形成陷获层
M3(idx2) = M0 + C1 * zb ...
              - Md .* (z(idx2) - zb) ./ z_thick;

% 波导层以上：恢复正常正梯度
M3(idx3) = M0 + C1 * zb ...
              - Md ...
              + C2 .* (z(idx3) - zb - z_thick);
M3 = M3 - C0*d*log((z + z0)/z0);  % 这一步加入蒸发波导

% 选择大气模型
M = M3;

% 将M转换为等效折射率
n = 1 + M*1e-6;

end