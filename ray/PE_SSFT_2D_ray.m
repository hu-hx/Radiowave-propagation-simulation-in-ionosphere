% 抛物方程法和射线追踪的比较
% 抛物方程：PE_SSFT.m
% 射线追踪：pharlap工具箱



clear;clc;
tic
%% 参数设置

% 基础参数
f = 20e6;                            % 频率 
f_Mhz = f/1e6;
c = 3e8;                             % 光速
lambda = c / f;                      % 波长
k0 = 2 * pi / lambda;                % 波数
theta0 = deg2rad(15);                % 仰角
z0 = 30;                            % 天线架设高度
beta = deg2rad(3);                   % 波束宽度
R = 6378.137;

% 地面电磁参数 PEC不考虑
eps_r = 80; % 相对介电常数
sigma = 4;  % 电导率

% 网格参数
x_max = 3000e3;
dx = 1000;
z_max = 400e3;
dz = lambda/2;                     

% 降采样参数  由于计算步长限制导致dz太小，所以计算之后提高dz减小内存开销
step = 50; 

% 方法选择
SSFT_methods = 2;


d = 0.75;  

%% 网格设置 含有吸收层
                          
Nx = floor(x_max/dx)+1;
x = (0:Nx-1)*dx;
x_km = x/1e3;

z_max_wid = z_max/d;                      
Nz = floor(z_max_wid/dz);           
z = (1:Nz).' * dz;                  % DST要求z从dz开始到Nz*dz (边界0在网格由DST隐含)
z_km = z/1e3;

z_down = 1:step:Nz;
z_down = z_down(z_down*dz <= z_max+dz*step);  % z_down记录的是降采样且不包含吸收层的部分
Nz_down = length(z_down);

% DST 谱变量
m  = (1:Nz).';
pz = m*pi / (Nz+1) / dz;            

% 电波函数u 
u = zeros(Nz_down,Nx);


%% 求解初始场

p0 = k0*sin(theta0);
w = sqrt(2 * log(2)) / (k0 * sin(beta / 2));
u_curr = exp(-(z - z0).^2 / w^2) .* exp( 1j * p0 .* (z - z0)) ...
    - exp(-(z + z0).^2 / w^2) .* exp(-1j * p0 .* (z + z0));
u_curr = u_curr/(w*sqrt(pi));
u_curr = u_curr/max(abs(u_curr));

%% 吸收上边界的缩放因子
d_rev = 1/(1-d);
w = 0.5+0.5*cos((d_rev*pi*(z-d*z_max_wid))/z_max_wid);
w( z < d*z_max_wid ) = 1;

%% 折射指数  

n = Ne2n(z_km,R,Nz,d);

%% SSFT步进傅里叶算法 PEC地表水平极化波采用快速正弦变换

fprintf('正在求解SSFT...\n');

switch SSFT_methods
    % k1:折射指数项
    % k2:绕射指数项
    case 1   % Talor展开近似的窄角抛物方程 SPE
        k1 = exp(1i*k0*(n.^2-1)*dx/2);   
        k2 = exp(-1i.*pz.^2*dx/(2*k0));

    case 2   % Feit‐Fleck型宽角抛物方程，
        k1 = exp(1j*k0*dx*(n-2));
        k2 = exp(1j*dx*sqrt(k0^2-pz.^2));
end

 
for j = 1:Nx-1 
    dst_u = dst(u_curr);
    u(:,j) = u_curr(z_down); 
    u_curr = k1 .* idst( k2.*dst_u ) .* w ; 
end
u(:,end) = u_curr(z_down);

%% 根据z_down重新设置网格

z = z(z_down);
z_km = z/1e3;
Nz = length(z);


%% 传播损耗计算 

fprintf('正在计算传播损耗...\n');

u = abs(u);
%F = 20*log10(u) + 10*log10(R*sin(x_km/R)+1e-6) + 10*log10(lambda);
PL = -20*log10(u+1e-6) + 20*log10(4*pi) + 10*log10(R*sin(x_km/R)+1) - 30*log10(lambda/1e3);


%% 图像的绘制

fprintf('正在绘制图像...\n');


figure(1)
pcolor(x_km,z_km,PL)
shading flat;
colormap(fliplr(jet))
clim([100 180]);    % 颜色范围锁定
colorbar;
xlabel('距离/km');
ylabel('高度/km');
title('抛物方程损失图像');

% 绘制射线追踪部分
hold on;
ray_path_data = load("./ray_data/ray_path_data.mat").ray_path_data;
for idx = 1:length(ray_path_data)
    gndrng = ray_path_data(idx).ground_range;
    h_real = ray_path_data(idx).height;
    z_flat_plot = R * log(1 + h_real ./ R);

    % 在图上绘制转换后的坐标
    plot(gndrng, z_flat_plot, 'Color', 'w', 'LineWidth', 0.5); % 建议用白色或灰色对比度更高
end
hold off;



toc


function n = Ne2n(z_km,R,Nz,d)

h_km = R * (exp(z_km ./ R) - 1);  % 网格z_km -> 真实高度h_km


n_pharlap = load('./ray_data/n.mat').n;
z_km_pharlap = load('./ray_data/zkm.mat').zkm;
n = interp1(z_km_pharlap,n_pharlap,h_km,'linear','extrap');
n = n .* (1 + h_km ./ R); 
idx_sponge_start = floor(Nz * d * 0.8); 
n(idx_sponge_start:end) = n(idx_sponge_start);

end

