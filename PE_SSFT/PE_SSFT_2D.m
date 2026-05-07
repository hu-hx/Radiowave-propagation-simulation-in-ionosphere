% 利用分步傅里叶变换(SSFT)求解电波在电离层传播的抛物方程法(PE)
% by huhaixiang  2026.4.24

% 初始条件 通过天线方向图(高斯)和口径场的傅里叶变换求解(窄角)，使用了进一步推导的解析式
% 上边界   平滑的吸收层
% 下边界   理想导体边界，水平极化波满足u(x,0)=0
% 步进算法 提供了Taylor近似窄角抛物方程和Feit‐Fleck近似宽角抛物方程，采用快速正弦变换



clear;clc;
tic
%% 参数设置

% 基础参数
f = 1e9;                            % 频率 
f_Mhz = f/1e6;
c = 3e8;                             % 光速
lambda = c / f;                      % 波长
k0 = 2 * pi / lambda;                % 波数
theta0 = deg2rad(0);                % 仰角
z0 = 200;                            % 天线架设高度
beta = deg2rad(5);                   % 波束宽度
R = 6378.137;

% 地面电磁参数 PEC不考虑
eps_r = 80; % 相对介电常数
sigma = 4;  % 电导率

% 网格参数
x_max = 500e3;
dx = 100;
z_max = 500;
dz = lambda/4;                     

% 降采样参数  由于计算步长限制导致dz太小，所以计算之后提高dz减小内存开销
step = 2; 

% 方法选择
SSFT_methods = 2;

% 折射指数
absorb = false; % 是否考虑电子碰撞吸收
file_Ne = "./Ne/NEDM_Ne1.txt";

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

n = Ne2n(file_Ne, z_km, f, absorb, false,R);

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


figure(1);
plot(z, abs(u(:,1)), 'b-', 'LineWidth', 1); hold on;
xline(0, 'k--', 'LineWidth', 1); 
xlabel('z (m)'); ylabel('场幅度');
title('高斯方向图天线初始场 \phi(0,z)');
xlim([0,2*z0])
grid on;
saveas(gcf, '高斯方向图天线初始场.png', 'png');


figure(2)
pcolor(x_km,z,abs(u))
shading flat;
%colormap(fliplr(jet))
colormap(cmap(256,[1,3,8]))
colorbar;
xlabel('距离/km');
ylabel('高度/m');
title('无线电波伪彩图');
saveas(gcf, '无线电波伪彩图.png', 'png');



figure(3)
pcolor(x_km,z_km,PL)
shading flat;
colormap(fliplr(jet))
clim([100 200]);    % 颜色范围锁定
colorbar;
xlabel('距离/km');
ylabel('高度/km');
title('抛物方程损失图像');



toc



function n = Ne2n(Ne_flie,z_km,f,absorb,figure,R)

h_km = R * (exp(z_km ./ R) - 1);  % 网格z_km -> 真实高度h_km

% 读取电子浓度Ne
Ne_file = table2array(readtable(Ne_flie,'VariableNamingRule', 'preserve'));
Ne_cm3 = interp1(Ne_file(:,1),Ne_file(:,2),h_km,'linear');
Ne_m3 = Ne_cm3*1e6;         % 原单位:cm-3 ->  m-3

if absorb % 考虑吸收效应 参考"一种基于宽角抛物方程的电离层行进式扰动短波传播效应数值计算方法"
    msis = table2array(readtable("nrlmsis_output.txt",'VariableNamingRule','preserve'));
    O  = interp1(msis(:,6), msis(:,9),  h_km, 'linear');     % 单位:cm-3
    N2 = interp1(msis(:,6), msis(:,10), h_km, 'linear');
    O2 = interp1(msis(:,6), msis(:,11), h_km, 'linear');
    Te = interp1(msis(:,6), msis(:,13), h_km, 'linear');     %单位：K

    v_ei = 54*Ne_cm3./Te.^(3/2);
    v_en = 9.32e-12*N2.*(1-3.44e-5*Te) + 1.21e-10*O2.*(1+2.15e-12*Te.^0.5).*Te+...
        5.49e-10.*O.*Te.^0.5;
    ve = v_en + v_ei;
    Z = ve/(2*pi*f);
else     % 仅考虑折射
    Z = 0;
end

n = sqrt(1 - 80.6*Ne_m3./f^2./(1-1j*Z));
n = n .* (1 + h_km ./ R); 


if figure
    figure(4);
    plot(Ne_m3, h_km, 'black-', 'LineWidth', 1); hold on;
    xlabel('电子浓度'); ylabel('z (km)');
    xlim([0, inf]);
    grid on;
end

end

function cmap = cmap(m, ratios)
    % m: 比色卡总长度（默认256）
    % ratios: 四个颜色的比例 [红到黄, 黄到绿, 绿到蓝]，默认等分
    ratios = ratios / sum(ratios);
    colors = [1.0, 0.0, 0.0;   
              1.0, 1.0, 0.0;  
              0.0, 1.0, 0.0;   
              0.0, 0.0, 1.0]; 
    segment_points = round(ratios * m);
    segment_points(end) = m - sum(segment_points(1:end-1));
    cmap = zeros(m, 3);
    current_idx = 1;
    for i = 1:3
        end_idx = current_idx + segment_points(i) - 1;
        t = linspace(0, 1, segment_points(i))';
        cmap(current_idx:end_idx, :) = (1-t).*colors(i,:) + t.*colors(i+1,:);
        current_idx = end_idx + 1;
    end
end
