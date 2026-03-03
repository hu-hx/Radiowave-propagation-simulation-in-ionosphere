clear;clc;
tic

% 利用分步傅里叶变换(SSFT)求解电波在电离层传播的抛物方程法(PE)
% by huhaixiang  2026.1.18
% 后续还可以加入two way,brick,3D

% 1.24 加入了可以使用pharlap工具箱中的折射指数的设置；绘图中加入了射线；考虑了地球曲率，做了变换
% 2.18 修改了PL计算中lambda单位错误的问题; 将展平变换部分放入到n的计算函数中

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

% 网格参数
x_max = 3000e3;
dx = 1000;
z_max = 500e3;
dz = lambda/4;                     

% 绘图参数
fig_step = 5;                       % 数据绘图的间隔，减少绘图时间防止崩溃

% 方法选择
inti_methods = 1;
SSFT_methods = 3;

% 折射指数
n_mat = true;
absorb = false; % 是否考虑电子碰撞吸收
file_Ne = "NEDM_Ne1.txt";

d = 0.2;  

%% 网格设置
                          
Nx = floor(x_max/dx)+1;
x = (0:Nx-1)*dx;
x_km = x/1e3;

                         % 吸收上边界的范围占比
z_max_km_plot = z_max/1e3;
z_max = z_max/(1-d);                      
Nz = floor(z_max/dz);           
z = (1:Nz).' * dz;                  % DST要求z从dz开始到Nz*dz (边界0在网格由DST隐含)
z_km = z/1e3;

% DST 谱变量
m  = (1:Nz).';
pz = m*pi / (Nz+1) / dz;            

% 电波函数u 
u = zeros(Nz,Nx);

%% 求解初始场

switch inti_methods

    case 1 % 基于《电波传播的抛物方程方法》
        u_fs1 = k0*beta/(2*sqrt(2*pi*log(2)))*(exp(1i*k0*theta0*z)).*exp(-beta^2/(8*log(2))*k0^2*(z-z0).^2);
        u_fs2 = k0*beta/(2*sqrt(2*pi*log(2)))*(exp(-1i*k0*theta0*z)).*exp(-beta^2/(8*log(2))*k0^2*(z+z0).^2);
        u(:,1) = u_fs1 - u_fs2;   
end
u(:,1) = u(:,1)/max(abs(u(:,1)));



%% 吸收上边界的缩放因子
d = 1-d;
d_rev = 1/(1-d);
w = 0.5+0.5*cos((d_rev*pi*(z-d*z_max))/z_max);
w( z < d*z_max ) = 1;

%% 折射指数  

n = Ne2n(file_Ne, z_km,R,Nz,d, f_Mhz, absorb, n_mat);

%% SSFT步进傅里叶算法 水平极化波采用快速正弦变换

fprintf('正在求解SSFT...\n');

switch SSFT_methods
    % k1:折射指数项
    % k2:绕射指数项
    case 1   % Talor展开近似的窄角抛物方程 SPE
        k1 = exp(1i*k0*(n.^2-1)*dx/2);   
        k2 = exp(-1i.*pz.^2*dx/(2*k0));

    case 2   % Feit‐Fleck型宽角抛物方程，不知怎么推导出来
        k1_1 = exp(-k0*imag(n)*dx); 
        k1_2 = exp(1j*k0*dx*(real(n)-2));
        k1 = k1_1.*k1_2;
        k2 = exp(1j*dx*sqrt(k0^2-pz.^2));

    case 3   % Feit‐Fleck型宽角抛物方程，可以正常推导出来
        k1 = exp(1i*k0*(n-1)*dx);       
        k2 = exp( 1i*k0*dx*(sqrt(1-pz.^2/k0^2)-1) );
end

 
for j = 1:Nx-1 
    dst_u   = dst(u(:,j));
    u(:,j+1) = k1 .* idst( k2.*dst_u ) .* w ; 
end


%% 传播损耗计算 

fprintf('正在计算传播损耗...\n');

%F = 20*log10(abs(u)) + 10*log10(R*sin(x_km/R)+1e-6) + 10*log10(lambda/1e3);
L = -20*log10(abs(u)) + 20*log10(4*pi) + 10*log10(R*sin(x_km/R)+1e-10) - 30*log10(lambda/1e3);

%% 图像的绘制

fprintf('正在绘制图像...\n');

step = fig_step;                    % 防止图像显示过慢或者卡死
secx = 1:step:size(x_km,2);
secz = 1:step:size(z_km,1);


% figure(1);
% plot(z, abs(u(:,1)), 'b-', 'LineWidth', 1); hold on;
% xline(0, 'k--', 'LineWidth', 1); 
% xlabel('z (m)'); ylabel('场幅度');
% title('高斯方向图天线初始场 \phi(0,z)');
% xlim([0,2*z0])
% grid on;



% figure(3)
% pcolor(x_km(secx),z_km(secz),abs(u(secz,secx)))
% shading flat;
% colormap(fliplr(jet))
% %colormap(cmap1(256,[1,6,12]))
% colorbar;
% xlabel('距离/km');
% ylabel('高度/km');
% title('基于抛物方程的无线电波传播');
%saveas(gcf, '无线电传播图.png', 'png');



figure('Position',[100, 100, 1000, 500])
%figure(50)
pcolor(x_km(secx),z_km(secz),L(secz,secx))
shading flat;
colormap(fliplr(jet))
clim([100 180]);    % 颜色范围锁定
colorbar;
hold on;

ray_path_data = load("ray_path_data.mat").ray_path_data;
for idx = 1:length(ray_path_data)
    gndrng = ray_path_data(idx).ground_range;
    h_real = ray_path_data(idx).height;
    z_flat_plot = R * log(1 + h_real ./ R);

    % 在图上绘制转换后的坐标
    plot(gndrng, z_flat_plot, 'Color', 'w', 'LineWidth', 0.5); % 建议用白色或灰色对比度更高
end
hold off;

ylim([0,z_max_km_plot])

xlabel('距离/km');
ylabel('高度/km');
title('抛物方程和射线追踪对照');
saveas(gcf, '../抛物方程和射线追踪对照.png', 'png');

% figure(5)
% l = mean(L);
% plot(x_km(secx),l(secx))
toc



function n = Ne2n(Ne_flie,z_km,R,Nz,d,f_Mhz,absorb,n_mat)

h_km = R * (exp(z_km ./ R) - 1);  % 网格z_km -> 真实高度h_km

if n_mat
n_pharlap = load('n.mat').n;
z_km_pharlap = load('zkm.mat').zkm;
n = interp1(z_km_pharlap,n_pharlap,h_km,'linear','extrap');
n = n .* (1 + h_km ./ R); 
idx_sponge_start = floor(Nz * d * 0.8); 
n(idx_sponge_start:end) = n(idx_sponge_start);
return
end


% 读取电子浓度Ne
Ne_file = table2array(readtable(Ne_flie,'VariableNamingRule', 'preserve'));
Ne_cm3 = interp1(Ne_file(:,1),Ne_file(:,2),h_km,'linear');
Ne_e12m3 = Ne_cm3*1e-6;         % 原单位:cm-3 -> 10^12 m-3

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
    Z = ve/(2*pi*f_Mhz*1e6);
else     % 仅考虑折射
    Z = 0;
end

n = sqrt(1 - 80.6*Ne_e12m3./f_Mhz^2./(1-1j*Z));

% 高度60km以下采用对流层大气折射率的指数模型
idx_60 = find(h_km < 60);
N = 313*exp(-0.137*h_km(idx_60));                   % 折射率，单位为km
n(idx_60) = 1 + N*1e-6;


%  应用地球展开变换得到 PE 等效折射率
% 精确公式: n_PE(z) = n_raw(h) * (R + h) / R
% 其中 (R+h)/R 正好等于 exp(z/R)
n = n .* (1 + h_km ./ R); 
% 或者写成: n = n_raw .* exp(z_km ./ R); 结果是一样的

% 防止折射指数无限增长
idx_sponge_start = floor(Nz * d * 0.8); 
n(idx_sponge_start:end) = n(idx_sponge_start);

end

