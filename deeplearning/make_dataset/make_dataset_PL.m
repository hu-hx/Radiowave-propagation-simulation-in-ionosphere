clear;clc;
tic

% 制作深度学习训练所用的数据集 适配CUG服务器
% SSFT_PE_2D version:1.0 2026.2.18

% 2.26.8:50 制作所有变量变化的数据集 预计计算66个小时 - 3.1.3:30

path_Ne = 'D:\\desktop\\full_unet_train\\full_data\\Ne\\';
Ne_files = dir([path_Ne,'*.dat']);


% 五个变量 f theta0 z0 beta Ne
f_range = [1:2:30,35:5:50];
theta_range = 5:10:25;
z_range = 100:200:300;
beta_range = 3:2:5;
total_files = length(Ne_files);


total_loops = length(f_range) * length(theta_range) * length(z_range) * ...
              length(beta_range) * total_files;


current_count = 0;

for f = f_range
    for theta0 = theta_range
        for z0 = z_range
            for beta = beta_range
                for file_i = 1:total_files
                    
                    current_count = current_count + 1;
                    percent = (current_count / total_loops) * 100;
                    fprintf('[总进度: %3.2f%%]   f=%d theta=%d z=%d beta=%d Ne_files:%d/%d\n', ...
                            percent, f, theta0, z0, beta, file_i, total_files);
                    
                    Ne_file = Ne_files(file_i).name;
                    PL = SSFT_PE_2D(path_Ne,f,theta0,z0,beta,Ne_file);
                    PL = round(PL, 3);
                    PL = imresize(PL, [512, 512]);
                    writematrix(PL, sprintf('D:\\desktop\\full_unet_train\\full_data\\PL\\%d_%d_%d_%d_%s.dat', f,theta0,z0,beta,Ne_file(1:end-4)), 'Delimiter', ' ');
                    clear PL;

                end
            end
        end
    end
end




function PL = SSFT_PE_2D(path,f,theta0_deg,z0,beta_deg,Ne_file)
%% 参数设置

% 基础参数
c = 3e8;                             % 光速
lambda = c / (f*1e6);                      % 波长
k0 = 2 * pi / lambda;                % 波数
beta = deg2rad(beta_deg);                   % 波束宽度
theta0 = deg2rad(theta0_deg);
R = 6378.137;


% 网格参数
x_max = 3000e3;
dx = 2000;
z_max = 400e3;
dz = lambda/4;                     

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

p0 = k0*sin(theta0);
w = sqrt(2 * log(2)) / (k0 * sin(beta / 2));
u(:,1) = exp(-(z - z0).^2 / w^2) .* exp( 1j * p0 .* (z - z0)) ...
    - exp(-(z + z0).^2 / w^2) .* exp(-1j * p0 .* (z + z0));

u(:,1) = u(:,1)/max(abs(u(:,1)));


%% 吸收上边界的缩放因子
d = 1-d;
d_rev = 1/(1-d);
w = 0.5+0.5*cos((d_rev*pi*(z-d*z_max))/z_max);
w( z < d*z_max ) = 1;

%% 折射指数  

n = Ne2n(path,Ne_file,f,x_km,z_km);

%% SSFT步进傅里叶算法 水平极化波采用快速正弦变换

k1 = exp(1i*k0*(n-1)*dx);
k2 = exp( 1i*k0*dx*(sqrt(1-pz.^2/k0^2)-1) );

for j = 1:Nx-1
    dst_u   = dst(u(:,j));
    u(:,j+1) = k1(:,j+1) .* idst( k2.*dst_u ) .* w ;
end


%% 传播损耗计算 



%F = 20*log10(abs(u)) + 10*log10(R*sin(x_km/R)+1e-6) + 10*log10(lambda);
PL = -20*log10(abs(u)+1e-6) + 20*log10(4*pi) + 10*log10(R*sin(x_km/R)+1e-6) - 30*log10(lambda/1e3);

fig = figure('Position',[100, 100, 1000, 500],'Visible', 'off');
pcolor(x_km,z_km,PL)
shading flat;
colormap(fliplr(jet))
clim([90 170]+f);    % 颜色范围锁定
colorbar;

ylim([0,z_max_km_plot])

xlabel('距离/km');
ylabel('高度/km');
title(sprintf('f%d-theta%d-zs%d-beta%d-date%s.dat', f,theta0_deg,z0,beta_deg,Ne_file(1:end-4)));
exportgraphics(fig, sprintf('D:\\desktop\\full_unet_train\\full_data\\figure\\%d_%d_%d_%d_%s.png', f,theta0_deg,z0,beta_deg,Ne_file(1:end-4)), 'Resolution', 150);

close(fig); % 强制关闭图形窗口，释放内存
delete(fig); % 双重保险清除句柄
clear fig;   % 清除变量


function n = Ne2n(path,Ne_file,f,x_km,z_km)
    Ne_file = [path,Ne_file];
    Ne_pharlap = load(Ne_file);
    x_km_pharlap = load([path,'xkm.txt']);
    z_km_pharlap = load([path,'zkm.txt']);
    Ne = interp2(x_km_pharlap,z_km_pharlap',Ne_pharlap,x_km,z_km,'linear')*1e-6;
    Ne(isnan(Ne)) = 0;
    n = sqrt(1 - 80.6*Ne./f^2);
    
end

end

toc

