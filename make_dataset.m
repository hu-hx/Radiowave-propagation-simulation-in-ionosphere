clear;clc;
tic

% 制作深度学习训练所用的数据集
% SSFT_PE_2D version:1.0 2026.2.18

path = './Ne/';
Ne_files = dir([path,'*.dat']);
gap1 = 10;
gap2 = 5;

for f = 1:50
    for file_i = 1:length(Ne_files)
        fprintf('当前进度: f=%d, file=%d/%d\n', f, file_i, length(Ne_files));
        theta0 = deg2rad(10);
        Ne_file = Ne_files(file_i).name;
        PL = SSFT_PE_2D(path,f,theta0,Ne_file);
        PL = round(PL, 3);
        PL = PL(1:gap1:end,1:gap2:end);
        writematrix(PL, ['output_dataset/',num2str(f),'_',Ne_file(1:end-4),'.dat'], 'Delimiter', ' ');
        clear PL;
        drawnow limitrate;
    end
end




function PL = SSFT_PE_2D(path,f,theta0,Ne_file)
%% 参数设置

% 基础参数
c = 3e8;                             % 光速
lambda = c / (f*1e6);                      % 波长
k0 = 2 * pi / lambda;                % 波数
beta = deg2rad(3);                   % 波束宽度
R = 6378.137;
z0 = 300;

% 网格参数
x_max = 3000e3;
dx = 1000;
z_max = 400e3;
dz = 50;                     

% 方法选择
inti_methods = 1;
SSFT_methods = 3;

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

n = Ne2n(path,Ne_file,f,x_km,z_km);

%% SSFT步进傅里叶算法 水平极化波采用快速正弦变换


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
    u(:,j+1) = k1(:,j+1) .* idst( k2.*dst_u ) .* w ; 
end


%% 传播损耗计算 



%F = 20*log10(abs(u)) + 10*log10(R*sin(x_km/R)+1e-6) + 10*log10(lambda);
PL = -20*log10(abs(u)) + 20*log10(4*pi) + 10*log10(R*sin(x_km/R)+1e-10) - 30*log10(lambda/1e3);

fig = figure('Position',[100, 100, 1000, 500],'Visible', 'off');
pcolor(x_km,z_km,PL)
shading flat;
colormap(fliplr(jet))
clim([90 170]+f);    % 颜色范围锁定
colorbar;

ylim([0,z_max_km_plot])

xlabel('距离/km');
ylabel('高度/km');
title([num2str(f),'-',num2str(Ne_file)]);
exportgraphics(fig, ['./output_dataset/figure/',num2str(f),'_',Ne_file(1:end-4),'.png'], 'Resolution', 150);

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

