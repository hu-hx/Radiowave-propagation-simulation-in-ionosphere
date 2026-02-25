
R12 = 100;                   % R12 index
speed_of_light = 2.99792458e8;
  
elevs = 11:0.5:19;            % initial ray elevation
num_elevs = length(elevs);
freq = 20.0;                 % ray frequency (MHz)
freqs = freq.*ones(size(elevs));
ray_bear = 324.7;            % bearing of rays
origin_lat = 49;          % latitude of the start point of ray
origin_long = 110;         % longitude of the start point of ray
tol = [1e-7 .01 10];         % ODE tolerance and min/max step sizes
nhops = 2;                   % number of hops to raytrace
doppler_flag = 0;            % generate ionosphere 5 minutes later so that
                             % Doppler shift can be calculated
irregs_flag = 0;             % no irregularities - not interested in 
                             % Doppler spread or field aligned irregularities
kp = 0;                      % kp not used as irregs_flag = 0. Set it to a 
                             % dummy value 

%
% generate ionospheric, geomagnetic and irregularity grids
%
max_range = 3200;      % maximum range for sampling the ionosphere (km)
num_range = 1001;        % number of ranges (must be < 2000)
range_inc = max_range ./ (num_range - 1);  % range cell size (km)

start_height = 0 ;      % start height for ionospheric grid (km)
height_inc = 1;         % height increment (km)
num_heights = 1000;      % number of  heights (must be < 2000)

clear iri_options
iri_options.Ne_B0B1_model = 'Bil-2000'; % this is a non-standard setting for 
                                        % IRI but is used as an example

% 坐标轴向量仅依赖于循环外参数，提取到循环外
zkm = 1:height_inc:height_inc*num_heights;
xkm = linspace(0,max_range,num_range);
path = 'D:\desktop\短波传播模拟/testNe/';

% 保存坐标轴文件（内容固定，只需保存一次）
save([path,'zkm.txt'],'zkm','-ascii')
save([path,'xkm.txt'],'xkm','-ascii')

% --- 循环开始 ---
for ii=1:50
year = randi([2010, 2016]);
month = randi([1, 12]);
day = randi([1,25]);

UT = [year month day 0 0];        % UT - year, month, day, hour, minute
disp(ii)


[iono_pf_grid, iono_pf_grid_5, collision_freq, irreg] = ...
    gen_iono_grid_2d(origin_lat, origin_long, R12, UT, ray_bear, ...
                     max_range, num_range, range_inc, start_height, ...
		     height_inc, num_heights, kp, doppler_flag, 'iri2016', ...
		     iri_options);
 

% convert plasma frequency grid to  electron density in electrons/cm^3
iono_Ne_grid = iono_pf_grid.^2 / 80.6164e-6; % 这个单位也是cm-3
%iono_Ne_grid_5 = iono_pf_grid_5.^2 / 80.6164e-6;

n = sqrt(1-iono_pf_grid.^2/freq^2);
%n = n(:,1);

filename = sprintf('%s%04d%02d%02d.dat', path, UT(1), UT(2), UT(3));
save(filename, 'iono_Ne_grid', '-ascii');
end
