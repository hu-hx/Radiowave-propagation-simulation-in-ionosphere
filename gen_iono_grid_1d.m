%
% Name :
%   gen_iono_grid_1d.m
%
% Purpose :
%   Generates ionospheric plasma density, collision frequency and irregularity
%   grid as a function of range and height using IRI. 
%   
%   ** MODIFIED VERSION (1D) **: This function assumes the ionosphere is 
%   horizontally stratified (does not change with range). It calculates the 
%   profile once at the origin and replicates it across the range grid.
%   The format of the outputs remains identical to gen_iono_grid_2d.m for 
%   compatibility with the 2d raytracing engine.
%
% Inputs :
%   (Same as gen_iono_grid_2d.m)
%
% Outputs :
%   (Same as gen_iono_grid_2d.m)
%
% Author:
%   Based on gen_iono_grid_2d.m by M.A. Cervera
%   Modified to 1D version.
%

function [iono_pf_grid,iono_pf_grid_5,collision_freq,irreg,iono_te_grid] = ...
    gen_iono_grid_1d(origin_lat, origin_lon, R12, UT, azim, ...
		     max_range, num_range, range_inc, start_height, ...
		     height_inc, num_heights, kp, doppler_flag, ...
		     profile_type, varargin);

  re_eq = 6378137.0;                 % equatorial radius of Earth
  dtor = pi / 180.0d0;               % degrees to radians conversion
  pfsq_conv = 80.6163849431291e-12;  % mult. factor to convert elec density
				     % in m^-3 to plasma freq squared in MHz^2

  % get the iri_options structure if it has been input
  if (length(varargin) > 0) 
    iri_options = varargin{1};
  else
    iri_options = struct;
  end

  % make sure we have a valid profile type
  if (~exist('profile_type'))
    profile_type = 'iri';
  end
  if (~strcmp(lower(profile_type), 'chapman_fllhc') & ...
	~strcmp(lower(profile_type), 'chapman') & ...
	~strcmp(lower(profile_type), 'iri')  & ...
	~strcmp(lower(profile_type), 'iri2007')  & ...
	~strcmp(lower(profile_type), 'iri2012')  & ...
	~strcmp(lower(profile_type), 'iri2016')  & ...
	~strcmp(lower(profile_type), 'firi') )
    error('invalid profile type')
    return
  end
  fllhc_flag = 0;
  if (strcmp(lower(profile_type), 'chapman_fllhc'))
    profile_type = 'chapman';
    fllhc_flag = 1;
  end

  % array of heights (km) - Row Vector [1 x num_heights]
  height_arr = [0:num_heights-1].*height_inc+start_height;

  % -----------------------------------------------------------------------
  % 1D Modification: 
  % We calculate the parameters ONLY at the origin, and then replicate
  % them across the range dimension.
  % -----------------------------------------------------------------------

  % 1. Generate the ionospheric profile at the origin
  re_wgs84 = earth_radius_wgs84(origin_lat);
  ht = re_eq - re_wgs84;
  origin_lat_gc = wgs842gc_lat(origin_lat, ht);
  origin_lon_gc = origin_lon;
  
  % gen_iono_profile returns ROW vectors (1 x num_heights)
  [iono_pf_prof, iono_pf_prof5, iono_extra, T_e, T_ion] = ...
    gen_iono_profile(origin_lat, origin_lon, num_heights, start_height, ...
                     height_inc, origin_lat_gc, origin_lon_gc, ...
                     UT, R12, profile_type, iri_options, ...
                     doppler_flag, fllhc_flag);

  % 2. Calculate Neutral Densities at the origin
  % Create lat/lon arrays matching height_arr (Row vectors)
  lat_arr = origin_lat * ones(size(height_arr));
  lon_arr = origin_lon * ones(size(height_arr));
  
  if (R12 == -1)
      [neutral_dens, temp] = nrlmsise00(lat_arr, lon_arr, height_arr, UT);
  else
      % calculate f10.7 : see Davies, "Ionospheric Radio", 1990, pp442
      f107 = 63.75 + R12*(0.728 + R12*0.00089);
      [neutral_dens, temp] = nrlmsise00(lat_arr, lon_arr, height_arr, UT, ...
                                    f107, f107, 4);
  end
  % neutral_dens is typically [9 x num_heights]

  % 3. Calculate Collision Frequency at the origin
  % elec_dens must match T_e orientation (Row vector)
  elec_dens = iono_pf_prof.^2 ./ pfsq_conv; 
  
  % All inputs here are consistent (1 x N or matching columns of 9 x N)
  collision_freq_prof = eff_coll_freq(T_e, T_ion, elec_dens, neutral_dens);

  % 4. Calculate Irregularity Parameters at the origin
  if doppler_flag
      dop_spread = dop_spread_eq(origin_lat, origin_lon, UT, R12);
  else
      dop_spread = 0;
  end

  % call geomag field routine at phase screen height of irregularities (300km)
  long = origin_lon;
  if (long < 0) long = long + 360; end
  mag_field = igrf2016(origin_lat, origin_lon, UT, 300);
  dip = mag_field(8);
  dec = mag_field(10);

  % generate the irregularity strength
  strength = irreg_strength(origin_lat, origin_lon, UT, kp);

  % Irregularity parameter vector (4x1)
  irreg_parms = [strength; dip; dec; dop_spread];

  % -----------------------------------------------------------------------
  % Populate Output Grids by Replicating 1D Profiles
  % -----------------------------------------------------------------------
  % The output grids must be [num_heights x num_range].
  % The profiles calculated above are [1 x num_heights] (Row vectors).
  % We must Transpose (.') them to columns before replicating.
  
  iono_pf_grid   = repmat(iono_pf_prof.', 1, num_range);
  iono_pf_grid_5 = repmat(iono_pf_prof5.', 1, num_range);
  iono_te_grid   = repmat(T_e.', 1, num_range);
  collision_freq = repmat(collision_freq_prof.', 1, num_range);
  
  % irreg_parms is already 4x1, so we replicate it to 4 x num_range
  irreg = repmat(irreg_parms, 1, num_range);

  return
end


%
% subfunction which generates ionospheric plasma frequency height profile at
% a given input latitude and longitude
% (This remains unchanged from the original gen_iono_grid_2d.m)
%
function [iono_pf_prof,iono_pf_prof5,iono_extra,iono_te_prof,iono_ti_prof] = ...
      gen_iono_profile(lat, lon, num_heights, start_height, height_inc, ...
		       origin_lat_gc, origin_lon_gc, UT, R12, ...
		       profile_type, iri_options, doppler_flag, fllhc_flag);

  re_eq = 6378137.0;                 % equatorial radius of Earth
  dtor = pi / 180.0d0;               % degrees to radians conversion
  pfsq_conv = 80.6163849431291e-12;  % mult. factor to convert elec. density

  % UT 5 minutes later
  UT_5 = UT + [0 0 0 0 5];
  if (UT_5(5) > 59)
    UT_5(5) = UT_5(5) - 60;
    UT_5(4) = UT_5(4) + 1;
    if (UT_5(4) > 23) UT_5(4) = 0; end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % This is IRI2016 with or without FIRI option %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if (strcmp(lower(profile_type), 'iri') | ...
	  strcmp(lower(profile_type), 'iri2016') | ...
	  strcmp(lower(profile_type), 'firi') )

    firi_flag = strcmp(lower(profile_type), 'firi');

    if (firi_flag)
      % call IRI 2016, with FIRI option on
      [iono, iono_extra] = iri2016_firi_interp(lat, lon, R12, UT, ...
	           start_height, height_inc, num_heights, iri_options);

      if doppler_flag
	[iono5, iono_extra5] = iri2016_firi_interp(lat, lon, R12, ...
	           UT_5, start_height, height_inc, num_heights, iri_options);
      end

    else
      % call IRI 2016	    
      [iono, iono_extra] = iri2016(lat, lon, R12, UT, start_height, ...
	                           height_inc, num_heights, iri_options);

      if doppler_flag
	  [iono5, iono_extra5] = iri2016(lat, lon, R12, UT_5, ...
	             start_height, height_inc, num_heights, iri_options);
      end
    end

    % get the electron density (Num electrons per m^3)
    elec_dens = iono(1, :);
    idx_neg = find(elec_dens == -1);
    elec_dens(idx_neg) = 0;

    if doppler_flag
      elec_dens5 = iono5(1, :);
      idx_neg = find(elec_dens5 == -1);
      elec_dens5(idx_neg) = 0;
    end

    % get the electron and ion temperature profiles
    iono_ti_prof = iono(3, :);
    iono_te_prof = iono(4, :);
    iono_ti_prof(iono_ti_prof == -1) = NaN;
    iono_te_prof(iono_te_prof == -1) = NaN;

    % calculate plasma frequency (MHz) profile and fill the array
    iono_pf_prof = sqrt(elec_dens .* pfsq_conv);     % (MHz)
    if doppler_flag
	iono_pf_prof5 = sqrt(elec_dens5 .* pfsq_conv);  % (MHz)
    else
	iono_pf_prof5 = sqrt(elec_dens .* pfsq_conv);
    end

  %%%%%%%%%%%%%%%%%%%
  % This is IRI2012 %
  %%%%%%%%%%%%%%%%%%%
  elseif (strcmp(lower(profile_type), 'iri2012') )

     % call IRI 2012	   
    [iono, iono_extra] = iri2012(lat, lon, R12, UT, start_height, ...
	                         height_inc, num_heights);
    if doppler_flag
      [iono5, iono_extra5] = iri2012(lat, lon, R12, UT_5, start_height, ...
	                             height_inc, num_heights);
    end

    % get the electron density (Num electrons per m^3)
    elec_dens = iono(1, :);
    idx_neg = find(elec_dens == -1);
    elec_dens(idx_neg) = 0;

    if doppler_flag
      elec_dens5 = iono5(1, :);
      idx_neg = find(elec_dens5 == -1);
      elec_dens5(idx_neg) = 0;
    end

    % get the electron and ion temperature profiles
    iono_ti_prof = iono(3, :);
    iono_te_prof = iono(4, :);
    iono_ti_prof(iono_ti_prof == -1) = NaN;
    iono_te_prof(iono_te_prof == -1) = NaN;

    % calculate plasma frequency (MHz) profile and fill the array
    iono_pf_prof = sqrt(elec_dens .* pfsq_conv);     % (MHz)
    if doppler_flag
      iono_pf_prof5 = sqrt(elec_dens5 .* pfsq_conv);  % (MHz)
    else
      iono_pf_prof5 = sqrt(elec_dens .* pfsq_conv);
    end

  %%%%%%%%%%%%%%%%%%%
  % This is IRI2007 %
  %%%%%%%%%%%%%%%%%%%
  elseif strcmp(lower(profile_type), 'iri2007')

    % IRI2007 only returns 100 values for electron density with height - so
    % determine the number of  multiple calls required.
    max_iri_numhts = 100;
    num_iri_calls = ceil(num_heights ./ max_iri_numhts);
    for ii = 1:num_iri_calls
      % call IRI 2007
      height_start = start_height + (ii - 1) .* max_iri_numhts .* height_inc;
      [iono, iono_extra] = iri2007(lat, lon, R12, UT, height_start, height_inc);

      if doppler_flag
	[iono5, iono_extra5] = iri2007(lat, lon, R12, UT_5, height_start, ...
	                               height_inc);
      end

      % get the electron density (Num electrons per m^3)
      remaining_heights = num_heights - (ii - 1)*max_iri_numhts;
      idx_end = min([remaining_heights max_iri_numhts]);

      elec_dens = iono(1, 1:idx_end);
      idx_neg = find(elec_dens == -1);
      elec_dens(idx_neg) = 0;

      if doppler_flag
	elec_dens5 = iono5(1, 1:idx_end);
	idx_neg = find(elec_dens5 == -1);
	elec_dens5(idx_neg) = 0;
      end

      % get the electron and ion temperature profiles
      ion_temp  = iono(3, 1:idx_end);
      elec_temp = iono(4, 1:idx_end);

      % calculate plasma frequency (MHz) profile and fill the array
      idx = [(ii - 1)*max_iri_numhts + 1 :  ...
	  ii*max_iri_numhts - (max_iri_numhts - idx_end)];
      iono_pf_prof(idx) = sqrt(elec_dens .* pfsq_conv);     % (MHz)
      if doppler_flag
	iono_pf_prof5(idx) = sqrt(elec_dens5 .* pfsq_conv);  % (MHz)
      else
	iono_pf_prof5(idx) = sqrt(elec_dens .* pfsq_conv);
      end
      iono_te_prof(idx) = elec_temp;
      iono_ti_prof(idx) = ion_temp;
      iono_ti_prof(iono_ti_prof == -1) = NaN;
      iono_te_prof(iono_te_prof == -1) = NaN;
    end

  end

  return
end