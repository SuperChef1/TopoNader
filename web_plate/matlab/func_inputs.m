function inputs = func_inputs(n, coordx, coordy, r, num_bars, random_seed)
% 生成参数化输入样本（覆盖率定义与 WAE_func_edit.py 一致：有效域 mask 上求杆带并集占比）
% 输出尺寸: n x num x 4
% 每一行: [xc, yc, theta_deg, w]
% 杆件数由 num_bars 指定，且所有参数最终都为整数
%
% 有效域：默认全板参与；若脚本同目录存在 Coord_True_coords.mat 且含变量 coords，
%         则与 Python 相同——coords 每行为 [i,j]（COMSOL arr1(i,j) 的 1 基下标），
%         对应格点从覆盖率分子/分母中剔除（mask_coords=false）。

    if nargin < 5 || isempty(num_bars)
        num_bars = 4;
    end
    if nargin >= 6 && ~isempty(random_seed)
        rng(double(random_seed), 'twister');
    end
    validateattributes(num_bars, {'numeric'}, {'scalar', 'integer', 'positive'});
    w_min = 10;      % 宽度下限（整数）
    w_max = 100;     % 宽度上限（整数）

    xmin = ceil(min(coordx(:)));
    xmax = floor(max(coordx(:)));
    ymin = ceil(min(coordy(:)));
    ymax = floor(max(coordy(:)));

    mask_coords = load_mask_coords_wae(size(coordx));

    inputs = zeros(n, num_bars, 4);

    for k = 1:n
        sample = zeros(num_bars, 4);

        % 生成整数参数
        xc    = randi([xmin, xmax], num_bars, 1);
        yc    = randi([ymin, ymax], num_bars, 1);
        theta = randi([0, 179], num_bars, 1);      % 角度取整数
        w     = randi([w_min, w_max], num_bars, 1); % 宽度取整数

        sample(:, 1) = xc;
        sample(:, 2) = yc;
        sample(:, 3) = theta;
        sample(:, 4) = w;

        % 用整数宽度约束材料分布率（覆盖率在 mask 有效格上计算，r_safe 与 Python 对齐）
        s_opt = find_safe_scale(sample, coordx, coordy, r, num_bars, mask_coords);

        % 最终宽度仍然强制为整数
        sample(:, 4) = max(1, floor(s_opt * sample(:, 4)));

        inputs(k,:,:) = sample;
    end
end


function mask = load_mask_coords_wae(sz_coord)
% 与 WAE_func_edit.py 模块级 mask_coords 一致：默认全 true，coords 指定格剔除
    nx = sz_coord(1);
    ny = sz_coord(2);
    mask = true(nx, ny);

    thisDir = fileparts(mfilename('fullpath'));
    matPath = fullfile(thisDir, 'Coord_True_coords.mat');
    if exist(matPath, 'file') ~= 2
        return;
    end
    S = load(matPath);
    if ~isfield(S, 'coords')
        return;
    end
    cc = double(S.coords);
    cc = squeeze(cc);
    if isempty(cc)
        return;
    end
    if numel(cc) < 2
        return;
    end
    if size(cc, 2) ~= 2
        if size(cc, 1) == 2 && size(cc, 2) ~= 2
            cc = cc';
        else
            cc = reshape(cc, [], 2);
        end
    end
    cc = round(cc);
    % 与 Python 读入约定一致：文件中为 COMSOL/MATLAB 的 1 基 (i,j)，与 coordx(i,j) 对齐
    for k = 1:size(cc, 1)
        ii = cc(k, 1);
        jj = cc(k, 2);
        if ii >= 1 && ii <= nx && jj >= 1 && jj <= ny
            mask(ii, jj) = false;
        end
    end
end


function s_opt = find_safe_scale(sample, coordx, coordy, r, num, mask_coords)
% 与 WAE_func_edit.find_safe_scale：r_safe 用 n_valid = nnz(mask_coords)（与 coordx 同尺寸时）

    if num <= 0
        s_opt = 1;
        return;
    end

    if isequal(size(coordx), size(mask_coords))
        n_valid = nnz(mask_coords);
    else
        n_valid = numel(coordx);
    end
    n_valid = max(n_valid, 1);
    r_safe = max(r - 1 / n_valid, 0);

    ratio_fun = @(s) scaled_ratio(sample, s, coordx, coordy, num, mask_coords);
    ratio1 = ratio_fun(1);

    if ratio1 <= r_safe
        s_low = 1;
        s_high = 2;

        while ratio_fun(s_high) <= r_safe && s_high < 1e6
            s_low = s_high;
            s_high = 2 * s_high;
        end

        if s_high >= 1e6
            s_opt = s_low;
            return;
        end
    else
        s_low = 0;
        s_high = 1;
    end

    best_feasible = s_low;

    for it = 1:40
        s_mid = 0.5 * (s_low + s_high);
        ratio_mid = ratio_fun(s_mid);

        if ratio_mid <= r_safe
            best_feasible = s_mid;
            s_low = s_mid;
        else
            s_high = s_mid;
        end
    end

    s_opt = best_feasible;
end


function ratio = scaled_ratio(sample, s, coordx, coordy, num, mask_coords)
    temp = sample;

    if num > 0
        temp(1:num, 4) = max(1, floor(s * temp(1:num, 4)));
    end

    ratio = func_coverage_ratio(temp, coordx, coordy, num, mask_coords);
end


function ratio = func_coverage_ratio(sample, coordx, coordy, num, mask_coords)
% 与 WAE_func_edit.coverage_ratio：filled 为杆带并集；仅在 mask_coords==true 处取均值

    filled = false(size(coordx));

    for i = 1:num
        xc = sample(i, 1);
        yc = sample(i, 2);
        theta_deg = sample(i, 3);
        w = sample(i, 4);

        filled = filled | strip_mask(xc, yc, theta_deg, w, coordx, coordy);
    end

    if ~isequal(size(coordx), size(mask_coords)) || ~any(mask_coords(:))
        if ~isequal(size(coordx), size(mask_coords))
            ratio = mean(filled(:));
        else
            ratio = 0;
        end
        return;
    end

    ratio = nnz(filled & mask_coords) / nnz(mask_coords);
end


function mask = strip_mask(xc, yc, theta_deg, w, coordx, coordy)
% 与 WAE_func_edit.strip_mask：法向距离 y_local = dx*cos(theta)+dy*sin(theta)

    dx = coordx - xc;
    dy = coordy - yc;
    theta_rad = deg2rad(theta_deg);
    y_local = dx .* cos(theta_rad) + dy .* sin(theta_rad);

    mask = abs(y_local) <= w / 2;
end
