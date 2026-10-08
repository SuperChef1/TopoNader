function [output, model, coverage_ratio] = func_outputs_inspect( ...
        input_sample, model_file, config_file)
%FUNC_OUTPUTS_INSPECT 检查一个 inputs 样本的 COMSOL 几何并计算结果。
%
% 典型调用（先双击 .mat 文件，将 inputs 加载到 MATLAB 工作区）：
%
%   dbstop in func_outputs_inspect at 64
%   output = func_outputs_inspect(inputs(1, :, :));
%
% input_sample 只允许包含一个样本，支持以下形状：
%   - 1 x num_bars x 4，例如 inputs(1, :, :)
%   - num_bars x 4，例如 squeeze(inputs(1, :, :))
%   - 1 x (num_bars*4)，即一个展平样本
% 每根杆的四个参数依次为 [xc, yc, theta_deg, width]，杆件数由形状推断。
%
% model_file 为可选的 COMSOL .mph 路径；省略时使用本文件同目录下的
% fuban_Beam.mph。
% config_file 为可选的实验配置路径；省略时使用 ../configs/default.json。
%
% 输出
%   output
%       当前样本的目标值，计算方式与 func_outputs_nowrite 相同。
%   model
%       可选的当前 COMSOL 模型对象。
%   coverage_ratio
%       杆带并集在有效离散网格上的材料覆盖率，与 Topo_Nader 优化约束口径一致。
%
% 调试方法
%   在下方标有 "INSPECTION BREAKPOINT" 的语句处设置断点。程序停下时，
%   几何和网格已经生成，但求解尚未开始。在 MATLAB 的 K>> 提示符执行：
%
%       mphlaunch(model, 10000);
%
%   查看完成后，执行 dbcont 继续求解，或执行 dbquit 退出调试。

    if nargin < 2 || isempty(model_file)
        model_file = fullfile(fileparts(mfilename('fullpath')), 'fuban_Beam.mph');
    end
    if nargin < 3 || isempty(config_file)
        config_file = fullfile( ...
            fileparts(mfilename('fullpath')), '..', 'configs', 'default.json');
    end
    model_file = char(model_file);
    config_file = char(config_file);
    if exist(model_file, 'file') ~= 2
        error('func_outputs_inspect:MissingModel', '未找到 COMSOL 模型: %s', model_file);
    end
    if exist(config_file, 'file') ~= 2
        error('func_outputs_inspect:MissingConfig', '未找到实验配置: %s', config_file);
    end

    rod_params = local_normalize_single_sample(input_sample);
    coverage_ratio = local_coverage_ratio(rod_params, config_file);

    model = mphload(model_file);
    model.disableUpdates(true);

    local_apply_rod_params(model, rod_params);

    model.component('comp1').geom('geom1').run;
    model.component('comp1').mesh('mesh1').run;

    model.disableUpdates(false);
    inspection_breakpoint = true; %#ok<NASGU> % <-- INSPECTION BREAKPOINT

    model.sol('sol1').runAll;

    disp1 = model.result.numerical('av1').getReal;
    disp2 = model.result.numerical('av2').getReal;
    output = 0.1 / (disp1 + disp2);

    fprintf(1, ...
        'disp1=%g, disp2=%g, output=%g, material_coverage=%g (%.2f%%)\n', ...
        disp1, disp2, output, coverage_ratio, 100 * coverage_ratio);
end


function rod_params = local_normalize_single_sample(input_sample)
    values = double(input_sample);

    if ndims(values) == 3 && size(values, 3) == 4
        if size(values, 1) ~= 1
            error( ...
                'func_outputs_inspect:MultipleSamples', ...
                '该调试函数一次只检查一个样本，请传入 inputs(i, :, :)。');
        end
        num_bars = size(values, 2);
        rod_params = reshape(values, num_bars, 4);
    elseif ismatrix(values) && size(values, 2) == 4
        num_bars = size(values, 1);
        rod_params = values;
    elseif ismatrix(values) && size(values, 1) == 1 && ...
            mod(size(values, 2), 4) == 0
        num_bars = size(values, 2) / 4;
        rod_params = reshape(values, 4, num_bars)';
    else
        error( ...
            'func_outputs_inspect:InvalidInputs', ...
            ['input_sample 必须是一个样本，形状为 1×num_bars×4、' ...
             'num_bars×4 或 1×(num_bars*4)。']);
    end

    if num_bars < 1
        error('func_outputs_inspect:EmptyInputs', 'input_sample 不能为空。');
    end
    if any(~isfinite(rod_params), 'all')
        error('func_outputs_inspect:NonFiniteInputs', 'input_sample 包含 NaN 或 Inf。');
    end
end


function ratio = local_coverage_ratio(rod_params, config_file)
% 与 topo_nader.optimization.geometry.TopoNaderGeometry.coverage 保持一致。
    config = jsondecode(fileread(config_file));
    if ~isfield(config, 'optimization')
        error( ...
            'func_outputs_inspect:InvalidConfig', ...
            '配置文件缺少 optimization 节点: %s', config_file);
    end
    optimization = config.optimization;
    required_fields = {'grid_nx', 'grid_ny', 'grid_step', 'grid_x0', 'grid_y0'};
    for field_index = 1:numel(required_fields)
        field_name = required_fields{field_index};
        if ~isfield(optimization, field_name)
            error( ...
                'func_outputs_inspect:InvalidConfig', ...
                '配置文件 optimization 缺少字段 %s: %s', field_name, config_file);
        end
    end

    grid_nx = double(optimization.grid_nx);
    grid_ny = double(optimization.grid_ny);
    grid_step = double(optimization.grid_step);
    grid_x0 = double(optimization.grid_x0);
    grid_y0 = double(optimization.grid_y0);
    validateattributes(grid_nx, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(grid_ny, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(grid_step, {'numeric'}, {'scalar', 'positive', 'finite'});

    x = grid_x0 - (0:grid_nx - 1) * grid_step;
    y = grid_y0 - (0:grid_ny - 1) * grid_step;
    [coordx, coordy] = ndgrid(x, y);
    design_mask = local_load_design_mask(size(coordx));
    filled = false(size(coordx));

    for rod_index = 1:size(rod_params, 1)
        xc = rod_params(rod_index, 1);
        yc = rod_params(rod_index, 2);
        theta = deg2rad(rod_params(rod_index, 3));
        width = rod_params(rod_index, 4);
        normal_distance = ...
            (coordx - xc) .* cos(theta) + (coordy - yc) .* sin(theta);
        filled = filled | abs(normal_distance) <= width / 2;
    end

    if any(design_mask(:))
        ratio = nnz(filled & design_mask) / nnz(design_mask);
    else
        ratio = 0.0;
    end
end


function mask = local_load_design_mask(grid_size)
    mask = true(grid_size);
    mask_file = fullfile( ...
        fileparts(mfilename('fullpath')), 'Coord_True_coords.mat');
    if exist(mask_file, 'file') ~= 2
        return;
    end

    payload = load(mask_file, 'coords');
    if ~isfield(payload, 'coords')
        return;
    end
    coords = squeeze(double(payload.coords));
    if isempty(coords) || numel(coords) < 2
        return;
    end
    if size(coords, 2) ~= 2
        if size(coords, 1) == 2
            coords = coords';
        else
            coords = reshape(coords, [], 2);
        end
    end

    coords = round(coords);
    for coord_index = 1:size(coords, 1)
        ix = coords(coord_index, 1);
        iy = coords(coord_index, 2);
        if ix >= 1 && ix <= grid_size(1) && iy >= 1 && iy <= grid_size(2)
            mask(ix, iy) = false;
        end
    end
end


function local_apply_rod_params(model, rod_params)
    num_bars = size(rod_params, 1);
    if size(rod_params, 2) ~= 4 || num_bars < 1
        error('func_outputs_inspect:InvalidRodMatrix', 'rod_params 必须为 num_bars×4。');
    end

    feature_collection = ...
        model.component('comp1').geom('geom1').feature('wp4').geom.feature;
    feature_tags = cell(feature_collection.tags);
    available_indices = [];
    for tag_index = 1:numel(feature_tags)
        token = regexp(char(feature_tags{tag_index}), '^r([0-9]+)$', 'tokens', 'once');
        if ~isempty(token)
            available_indices(end + 1) = str2double(token{1}); %#ok<AGROW>
        end
    end

    missing = setdiff(1:num_bars, available_indices);
    if ~isempty(missing)
        error( ...
            'func_outputs_inspect:MissingRodFeatures', ...
            'COMSOL 模型缺少杆件特征 %s；wp4 当前可用的杆件编号为 %s。', ...
            mat2str(missing), mat2str(sort(available_indices)));
    end

    % 启用需要的杆件，并禁用模型中多余的预建杆件。
    for feature_index = available_indices
        feature_name = sprintf('r%d', feature_index);
        model.component('comp1').geom('geom1').feature('wp4') ...
            .geom.feature(feature_name).active(feature_index <= num_bars);
    end

    for rod_index = 1:num_bars
        feature_name = sprintf('r%d', rod_index);
        xc = rod_params(rod_index, 1);
        yc = rod_params(rod_index, 2);
        theta = rod_params(rod_index, 3);
        width = rod_params(rod_index, 4);

        model.component('comp1').geom('geom1').feature('wp4') ...
            .geom.feature(feature_name).set('pos', {num2str(xc), num2str(yc)});
        model.component('comp1').geom('geom1').feature('wp4') ...
            .geom.feature(feature_name).set('size', {num2str(width), '4000'});
        model.component('comp1').geom('geom1').feature('wp4') ...
            .geom.feature(feature_name).set('rot', num2str(theta));
    end
end
