function [results, models] = func_outputs_extremes(model_file)
%FUNC_OUTPUTS_EXTREMES 计算无杆件（全空）和设计域全填满的参考结果。
%
% 用法：
%   [results, models] = func_outputs_extremes();
%
%   results.empty.output
%   results.full.output
%   results.empty.coverage_ratio   % 0
%   results.full.coverage_ratio    % 1
%
% 可选模型路径：
%   [results, models] = func_outputs_extremes('D:\path\fuban_Beam.mph');
%
% 本函数每种情况都重新加载原始 MPH，不保存模型，也不写计算历史。
%
% 几何含义（已按当前 fuban_Beam.mph 的特征依赖实现）：
%   - dif4 是不含可设计杆件的主体；
%   - wp4/int1 是设计域 csol1 与杆件并集 uni1 的交集；
%   - ext3 将 wp4 的结果拉伸，再由顶层 uni1 与 dif4 合并。
%   - 全空：停用 wp4/ext3，顶层 uni1 只接收 dif4；
%   - 全满：用 r1 完全覆盖 csol1，使 int1 的结果等于完整设计域。
%
% 查看几何：在下方 "EXTREME INSPECTION BREAKPOINT" 处设置断点。
% 第一次停下是 empty，第二次是 full；在 K>> 提示符执行：
%
%   case_name
%   mphlaunch(model, 10000)
%
% 查看后执行 dbcont 继续求解。

    if nargin < 1 || isempty(model_file)
        model_file = fullfile(fileparts(mfilename('fullpath')), 'fuban_Beam.mph');
    end
    model_file = char(model_file);
    if exist(model_file, 'file') ~= 2
        error( ...
            'func_outputs_extremes:MissingModel', ...
            '未找到 COMSOL 模型: %s', model_file);
    end

    [results.empty, models.empty] = local_run_extreme('empty', model_file);
    [results.full, models.full] = local_run_extreme('full', model_file);

    fprintf(1, '\n===== extreme reference results =====\n');
    local_print_result(results.empty);
    local_print_result(results.full);
end


function [result, model] = local_run_extreme(case_name, model_file)
    result = struct( ...
        'case_name', case_name, ...
        'success', false, ...
        'coverage_ratio', NaN, ...
        'disp1', NaN, ...
        'disp2', NaN, ...
        'output', NaN, ...
        'error', '');

    model = mphload(model_file);
    model.disableUpdates(true);

    try
        switch case_name
            case 'empty'
                local_configure_empty(model);
                result.coverage_ratio = 0.0;
            case 'full'
                local_configure_full(model);
                result.coverage_ratio = 1.0;
            otherwise
                error( ...
                    'func_outputs_extremes:UnknownCase', ...
                    '未知极限模型类型: %s', case_name);
        end

        model.component('comp1').geom('geom1').run;
        model.component('comp1').mesh('mesh1').run;
        model.disableUpdates(false);

        extreme_inspection_breakpoint = true; %#ok<NASGU> % <-- EXTREME INSPECTION BREAKPOINT

        model.sol('sol1').runAll;
        result.disp1 = model.result.numerical('av1').getReal;
        result.disp2 = model.result.numerical('av2').getReal;
        result.output = 0.1 / (result.disp1 + result.disp2);
        result.success = true;
    catch ME
        try
            model.disableUpdates(false);
        catch
            % 保留原始 COMSOL 错误，不用界面更新错误覆盖它。
        end
        result.error = getReport(ME, 'extended', 'hyperlinks', 'off');
        warning( ...
            'func_outputs_extremes:CaseFailed', ...
            '%s 极限模型计算失败:\n%s', case_name, result.error);
    end
end


function local_configure_empty(model)
% 精确的“无杆件”模型：移除杆件生成支路，而不是创建零宽矩形。
    geometry = model.component('comp1').geom('geom1');
    local_require_top_level_features(geometry, {'wp4', 'ext3', 'dif4', 'uni1'});

    geometry.feature('wp4').active(false);
    geometry.feature('ext3').active(false);

    % 顶层 uni1 原本合并 dif4 和 ext3。空模型只保留主体 dif4。
    geometry.feature('uni1').selection('input').set({'dif4'});
end


function local_configure_full(model)
% 精确的“设计域全满”模型：大矩形与 csol1 相交后恰好得到 csol1。
    geometry = model.component('comp1').geom('geom1');
    local_require_top_level_features(geometry, {'wp4', 'ext3', 'dif4', 'uni1'});

    geometry.feature('wp4').active(true);
    geometry.feature('ext3').active(true);
    work_plane_geometry = geometry.feature('wp4').geom;
    local_require_work_plane_features( ...
        work_plane_geometry, {'csol1', 'r1', 'uni1', 'int1'});

    feature_tags = cell(work_plane_geometry.feature.tags);
    rod_indices = [];
    for tag_index = 1:numel(feature_tags)
        token = regexp(char(feature_tags{tag_index}), '^r([0-9]+)$', 'tokens', 'once');
        if ~isempty(token)
            rod_indices(end + 1) = str2double(token{1}); %#ok<AGROW>
        end
    end
    if ~ismember(1, rod_indices)
        error( ...
            'func_outputs_extremes:MissingRodFeature', ...
            'wp4 中缺少用于构造全满模型的 r1。');
    end

    for rod_index = rod_indices
        work_plane_geometry.feature(sprintf('r%d', rod_index)).active(rod_index == 1);
    end

    % 当前设计区约为 x=[-550,0], y=[-450,450]。以中心为基准的
    % 4000×4000 矩形远大于该区域；int1 会把它严格裁剪为 csol1。
    work_plane_geometry.feature('r1').set('base', 'center');
    work_plane_geometry.feature('r1').set('pos', {'-275', '0'});
    work_plane_geometry.feature('r1').set('size', {'4000', '4000'});
    work_plane_geometry.feature('r1').set('rot', '0');

    work_plane_geometry.feature('uni1').active(true);
    work_plane_geometry.feature('uni1').selection('input').set({'r1'});
    work_plane_geometry.feature('int1').active(true);
    work_plane_geometry.feature('int1').selection('input').set({'csol1', 'uni1'});
    geometry.feature('uni1').selection('input').set({'dif4', 'ext3'});
end


function local_require_top_level_features(geometry, required_tags)
    available_tags = cell(geometry.feature.tags);
    local_require_features(available_tags, required_tags, 'geom1');
end


function local_require_work_plane_features(work_plane_geometry, required_tags)
    available_tags = cell(work_plane_geometry.feature.tags);
    local_require_features(available_tags, required_tags, 'wp4');
end


function local_require_features(available_tags, required_tags, scope_name)
    available_tags = cellfun(@char, available_tags, 'UniformOutput', false);
    missing_tags = setdiff(required_tags, available_tags);
    if ~isempty(missing_tags)
        error( ...
            'func_outputs_extremes:ModelStructureMismatch', ...
            '%s 缺少所需特征 %s；当前特征为 %s。', ...
            scope_name, strjoin(missing_tags, ', '), strjoin(available_tags, ', '));
    end
end


function local_print_result(result)
    if result.success
        fprintf(1, ...
            '%s: disp1=%g, disp2=%g, output=%g, material_coverage=%g (%.2f%%)\n', ...
            result.case_name, result.disp1, result.disp2, result.output, ...
            result.coverage_ratio, 100 * result.coverage_ratio);
    else
        fprintf(1, ...
            '%s: FAILED, output=NaN, material_coverage=%g (%.2f%%)\n', ...
            result.case_name, result.coverage_ratio, 100 * result.coverage_ratio);
    end
end
