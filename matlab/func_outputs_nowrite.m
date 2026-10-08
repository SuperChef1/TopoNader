function outputs = func_outputs_nowrite(inputs_rod, varargin)
    %FUNC_OUTPUTS_NOWRITE  梁模型批量求解：输入为杆件参数，几何更新方式与 MphTest.m 一致。
    %
    % 输入
    %   inputs_rod — iter×4×4 double（或 iter×16），每样本四根杆，每行 [xc, yc, theta_deg, w]。
    %   可选第二个参数 — COMSOL .mph 模型路径。其他旧式参数仍会被忽略。
    %
    % 输出
    %   outputs — iter×1，成功为标量目标（与 av1/av2 导出一致）；失败为 Inf。
    %
    % 记录
    %   不保存 .mph / 逐例 .mat；仅在命令行打印成功信息，并在与本 .m 同目录的
    %   beam_calculation_history.txt 中追加每次算例（时间、序号、可粘贴至 MphTest 的 rod_params 矩阵、指标；失败亦记录）。

    model_rel_path = 'fuban_Beam.mph';
    if ~isempty(varargin)
        configured_model = varargin{1};
        if (ischar(configured_model) || isstring(configured_model)) && endsWith(string(configured_model), ".mph", 'IgnoreCase', true)
            model_rel_path = char(configured_model);
        end
    end
    if exist(model_rel_path, 'file') ~= 2
        error('func_outputs_nowrite: 未找到模型文件 %s（请放在 MATLAB 当前工作目录）。', model_rel_path);
    end

    persistent case_counter history_seq;
    if isempty(case_counter)
        case_counter = 0;
    end
    if isempty(history_seq)
        history_seq = 0;
    end

    inputs_rod = double(inputs_rod);
    if size(inputs_rod, 2) == 16
        iter = size(inputs_rod, 1);
        inputs_rod = reshape(inputs_rod, iter, 4, 4);
    elseif ndims(inputs_rod) == 3 && size(inputs_rod, 2) == 4 && size(inputs_rod, 3) == 4
        iter = size(inputs_rod, 1);
    else
        error('func_outputs_nowrite: inputs_rod 须为 iter×4×4 或 iter×16。');
    end

    outputs = Inf(iter, 1);

    for i = 1:iter
        rod_i = squeeze(inputs_rod(i, :, :));
        if ~isequal(size(rod_i), [4, 4])
            rod_i = reshape(rod_i, 4, 4);
        end

        history_seq = history_seq + 1; % 每次求解尝试唯一序号（成功/失败均递增，便于日志追加）

        try
            model = mphload(model_rel_path);
            model.disableUpdates(true);

            local_apply_rod_params_mptest(model, rod_i);

            model.component('comp1').geom('geom1').run;
            model.component('comp1').mesh('mesh1').run;
            model.sol('sol1').runAll;

            disp1 = model.result.numerical('av1').getReal;
            disp2 = model.result.numerical('av2').getReal;

            raw_value = disp1 + disp2;
            current_output = 0.1 / raw_value;

            outputs(i) = current_output;

            case_counter = case_counter + 1; % 成功序号，仅用于日志中 success_idx

            fprintf(1, 'calculation success: disp1 (av1) = %g, disp2 (av2) = %g, current_output = %g\n', ...
                disp1, disp2, current_output);
            local_append_beam_history(rod_i, current_output, history_seq, case_counter, true, '');

        catch ME
            outputs(i) = Inf;
            disp([num2str(i), ': beam calculation failed']);
            disp(ME.message);
            local_append_beam_history(rod_i, Inf, history_seq, NaN, false, ME.message);
        end
    end
end

function local_append_beam_history(rod_i, metric, history_id, success_idx, ok, errMsg)
% 将单次计算的 4×4 与指标追加写入与本函数文件同目录的 beam_calculation_history.txt（多次调用持续追加）
    logPath = fullfile(fileparts(mfilename('fullpath')), 'beam_calculation_history.txt');
    fid = fopen(logPath, 'a');
    if fid < 0
        warning('func_outputs_nowrite: 无法打开日志文件进行追加: %s', logPath);
        return;
    end
    cleaner = onCleanup(@() fclose(fid));
    if ok
        statStr = 'SUCCESS';
        idxStr = sprintf('%06d', success_idx);
    else
        statStr = 'FAILED';
        idxStr = 'N/A';
    end
    fprintf(fid, '===== history_%05d | success_idx_%s | %s | %s =====\n', history_id, idxStr, ...
        datestr(now, 'yyyy-mm-dd HH:MM:SS'), statStr);
    fprintf(fid, 'output_metric=%.17g\n', metric);
    % 与 MphTest.m 中 rod_params 写法一致，整段可粘贴覆盖该文件里的矩阵赋值
    fprintf(fid, '%% [xc, yc, theta_deg, w] per row — paste into MphTest as rod_params:\n');
    fprintf(fid, 'rod_params = [\n');
    for row = 1:4
        if row < 4
            fprintf(fid, ' %.15g, %.15g, %.15g, %.15g;\n', ...
                rod_i(row, 1), rod_i(row, 2), rod_i(row, 3), rod_i(row, 4));
        else
            fprintf(fid, ' %.15g, %.15g, %.15g, %.15g\n', ...
                rod_i(row, 1), rod_i(row, 2), rod_i(row, 3), rod_i(row, 4));
        end
    end
    fprintf(fid, '];\n');
    if ~ok
        em = char(errMsg);
        if ~isempty(em)
            fprintf(fid, 'error_message: %s\n', em);
        end
    end
    fprintf(fid, '\n');
end

function local_apply_rod_params_mptest(model, rod_params)
% 与 MphTest.m 相同：wp4 下 r1~r4 的 pos / size / rot
    if ~isequal(size(rod_params), [4, 4])
        error('rod_params 须为 4×4。');
    end
    for j = 1:4
        feat_name = sprintf('r%d', j);
        xc    = rod_params(j, 1);
        yc    = rod_params(j, 2);
        theta = rod_params(j, 3);
        w     = rod_params(j, 4);

        model.component('comp1').geom('geom1').feature('wp4') ...
            .geom.feature(feat_name).set('pos', {num2str(xc), num2str(yc)});

        model.component('comp1').geom('geom1').feature('wp4') ...
            .geom.feature(feat_name).set('size', {num2str(w), '4000'});

        model.component('comp1').geom('geom1').feature('wp4') ...
            .geom.feature(feat_name).set('rot', num2str(theta));
    end
end
