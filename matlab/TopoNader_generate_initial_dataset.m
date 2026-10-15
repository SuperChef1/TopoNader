function dataset_file = TopoNader_generate_initial_dataset(n, config_file, output_dir)
% Generate designs using func_inputs, evaluate the current model, and save labels.
% Run in a MATLAB session connected to COMSOL. Example:
% TopoNader_generate_initial_dataset(300, ...
%   'D:\Topo_Nader\configs\TopoNader-4bars-30coverage.json', ...
%   'D:\Topo_Nader\data\initial_newload_4bars_cov030');
% n is the number of FE attempts, not a guarantee of n successful samples.
% Existing output directories are rejected. Each FE attempt is checkpointed.

    project_root = fileparts(fileparts(mfilename('fullpath')));
    if nargin < 1 || isempty(n), n = 300; end
    if nargin < 2 || isempty(config_file)
        config_file = fullfile(project_root, 'configs', 'TopoNader-4bars-30coverage.json');
    end
    if nargin < 3 || isempty(output_dir)
        output_dir = fullfile(project_root, 'data', ...
            ['initial_' datestr(now, 'yyyymmdd_HHMMSS')]);
    end
    validateattributes(n, {'numeric'}, {'scalar', 'integer', '>=', 3});
    config_file = char(java.io.File(config_file).getCanonicalPath());
    output_dir = char(java.io.File(output_dir).getCanonicalPath());
    assert(~isfolder(output_dir) && ~isfile(output_dir), ...
        'Output already exists. Choose a new directory; existing data is never overwritten.');
    python_exe = getenv('TOPONADER_PYTHON');
    if isempty(python_exe)
        python_exe = fullfile(project_root, '.venv', 'Scripts', 'python.exe');
    end
    command = sprintf('cd /d "%s" && "%s" -m topo_nader --config "%s" resolved-config', ...
        project_root, python_exe, config_file);
    [status, text] = system(command);
    assert(status == 0, 'Config resolution failed: %s', text);
    config = jsondecode(text);
    o = config.optimization;
    c = o.coverage;
    assert(c.upper_bound_enabled && c.upper_bound > 0, ...
        'This generator requires a positive enabled coverage upper bound.');
    assert(config.data.parameters_per_bar == 4, 'Expected four parameters per bar.');
    source_dir = resolve_path(project_root, config.paths.source_topology_dir);
    model_file = resolve_path(project_root, config.paths.comsol_model);
    assert(isfile(model_file), 'COMSOL model not found: %s', model_file);
    assert(isfile(fullfile(source_dir, 'func_inputs.m')), 'func_inputs.m missing.');
    assert(exist('mphload', 'file') ~= 0, 'Start MATLAB with COMSOL LiveLink first.');
    addpath(source_dir);
    assert(strcmp(which('func_inputs'), fullfile(source_dir, 'func_inputs.m')), ...
        'Another func_inputs.m shadows the configured generator.');
    % Preflight the connection/model before sampling or consuming FE attempts.
    model = mphload(model_file); %#ok<NASGU>
    clear model;
    [coordx, coordy] = ndgrid( ...
        o.grid_x0 - (0:o.grid_nx-1)*o.grid_step, ...
        o.grid_y0 - (0:o.grid_ny-1)*o.grid_step);
    mask = true(size(coordx));
    mask_file = fullfile(source_dir, 'Coord_True_coords.mat');
    if isfile(mask_file)
        s = load(mask_file);
        if isfield(s, 'coords') && ~isempty(s.coords)
            coords = squeeze(double(s.coords));
            if size(coords, 2) ~= 2, coords = coords'; end
            assert(size(coords, 2) == 2, 'Invalid coords mask shape.');
            coords = round(coords);
            for j = 1:size(coords, 1)
                ix = coords(j, 1); iy = coords(j, 2);
                if ix >= 1 && ix <= o.grid_nx && iy >= 1 && iy <= o.grid_ny
                    mask(ix, iy) = false;
                end
            end
        end
    end
    assert(any(mask(:)), 'Design mask is empty.');
    previous_rng = rng;
    restore_rng = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
    rng(double(config.resolved_seeds.matlab_inputs), 'twister');
    num_bars = config.data.num_bars;
    raw_inputs = zeros(n, num_bars, 4);
    coverage_ratios = zeros(n, 1);
    accepted = 0;
    for attempt = 1:n*100
        sample = func_inputs(1, coordx, coordy, c.upper_bound, num_bars);
        design = reshape(sample, num_bars, 4);
        widths = design(:, 4);
        if any(widths < o.rod_width_min | widths > o.rod_width_max), continue; end
        filled = false(size(mask));
        for j = 1:num_bars
            distance = (coordx-design(j,1))*cosd(design(j,3)) + ...
                (coordy-design(j,2))*sind(design(j,3));
            filled = filled | abs(distance) <= widths(j)/2;
        end
        ratio = nnz(filled & mask)/nnz(mask);
        if ratio > c.upper_bound, continue; end
        if c.lower_bound_enabled && ratio < c.lower_bound, continue; end
        if accepted > 0
            prior = reshape(raw_inputs(1:accepted,:,:), accepted, []);
            if any(all(prior == reshape(sample, 1, []), 2)), continue; end
        end
        accepted = accepted + 1;
        raw_inputs(accepted,:,:) = sample;
        coverage_ratios(accepted) = ratio;
        if accepted == n, break; end
    end
    assert(accepted == n, 'Could not generate enough unique feasible designs.');
    mkdir(output_dir);
    % Freeze model and mask so later changes to the source model cannot mix labels.
    frozen_model = fullfile(output_dir, 'initial_model.mph');
    copyfile(model_file, frozen_model);
    if isfile(mask_file), copyfile(mask_file, fullfile(output_dir, 'Coord_True_coords.mat')); end
    provenance = struct('config', config, 'config_file', config_file, ...
        'source_model', model_file, 'frozen_model', frozen_model, ...
        'created_at', datestr(now, 30), 'requested_fe_samples', n);
    raw_outputs = nan(n, 1);
    completed = 0;
    checkpoint_file = fullfile(output_dir, 'initial_checkpoint.mat');
    save(checkpoint_file, 'raw_inputs', 'raw_outputs', 'coverage_ratios', ...
        'completed', 'provenance', '-v7');
    previous_dir = pwd;
    restore_dir = onCleanup(@() cd(previous_dir)); %#ok<NASGU>
    cd(source_dir);
    history_file = fullfile(output_dir, 'initial_FE_history.txt');
    for i = 1:n
        fprintf('Initial FE sample %d / %d\n', i, n);
        raw_outputs(i) = func_outputs_nowrite(raw_inputs(i,:,:), ...
            frozen_model, history_file, num_bars);
        completed = i;
        save(checkpoint_file, 'raw_inputs', 'raw_outputs', 'coverage_ratios', ...
            'completed', 'provenance', '-v7');
    end
    valid_indices = find(isfinite(raw_outputs) & raw_outputs > 0);
    assert(numel(valid_indices) >= 3, ...
        'Fewer than three valid FE samples; inspect checkpoint and FE log.');
    % Preserve generation order; never sort by performance.
    inputs = raw_inputs(valid_indices,:,:);
    outputs = raw_outputs(valid_indices);
    coverage_ratios = coverage_ratios(valid_indices);
    dataset_file = fullfile(output_dir, 'initial_dataset.mat');
    save(dataset_file, 'inputs', 'outputs', 'valid_indices', ...
        'coverage_ratios', 'provenance', '-v7');
    fprintf('Saved %d valid / %d attempted samples: %s\n', ...
        numel(outputs), n, dataset_file);
end

function path = resolve_path(root, configured)
    path = char(configured);
    if isempty(regexp(path, '^[A-Za-z]:[\\/]|^[\\/]', 'once'))
        path = fullfile(root, path);
    end
    path = char(java.io.File(path).getCanonicalPath());
end
