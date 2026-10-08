function TopoNader_main(initial_dataset, start_iteration, max_iteration, run_dir, config_file)
%TOPONADER_MAIN MATLAB/COMSOL outer loop for the new single-channel project.
%
% Example:
%   TopoNader_main('D:\data\initial.mat', 1, 20, '', ...
%       'D:\Nader\Topo_Nader\configs\default.json')

    arguments
        initial_dataset (1,:) char
        start_iteration (1,1) double {mustBeInteger,mustBePositive} = 1
        max_iteration (1,1) double {mustBeInteger,mustBePositive} = 20
        run_dir (1,:) char = ''
        config_file (1,:) char = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'configs', 'default.json')
    end
    project_root = fileparts(fileparts(mfilename('fullpath')));
    python_exe = getenv('TOPONADER_PYTHON');
    if isempty(python_exe)
        python_exe = 'python';
    end
    resolved_config_json = run_command( ...
        project_root, python_exe, config_file, 'resolved-config', false);
    config = jsondecode(resolved_config_json);
    if isempty(run_dir)
        run_dir = fullfile(project_root, config.paths.runs_dir, config.experiment_name);
    end
    rng(double(config.resolved_seeds.matlab_inputs), 'twister');
    source_topology_dir = resolve_project_path( ...
        project_root, config.paths.source_topology_dir);
    comsol_model = resolve_project_path( ...
        project_root, config.paths.comsol_model);
    run_command(project_root, python_exe, config_file, sprintf('init --run-dir "%s"', run_dir));
    history_file = fullfile(run_dir, 'logs', 'beam_calculation_history.txt');
    dataset_file = initial_dataset;
    for iteration = start_iteration:max_iteration
        iteration_dir = fullfile(run_dir, sprintf('iteration_%04d', iteration));
        run_command(project_root, python_exe, config_file, sprintf( ...
            'prepare --run-dir "%s" --dataset "%s" --iteration %d', run_dir, dataset_file, iteration));
        request_file = fullfile(iteration_dir, 'fe_request.mat');
        outputs = TopoNader_fe_batch( ...
            request_file, source_topology_dir, comsol_model, history_file, config.data.num_bars);
        result_file = fullfile(iteration_dir, 'fe_result.mat');
        save(result_file, 'outputs');
        next_dataset = fullfile(run_dir, 'datasets', sprintf('dataset_%04d.mat', iteration + 1));
        if exist(fileparts(next_dataset), 'dir') ~= 7
            mkdir(fileparts(next_dataset));
        end
        run_command(project_root, python_exe, config_file, sprintf( ...
            'feedback --run-dir "%s" --dataset "%s" --iteration %d --fe-result "%s" --next-dataset "%s"', ...
            run_dir, dataset_file, iteration, result_file, next_dataset));
        dataset_file = next_dataset;
    end
end

function output = run_command(project_root, python_exe, config_file, arguments_text, print_output)
    if nargin < 5
        print_output = true;
    end
    command = sprintf('cd /d "%s" && "%s" -m topo_nader --config "%s" %s', ...
        project_root, python_exe, config_file, arguments_text);
    [status, output] = system(command);
    if print_output
        fprintf('%s', output);
    end
    if status ~= 0
        error('TopoNader_main:PythonFailed', 'TopoNader command failed with status %d.', status);
    end
end

function resolved_path = resolve_project_path(project_root, configured_path)
    if ispc && ~isempty(regexp(configured_path, '^[A-Za-z]:[\\/]', 'once'))
        resolved_path = configured_path;
    elseif startsWith(configured_path, filesep)
        resolved_path = configured_path;
    else
        resolved_path = fullfile(project_root, configured_path);
    end
end
