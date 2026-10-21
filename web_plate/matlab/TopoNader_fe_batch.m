function outputs = TopoNader_fe_batch(fe_request_file, source_topology_dir, comsol_model, history_file, expected_num_bars)
%TOPONADER_FE_BATCH Evaluate a TopoNader request with the original COMSOL oracle.
% The adapter runs from the configured source directory and passes the configured
% COMSOL model path to func_outputs_nowrite.

    request = load(fe_request_file, 'inputs', 'num_bars');
    if ~isfield(request, 'inputs')
        error('TopoNader_fe_batch:MissingInputs', 'fe_request.mat does not contain inputs.');
    end
    source_topology_dir = char(java.io.File(source_topology_dir).getCanonicalPath());
    if nargin < 3 || isempty(comsol_model)
        comsol_model = fullfile(source_topology_dir, 'fuban_Beam.mph');
    end
    if nargin < 4
        history_file = '';
    end
    if nargin < 5
        expected_num_bars = [];
    end
    if isempty(expected_num_bars) && isfield(request, 'num_bars')
        expected_num_bars = double(request.num_bars(1));
    end
    comsol_model = char(java.io.File(comsol_model).getCanonicalPath());
    if exist(fullfile(source_topology_dir, 'func_outputs_nowrite.m'), 'file') ~= 2
        error('TopoNader_fe_batch:MissingOracle', 'Missing func_outputs_nowrite.m in %s', source_topology_dir);
    end
    if exist(comsol_model, 'file') ~= 2
        error('TopoNader_fe_batch:MissingModel', 'Missing COMSOL model: %s', comsol_model);
    end
    previous_dir = pwd;
    cleanup = onCleanup(@() cd(previous_dir));
    addpath(source_topology_dir);
    cd(source_topology_dir);
    outputs = func_outputs_nowrite(request.inputs, comsol_model, history_file, expected_num_bars);
end
