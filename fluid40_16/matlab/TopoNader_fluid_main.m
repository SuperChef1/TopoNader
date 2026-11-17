function TopoNader_fluid_main(initial_dataset, start_iteration, max_iteration, run_dir, config_file)
% MATLAB/COMSOL entry point. No dependency on the web_plate task or shell launcher.
% First run: TopoNader_fluid_main('',1,2,'','.../configs/smoke.json')
% Resume: initial_dataset must be the dataset for start_iteration.
    root=fileparts(fileparts(mfilename('fullpath')));
    if nargin<1 || isempty(initial_dataset), initial_dataset=fullfile(root,'data','initial_641.mat'); end
    if nargin<2 || isempty(start_iteration), start_iteration=1; end
    if nargin<3 || isempty(max_iteration), max_iteration=2; end
    if nargin<4, run_dir=''; end
    if nargin<5 || isempty(config_file), config_file=fullfile(root,'configs','default.json'); end
    validateattributes(start_iteration,{'numeric'},{'scalar','integer','positive'});
    validateattributes(max_iteration,{'numeric'},{'scalar','integer','>=',start_iteration});
    config_file=absolute_path(root,config_file);
    initial_dataset=absolute_path(root,initial_dataset);
    assert(isfile(initial_dataset),'Initial dataset not found.');
    assert(exist('mphload','file')~=0,'Start MATLAB with COMSOL LiveLink before optimization.');
    python=getenv('TOPONADER_PYTHON');
    if isempty(python), python=fullfile(fileparts(root),'.venv','Scripts','python.exe'); end
    assert(isfile(python),'Python missing; set TOPONADER_PYTHON.');
    % Reject path shadowing before training or FE calls.
    addpath(fileparts(mfilename('fullpath')),'-begin');
    cmd=sprintf('cd /d "%s" && "%s" -B -c "import topo_nader; print(topo_nader.__file__)"',root,python);
    [code,text]=system(cmd);
    assert(code==0 && contains(lower(strtrim(text)),lower(fullfile(root,'topo_nader'))), ...
        'Wrong Python task import: %s',text);
    config=jsondecode(command(root,python,config_file,'resolved-config'));
    if isempty(run_dir), run_dir=fullfile(root,config.paths.runs_dir,config.experiment_name); end
    run_dir=absolute_path(root,run_dir);
    command(root,python,config_file,sprintf('init --run-dir "%s"',run_dir));
    model_file=fullfile(run_dir,'runtime','fluid.mph');
    dataset_file=initial_dataset;
    fprintf('\n%s\n',repmat('=',1,88));
    fprintf('TopoNader fluid run\n');
    fprintf('Experiment : %s\n',config.experiment_name);
    fprintf('Run folder : %s\n',run_dir);
    fprintf('Iterations : %d through %d\n',start_iteration,max_iteration);
    fprintf('%s\n',repmat('=',1,88));
    for iteration=start_iteration:max_iteration
        folder=fullfile(run_dir,sprintf('iteration_%04d',iteration));
        request_file=fullfile(folder,'fe_request.mat');
        result_file=fullfile(folder,'fe_result.mat');
        next_dataset=fullfile(run_dir,'datasets',sprintf('dataset_%04d.mat',iteration+1));
        state=jsondecode(fileread(fullfile(run_dir,'state.json')));
        assert(state.next_iteration==iteration, ...
            'State expects iteration %d; use its matching dataset and start_iteration.',state.next_iteration);
        fprintf('\n%s\n',repmat('-',1,88));
        fprintf('STARTING ITERATION %d | completed rounds before this iteration: %d\n', ...
            iteration,iteration-1);
        fprintf('Dataset: %s\n',dataset_file);
        fprintf('%s\n',repmat('-',1,88));
        if ~isfile(request_file)
            command(root,python,config_file,sprintf( ...
                'prepare --run-dir "%s" --dataset "%s" --iteration %d',run_dir,dataset_file,iteration));
        end
        if ~isfile(result_file)
            assert(exist('mphload','file')~=0,'Start MATLAB with COMSOL LiveLink.');
            TopoNader_fluid_fe_batch(request_file,model_file,result_file);
        end
        command(root,python,config_file,sprintf( ...
            'feedback --run-dir "%s" --dataset "%s" --iteration %d --fe-result "%s" --next-dataset "%s"', ...
            run_dir,dataset_file,iteration,result_file,next_dataset));
        print_iteration_results(run_dir,result_file,iteration);
        dataset_file=next_dataset;
    end
end

function print_iteration_results(run_dir,result_file,iteration)
    fe=load(result_file);
    n=size(fe.inputs,1);
    sample=(1:n)';
    status=string(fe.status(:));
    objective=double(fe.outputs(:));
    inlet_pressure=double(fe.inlet_pressure(:));
    elapsed_seconds=double(fe.elapsed_seconds(:));
    solid_fraction=mean(reshape(double(fe.inputs),n,[]),2);
    results=table(sample,status,objective,inlet_pressure,solid_fraction,elapsed_seconds, ...
        'VariableNames',{'Sample','Status','Objective','InletPressure', ...
        'SolidFraction','ElapsedSeconds'});

    fprintf('\nRESULTS FOR ITERATION %d\n',iteration);
    disp(results);
    success_count=sum(status=="success");
    fprintf('Successful FE results: %d/%d\n',success_count,n);
    valid=isfinite(objective);
    if any(valid)
        fprintf('Iteration best objective: %.12g\n',max(objective(valid)));
        fprintf('Iteration best inlet pressure: %.12g\n',min(inlet_pressure(valid)));
    else
        fprintf('Iteration produced no finite objective values.\n');
    end

    state=jsondecode(fileread(fullfile(run_dir,'state.json')));
    summary=jsondecode(fileread(fullfile(run_dir,'summary.json')));
    fprintf('\nCOMPLETED ITERATION %d\n',iteration);
    fprintf('Total completed rounds: %d\n',summary.iterations_completed);
    fprintf('Next iteration: %d\n',state.next_iteration);
    fprintf('Cumulative valid new FE results: %d/%d\n', ...
        summary.cumulative_valid_fe_results,summary.cumulative_new_fe_calls);
    fprintf('Cumulative best objective: %.12g\n',summary.best_performance);
    fprintf('Cumulative best inlet pressure: %.12g\n',summary.best_inlet_pressure);
    fprintf('Architecture audits/changes/rollbacks: %d/%d/%d\n', ...
        summary.audits_performed,summary.architecture_changes,summary.rollbacks);
    fprintf('%s\n',repmat('=',1,88));
end

function output=command(root,python,config,args)
    cmd=sprintf('cd /d "%s" && "%s" -B -m topo_nader --config "%s" %s',root,python,config,args);
    [code,output]=system(cmd);
    assert(code==0,'TopoNader command failed:\n%s',output);
end

function p=absolute_path(root,p)
    if isempty(regexp(p,'^[A-Za-z]:[\\/]|^[\\/]','once')), p=fullfile(root,p); end
    p=char(java.io.File(p).getCanonicalPath());
end
