function report = TopoNader_fluid_validate_reference(output_dir, sample_indices, relative_tolerance)
% Explicit COMSOL validation, separate from optimization. New output dir required.
% MATLAB indices into data/initial_641.mat. Never relabel the original data.
    root=fileparts(fileparts(mfilename('fullpath')));
    if nargin<2 || isempty(sample_indices), sample_indices=[1,2,41,240,641]; end
    if nargin<3 || isempty(relative_tolerance), relative_tolerance=0.05; end
    assert(nargin>=1 && ~isempty(output_dir), 'Choose a new validation output directory.');
    output_dir=char(java.io.File(output_dir).getCanonicalPath());
    assert(~isfolder(output_dir) && ~isfile(output_dir),'Output directory already exists.');
    validateattributes(relative_tolerance,{'numeric'},{'scalar','finite','nonnegative'});
    data=load(fullfile(root,'data','initial_641.mat'));
    validateattributes(sample_indices,{'numeric'},{'vector','integer','positive','<=',size(data.inputs,1)});
    mkdir(output_dir);
    inputs=data.inputs(sample_indices,:,:);
    expected=data.outputs(sample_indices);
    request=fullfile(output_dir,'fe_request.mat');
    save(request,'inputs','sample_indices','expected','-v7');
    model_file=fullfile(output_dir,'fluid.mph');
    copyfile(fullfile(root,'matlab','fluid.mph'),model_file);
    outputs=TopoNader_fluid_fe_batch(request,model_file,fullfile(output_dir,'fe_result.mat'));
    relative_error=abs(outputs-expected)./abs(expected);
    passed=all(isfinite(outputs) & relative_error<=relative_tolerance);
    report=struct('sample_indices',sample_indices,'expected',expected,'outputs',outputs, ...
        'relative_error',relative_error,'relative_tolerance',relative_tolerance,'passed',passed);
    save(fullfile(output_dir,'comparison.mat'),'report');
    disp(report);
    assert(passed,'Reference validation failed. Inspect results before starting optimization.');
end
