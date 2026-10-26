function [outputs, diagnostics] = TopoNader_fluid_fe_batch(request_file, model_file, result_file)
% Evaluate the exact N-by-40-by-16 binary request. Never overwrite FE results.
    request=load(request_file);
    assert(isfield(request,'inputs'), 'Missing request inputs.');
    inputs=double(request.inputs);
    assert(size(inputs,2)==40 && size(inputs,3)==16 && ...
        all(inputs(:)==0 | inputs(:)==1), 'Expected binary N-by-40-by-16 inputs.');
    assert(~isfile(result_file), 'Result already exists; refusing to overwrite.');
    assert(isfile(model_file), 'Missing COMSOL model.');
    n=size(inputs,1); outputs=nan(n,1);
    inlet_pressure=nan(n,1); elapsed_seconds=zeros(n,1);
    status=repmat({'pending'},n,1); errors=repmat({''},n,1);
    % Native MATLAB checker retains the exact original Python rule.
    feasible=false(n,1);
    for i=1:n
        design=reshape(inputs(i,:,:),40,16);
        feasible(i)=TopoNader_fluid_legacy_isvalid(design');
    end
    model=mphload(model_file);
    geom=model.component('comp1').geom('geom1');
    squares=arrayfun(@(j) sprintf('sq%d',j),1:640,'UniformOutput',false);
    checkpoint_file=[result_file '.checkpoint.mat'];
    first=1;
    if isfile(checkpoint_file)
        previous=load(checkpoint_file);
        assert(isequal(previous.inputs,inputs) && strcmp(previous.model_file,model_file), ...
            'Checkpoint does not match this request/model.');
        outputs=previous.outputs; inlet_pressure=previous.inlet_pressure;
        status=previous.status; errors=previous.errors;
        elapsed_seconds=previous.elapsed_seconds;
        first=previous.completed+1;
    end
    model_tag=char(model.tag());
    cleanup=onCleanup(@() com.comsol.model.util.ModelUtil.remove(model_tag));
    for i=first:n
        timer=tic;
        if ~feasible(i)
            status{i}='legacy_connectivity_rejected';
        else
            try
                design=reshape(inputs(i,:,:),40,16);
                % MATLAB column-major flattening is the original sq numbering.
                solid=design(:)'==1;
                geom.feature('dif1').selection('input2').set([{'r2'},squares(solid)]);
                geom.run;
                % Rebuild the mesh for the new geometry before solving.
                model.component('comp1').mesh('mesh1').run;
                % Obtain the current inlet/outlet from physics, never assume edge 1.
                inlet_sel=mphgetselection(model.component('comp1').physics('spf').feature('inl1').selection);
                outlet_sel=mphgetselection(model.component('comp1').physics('spf').feature('out1').selection);
                inlet_ids=double(inlet_sel.entities); outlet_ids=double(outlet_sel.entities);
                assert(~isempty(inlet_ids) && ~isempty(outlet_ids), 'Empty inlet/outlet selection.');
                in_xyz=mphgetcoords(model,'geom1','boundary',inlet_ids);
                out_xyz=mphgetcoords(model,'geom1','boundary',outlet_ids);
                assert(all(abs(in_xyz(1,:)+0.4)<1e-7), 'Inlet selection moved from x=-0.4.');
                assert(all(abs(out_xyz(1,:)-2.4)<1e-7), 'Outlet selection moved from x=2.4.');
                model.sol('sol1').runAll;
                pressure=double(mphmean(model,'p','line','selection',inlet_ids));
                assert(isscalar(pressure) && isfinite(pressure) && pressure>0, 'Invalid inlet pressure.');
                inlet_pressure(i)=pressure; outputs(i)=1/pressure;
                status{i}='success';
            catch exception
                status{i}='calculation_failed';
                errors{i}=getReport(exception,'extended','hyperlinks','off');
            end
        end
        elapsed_seconds(i)=toc(timer);
        completed=i;
        save(checkpoint_file,'inputs','outputs','inlet_pressure','status','errors', ...
            'elapsed_seconds','completed','model_file','-v7');
        fprintf('Fluid FE %d/%d: %s\n',i,n,status{i});
    end
    diagnostics=struct('inlet_pressure',inlet_pressure,'status',{status}, ...
        'errors',{errors},'elapsed_seconds',elapsed_seconds);
    connectivity_mode='legacy_unchanged';
    save(result_file,'inputs','outputs','inlet_pressure','status','errors', ...
        'elapsed_seconds','model_file','connectivity_mode','-v7');
end
