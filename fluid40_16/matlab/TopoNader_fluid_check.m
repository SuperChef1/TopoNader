function TopoNader_fluid_check()
% Lightweight MATLAB contract tests. Does not use COMSOL or train a model.
    root=fileparts(fileparts(mfilename('fullpath')));
    fixture=load(fullfile(root,'tests','legacy_connectivity_fixture.mat'));
    for i=1:size(fixture.inputs,1)
        design=reshape(fixture.inputs(i,:,:),40,16);
        actual=TopoNader_fluid_legacy_isvalid(design');
        assert(actual==logical(fixture.expected(i)), ...
            'Legacy connectivity mismatch for fixture %d.',i);
    end
    cases=[1,1,1;40,1,40;1,2,41;40,16,640;8,6,208];
    for i=1:size(cases,1)
        design=zeros(40,16); design(cases(i,1),cases(i,2))=1;
        assert(isequal(find(design(:)),cases(i,3)), 'Incorrect sq numbering.');
    end
    fprintf('Passed %d legacy connectivity cases and 5 sq mappings.\n',numel(fixture.expected));
end
