function valid = TopoNader_fluid_legacy_isvalid(X)
% Exact binary-grid semantics of reference/utils.py, including its port rules.
% X is 16 rows by 40 columns, top row first. No repair or widened ports.
    assert(isequal(size(X), [16, 40]) && all(X(:)==0 | X(:)==1), ...
        'Expected a binary 16-by-40 grid.');
    inlet = find(X(1:4,1)==0, 1, 'first');
    outlet = find(X(1:8,end)==0, 1, 'first');
    valid = false;
    if isempty(inlet) || isempty(outlet), return; end
    source = sub2ind([16,40], inlet, 1);
    destination = sub2ind([16,40], outlet, 40);
    visited = false(16,40); visited(source) = true;
    queue = zeros(640,1); queue(1) = source; head = 1; tail = 1;
    while head <= tail
        current = queue(head); head = head + 1;
        if current == destination, valid = true; return; end
        [row,col] = ind2sub([16,40],current);
        neighbors = [row,col+1; row,col-1; row+1,col; row-1,col];
        for j=1:4
            rr=neighbors(j,1); cc=neighbors(j,2);
            if rr<1 || rr>16 || cc<1 || cc>40, continue; end
            if X(rr,cc)==1 || visited(rr,cc), continue; end
            visited(rr,cc)=true; tail=tail+1;
            queue(tail)=sub2ind([16,40],rr,cc);
        end
    end
end
