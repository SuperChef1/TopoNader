clc;
clear all;

latentfile = '10ntrain_train_step.mat';
latentdataC=load(latentfile);
% imshow(squeeze(latentdata.inputs(17,:,:)));
folder='data_g'; % name of data folder
mkdir(folder);

folder_latent='data_latent_g';
mkdir(folder_latent);

folder_opt='opt_g';
mkdir(folder_opt);

folder_sample = 'sample';
mkdir(folder_sample);

layers = [
    featureInputLayer(640,"Name","featureinput");
    fullyConnectedLayer(1024,"Name","fc")
    batchNormalizationLayer("Name","batchnorm")
    leakyReluLayer(0.2,"Name","leakyrelu")
    fullyConnectedLayer(512,"Name","fc_1")
    dropoutLayer(0.4,"Name","dropout")
    batchNormalizationLayer("Name","batchnorm_1")
    leakyReluLayer(0.2,"Name","leakyrelu_1")
    fullyConnectedLayer(256,"Name","fc_2")
    batchNormalizationLayer("Name","batchnorm_2")
    dropoutLayer(0.4,"Name","dropout_1")
    leakyReluLayer(0.2,"Name","leakyrelu_2")
    fullyConnectedLayer(1,"Name","fc_3")
    regressionLayer("Name","regressionoutput")];
optionss = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs',1000, ...
    'MiniBatchSize', 1024,...
    'Verbose',false, ...
    'Plots','none'...
    );
n=10;
PSO_inputs=[];
PSO_outputs=[];
AE_inputs=[];
AE_outputs=[];
PSO_num=[];
AE_num=[];
best_PSO=[];
best_AE=[];
outputss=[];
for i=1:1000
    tic
    trainfile=['./',folder,'/',num2str(n),'ntrain_train_step',num2str(i),'.mat'];
    latentfile=['./',folder_latent,'/',num2str(n),'ntrain_train_step',num2str(i),'.mat'];
    optfile=['./',folder_opt,'/',num2str(n),'nopt_step',num2str(i),'.mat'];
    PSOfile=['./',folder_sample,'/',num2str(n),'nPSO_step',num2str(i),'.mat'];
    AEfile=['./',folder_sample,'/',num2str(n),'nAE_step',num2str(i),'.mat'];
    % run python, CHANGE the following to your python executable directory 
    cmdstr = sprintf(['call D:/ProgramFiles/Anaconda3/Scripts/activate pytorch & ', ...
                      'C:/Users/Administrator/.conda/envs/to/python.exe ', ...
                  '-c "import WAE_func_edit; WAE_func_edit.func(''%s'',''%s'',topk=%d)"'], ...
                  trainfile, latentfile, n);
    [status,commandOut]=system(cmdstr); % run mlopt.py
    latentdata = load(latentfile);
    if i>1
        aaaa=unique(reshape(latentdata.base,size(latentdata.base,1),640),'rows');
        [~, ia, ib] = intersect(reshape(inputsX,m,640), aaaa, 'rows');
        aaaa(ib, :) = [];
        % aaaa=[];
        if ~isempty(aaaa)
            outputsC=func_outputs_nowrite(aaaa,40,16);
            aaaa(outputsC==Inf,:)=[];
            outputsC(outputsC==Inf)=[];
            % aaaa(outputsC<=1.5288,:)=[];
            % outputsC(outputsC<=1.5288)=[];
        else
            outputsC=[];
        end
        inputss=[reshape(inputsX,m,640);aaaa];
        outputss=[ outputsX;outputsC];
    else
        aaaa=unique(reshape(latentdata.base,size(latentdata.base,1),640),'rows');
        [~, ia, ib] = intersect(reshape(latentdataC.inputs,size(latentdataC.inputs,1),640), aaaa, 'rows');
        aaaa(ib, :) = [];
        % aaaa=[];
        if ~isempty(aaaa)
            outputsC=func_outputs_nowrite(aaaa,40,16);
            aaaa(outputsC==Inf,:)=[];
            outputsC(outputsC==Inf)=[];
        else
            outputsC=[];
        end
        inputss=[reshape(latentdataC.inputs,size(latentdataC.inputs,1),640);aaaa];
        outputss=[latentdataC.outputs; outputsC];
    end
    AE_inputs = [AE_inputs;aaaa];
    AE_outputs = [AE_outputs;outputsC];
    AE_num = [AE_num;i*ones(size(outputsC,1),1)];
    if outputsC
        fprintf('python: [%s]',strjoin(string(outputsC),','));
    end
    
    % yreal=predict(net,latentdata.latent);
    
    lb=zeros(1,640);
    ub=ones(1,640);
    [maxo,b]=sort(outputss,'descend');
    maxo(1)
    info=[' ans= ',num2str(maxo(1)),'; iter= ',num2str(i),'; num_of_sample= ',num2str(size(inputss,1))];
    disp(info);
    zui(i)=maxo(1);
    best_AE=[best_AE;zui(i)*ones(size(outputsC,1),1)];
    popp=inputss(b,:);
    pop=[popp(1:100,:);randi([0, 1], 200, 640)];

    %GA
    % options = optimoptions(@ga, 'PopulationSize', 100, 'PopulationType', 'bitString','InitialPopulationMatrix', initialPopulation(inputss, outputss));
    % [x,fval,exitflag,output,population,scores] = ga(@(x) fitnessFunctionWithPenalty(net,x),640,[],[],[],[],lb,ub,[],[],options);
    % [a,b]=sort(scores);
    % xnew=population(b,:);

    %PSO
    % net = trainNetwork(inputss, outputss, layers, optionss);
    % options = optimoptions('particleswarm', 'MaxIterations', 2000, 'InitialPoints',pop,'FunctionTolerance', 1e-88,'SwarmSize', 300);
    % [x,fval,exitflag,output,points] = particleswarm(@(x)fitnessFunctionWithPenalty(net,x), 640, lb, ub, options);
    % [a,b]=sort(points.Fval);
    % xnew=points.X(b,:);   
    % xnew=unique(xnew,'row','stable');
    % cmdstr1 = sprintf(['call D:/ProgramFiles/Anaconda3/Scripts/activate pytorch & ', ...
    %                   'C:/Users/Administrator/.conda/envs/to/python.exe ', ...
    %                   '-c "import BPSO1; BPSO1.func(''%s'',''%s'')"'], ...
    %                   trainfile, optfile);
    cmdstr1 = sprintf(['call D:/ProgramFiles/Anaconda3/Scripts/activate pytorch & ', ...
                      'C:/Users/Administrator/.conda/envs/to/python.exe ', ...
                      '-c "import PSO20; PSO20.func(''%s'',''%s'')"'], ...
                      trainfile, optfile);
    % cmdstr1 = sprintf(['call D:/ProgramFiles/Anaconda3/Scripts/activate pytorch & ', ...
    %               'C:/Users/Administrator/.conda/envs/to/python.exe ', ...
    %               '-c "import bba; bba.func(''%s'',''%s'')"'], ...
                  % trainfile, optfile);
    [status1,commandOut1]=system(cmdstr1);
    optdata=load(optfile);
    xnew=reshape(double(optdata.x),size(optdata.x,1),640);

    for j=1:size(xnew,1)
        xnew(j,:)=round(xnew(j,:));
    end
    xnew=unique(xnew,'row','stable');
    [~, ia, ib] = intersect(inputss, xnew, 'rows');
    xnew(ib, :) = [];
    if ~isempty(xnew)
        [sa,sb]=size(xnew);
        rowsToDelete = [];
        for k=1:sa
            import py.utils.isvalid
            if ~isvalid(reshape(xnew(k,:),40,16)') 
            rowsToDelete = [rowsToDelete, k]; 
            end
        end
        xnew(rowsToDelete,:)=[];
        [sa,sb]=size(xnew);
        if sa>10
           xneww=xnew(1:10,:);
        else
            xneww=xnew;
        end
        
        if ~isempty(xneww)
            outputsS=func_outputs_nowrite(xneww,40,16);
            xneww(outputsS==Inf,:)=[];
            outputsS(outputsS==Inf)=[];
        else
            outputsS=[];
        end
        if outputsS
            fprintf('evolution: [%s]',strjoin(string(outputsS),','));
        end
        xneww=reshape(xneww,size(xneww,1),640);
        inputsX=[inputss;xneww];
        outputsX=[ outputss;outputsS];
    else 
        inputsX=inputss;
        outputsX=outputss;
    end
    PSO_inputs = [PSO_inputs;xneww];
    PSO_outputs = [PSO_outputs;outputsS];
    PSO_num = [PSO_num;i*ones(size(outputsS,1),1)];
    best_PSO=[best_PSO;zui(i)*ones(size(outputsS,1),1)];
    [maxoc,c]=sort(outputsX,'descend');
    [m,~]=size(inputsX);
    inputs=inputsX(c,:);
    % inputs=inputs(1:640,:);
    inputs=reshape(inputs,size(inputs,1),40,16);
    outputs=maxoc;
    trainfile=['./',folder,'/',num2str(n),'ntrain_train_step',num2str(i+1),'.mat'];
    
    save(trainfile,'inputs','outputs');
    PSO_outputss = [PSO_num,best_PSO, PSO_outputs];
    save(PSOfile,'PSO_inputs','PSO_outputss');
    AE_outputss = [AE_num,best_AE,AE_outputs];
    save(AEfile,'AE_inputs','AE_outputss');
    toc
end


h=heatmap((reshape((1-xnew(1,:)),40,16))','CellLabelColor','none','FontName','Times New Roman','FontSize',20);
% h=heatmap((reshape((1-inputss(759,:)),40,16))','CellLabelColor','none','FontName','Times New Roman','FontSize',20);
colormap(gray)
colorbar off
h.Position=[0.1,0.18,0.8,0.6];
% function pop = initialPopulation(inputss, outputss)
% % pop = rand(options.PopulationSize, GenomeLength) > 0.9; % 生成随机的二进制种群，大部分元素为0
% [~,b]=sort(outputss,'descend');
% popp=inputss(b,:);
% pop=(popp(1:100,:));
% end

