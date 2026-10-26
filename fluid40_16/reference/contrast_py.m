clc;
clear all;

folder_opt='opt_g';
mkdir(folder_opt);
trainfile='./train.mat';

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

%加载数据
data = load('data.mat');
data_inputs = data.inputs;
data_outputs = data.outputs;
data_inputs = reshape(data_inputs, size(data_outputs,1),640);
train_num = 641;
xplot = [0];
xfit = [];
xinputs=[];

for i=200:500
    optfile=['./',folder_opt,'/nopt_step',num2str(i),'.mat'];
    %训练网络
    train_inputs = data_inputs(1:train_num,:);
    train_outputs = data_outputs(1:train_num,:);
    inputs=reshape(train_inputs,size(train_inputs,1),40,16);
    outputs=train_outputs;
    save(trainfile,'inputs','outputs');

    % stan_train_outputs=zscore(train_outputs);
    % net = trainNetwork(train_inputs, stan_train_outputs, layers, optionss);

    %PSO
    % lb=zeros(1,640);
    % ub=ones(1,640);
    % pop=randi([0, 1], 200, 640);
    % options = optimoptions('particleswarm', 'MaxIterations', 2000, 'InitialPoints',pop,'FunctionTolerance', 1e-88,'SwarmSize', 200);
    % [x,fval,exitflag,output,points] = particleswarm(@(x)fitnessFunctionWithPenalty(net,x), 640, lb, ub, options);
    % [a,b]=sort(points.Fval);
    % xnew=points.X(b,:);   
  
    cmdstr1 = sprintf(['call D:/ProgramFiles/Anaconda3/Scripts/activate pytorch & ', ...
                      'C:/Users/Administrator/.conda/envs/to/python.exe ', ...
                      '-c "import PSO20; PSO20.func(''%s'',''%s'')"'], ...
                      trainfile, optfile);
    [status1,commandOut1]=system(cmdstr1);
    optdata=load(optfile);
    xnew=reshape(double(optdata.x),size(optdata.x,1),640);
    for j=1:size(xnew,1)
        xnew(j,:)=round(xnew(j,:));
    end
    xnew=unique(xnew,'row','stable');

    %去除堵的
    [sa,sb]=size(xnew);
    rowsToDelete = [];
    for k=1:sa
        import py.utils.isvalid
        if ~isvalid(reshape(xnew(k,:),40,16)') 
            rowsToDelete = [rowsToDelete, k]; 
        end
    end
    xnew(rowsToDelete,:)=[];

    %计算
    [sa,sb]=size(xnew);
    if sa>0
        if sa>10
           xnew=xnew(1:10,:);
        end
        xoutput = func_outputs_nowrite(xnew,40,16);
        fprintf('xoutput: [%s]\n',strjoin(string(xoutput),','));
        [maxo,b]=sort(xoutput,'descend');
        max_xoutput=maxo(1);
        disp(['max_xoutput=',num2str(max_xoutput)]);
        xfit=[xfit;xoutput];
        xinputs=[xinputs;xnew(b(1),:)];
        if i>1
            if max_xoutput>xplot(i-1)
                xplot=[xplot;max_xoutput];
            else
                xplot=[xplot;xplot(i-1)];
            end
        else
            xplot=max_xoutput;
        end
    else
        if i>1
            xplot=[xplot;xplot(i-1)];
        else
            xplot=0;
        end
        xfit=[xfit;0];
    end
    %更新train num
    train_num=train_num+10;
end

plot(xplot);
save('xplot_py.mat','xplot');

% %适应度函数
% function [fitness] = fitnessFunctionWithPenalty(net,x)
%    x=round(x);
%    import py.utils.isvalid
%    if isvalid(reshape(x,40,16)') 
%      fitness = -predict(net,x);
%    else
%      fitness =10;
%    end
% end
