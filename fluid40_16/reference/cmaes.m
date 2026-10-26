function [best_x, best_y] = cmaes(func, initial_population)

%function [fitness] = func(net,x)
%    x=reshape(x,1,160,1,size(x,1));
%    fitness = -predict(net,x); 
%end

    % 默认参数
    if nargin < 3
        options = struct();
    end

    % 维度和种群大小
    dim = size(initial_population, 2); % 108 维
    lambda = size(initial_population, 1); % 10 个样本
    mu = floor(lambda / 2); % 精英个体数

    % 算法参数
    sigma = 0.5;        % 初始步长
    max_iter = 100;     % 最大迭代次数

    % 初始化均值
    xmean = mean(initial_population, 1)'; % 初始均值
    pc = zeros(dim, 1);   % 进化路径
    ps = zeros(dim, 1);   % 均值向量的变化
    C = eye(dim);         % 协方差矩阵

    % 适应度
    fitness = zeros(lambda, 1);

    best_x=[];
    best_y=[];
    for iter = 1:max_iter
        % 生成种群
        for k = 1:lambda
            z(k, :) = randn(1, dim);
            offspring(:, k) = xmean + sigma * (C^(0.5) * z(k, :)');
        end
        
        % 评估适应度
        for k = 1:lambda
            fitness(k) = func(offspring(:, k)');
        end
        
        % 选择最优个体
        [~, indices] = sort(fitness);
        xmean = mean(offspring(:, indices(1:mu)), 2); % 更新均值
        best_solution = xmean;
        best_fitness = fitness(indices(1));
        best_x=[best_x;best_solution'];
        best_y=[best_y;best_fitness];

        % 更新协方差矩阵和步长
        % 这里可以加入更新C、sigma和其他参数的具体实现
    end
    [~,loc]=sort(best_y,'descend');
%    loc=loc(1:topk);
    best_x=best_x(loc,:);
    best_y=best_y(loc);
end