%--------------------------------------------------------------------------
% Sample run of the sparse CGGM algorithm with cross-validation
%--------------------------------------------------------------------------
%% sCGGM demo with cross-validation 
% specify the search grid of regularization parameters
clear 
count=0;
addpath('F:\shiyingdi\SCGGM_code');
for i=1:200
    i
    count=count+1;
    lambda1_seq = [0.00008]; %lambda_1 / lambda_2 - 正则化参数。 lambda_1控制Theta.yy的Theta.xy和lambda_2的稀疏级别。
    lambda2_seq = [0.0052]; 
    % lambda1_seq = [0.16,0.06]; %lambda_1 / lambda_2 - 正则化参数。 lambda_1控制Theta.yy的Theta.xy和lambda_2的稀疏级别。
    % lambda2_seq = [0.16,0.08]; 

    % performs kcv-fold cross validation, kcv must be >= 3
    kcv = 5;  %交叉验证的次数

    % loading traing data and test data 
    xtrain1 = dlmread('F:\shiyingdi\Emotion6_feat43_norm.mat'); %基于N型矩阵的训练数据，其中N是训练样本的数量，J是特征的数量
    % % xtrain2 = xtrain1.Flickr_LDL_feat39;
    % xtrain2 = dlmread('Flickr_LDL_Sentiment_1200_norm.mat')
    % xtrain = [xtrain1,xtrain2];
    %xtrain = xtrain1 (1+396:1584+396,:);
    random_generator=randperm(330);
    random_save(count,:)=random_generator;
    tran_random=random_generator(1:264);
    test_random=random_generator(264+1:330);
    xtrain_C1=xtrain1(tran_random,:);
    xtrain_C2=xtrain1(tran_random+330,:);
    xtrain_C3=xtrain1(tran_random+330*2,:);
    xtrain_C4=xtrain1(tran_random+330*3,:);
    xtrain_C5=xtrain1(tran_random+330*4,:);
    xtrain_C6=xtrain1(tran_random+330*5,:);
    xtrain=[xtrain_C1;xtrain_C2;xtrain_C3;xtrain_C4;xtrain_C5;xtrain_C6]; 
    ytrain1 = load('F:\shiyingdi\y_emotion6.txt');
    ytrain_C1=ytrain1(tran_random,:);
    ytrain_C2=ytrain1(tran_random+330,:);
    ytrain_C3=ytrain1(tran_random+330*2,:);
    ytrain_C4=ytrain1(tran_random+330*3,:);
    ytrain_C5=ytrain1(tran_random+330*4,:);
    ytrain_C6=ytrain1(tran_random+330*5,:);
    ytrain=[ytrain_C1;ytrain_C2;ytrain_C3;ytrain_C4;ytrain_C5;ytrain_C6];

    xtest1 = dlmread('F:\shiyingdi\Emotion6_feat43_norm.mat');%基于N型矩阵的训练数据，其中N是训练样本的数量，J是特征的数量
    % xtest2 = xtest1.Flickr_LDL_feat39;
    %xtest = xtest1(1:396,:);
    xtest_C1=xtest1(test_random,:);
    xtest_C2=xtest1(test_random+330,:);
    xtest_C3=xtest1(test_random+330*2,:);
    xtest_C4=xtest1(test_random+330*3,:);
    xtest_C5=xtest1(test_random+330*4,:);
    xtest_C6=xtest1(test_random+330*5,:);
    xtest=[xtest_C1;xtest_C2;xtest_C3;xtest_C4;xtest_C5;xtest_C6]; 


    ytest1 = load('F:\shiyingdi\y_emotion6.txt');
    ytest_C1=ytest1(test_random,:);
    ytest_C2=ytest1(test_random+330*1,:);
    ytest_C3=ytest1(test_random+330*2,:);
    ytest_C4=ytest1(test_random+330*3,:);
    ytest_C5=ytest1(test_random+330*4,:);
    ytest_C6=ytest1(test_random+330*5,:);
    ytest=[ytest_C1;ytest_C2;ytest_C3;ytest_C4;ytest_C5;ytest_C6];
    %ytest=ytest1;

    fprintf('sCGGM demo with %d-fold cross-validation...\n', kcv);

    option.verbose = true; 
    option.maxiter = 100; 

    opt = scggm_cv( xtrain, ytrain, kcv, lambda1_seq, lambda2_seq, option);

    % compute prediction errors
    

    % perform inference
%     [Beta] = scggm_indirect_SNP_overall(opt.Theta); 
 xtests= xtest-repmat(mean(xtrain),396,1);
 ytests= ytest-repmat(mean(ytrain),396,1);
    [pre_y, e] = scggm_predict(opt.Theta, opt.intercept,  xtests, ytests);% 截距项是与标准回归模型中的截距对应的长度K向量
    fprintf('sCGGM demo  completed, test set prediction error: %g\n', e); 

    % decomposition of overall indirect SNP perturbations
    % passed by the k-th gene
%     k = 3  % k = 1 ... 30 
%     [Beta_k] = scggm_indirect_SNP_decompose(opt.Theta, k);
   for i=1:396
        % Show the comparisons between the predicted distribution
        [disName, distance(count,:)] = computeMeasures(ytest(i,:), pre_y(i,:));
        % Draw the picture of the real and prediced distribution.
        %drawDistribution(ytest(i,:),pre_y(i,:),disName, distance);
    end
    distance(count,:)
    
    % decomposition of gene-expression covariance
    Cov = scggm_cov_decompose(opt.Theta, xtrain, ytrain);

    if ~exist('F:\shiyingdi\SCGGM_code\results\demo_cv', 'dir')
        mkdir('F:\shiyingdi\SCGGM_code\results\demo_cv'); 
    end
    % 1）optimal_Theta_xy.txt - 包含估计的Theta_xy（J由K矩阵）。
    % 2）optimal_Theta_yy.txt - 包含估计的Theta_yy（K by K matrix）。
    % 3）optimal_intercept.txt - 估计截距（长度K向量）。
    % 4）Beta.txt - 间接SNP扰动（J由K矩阵）。
    % 5）Beta_2.txt - 总体间接SNP效应的分解。结果保存
    % 仅针对通过SNP的第二基因表达特征的组分
    % 扰动效应。 （J由K矩阵）。
    % 6）Cov_Overall.txt - 整体基因表达协方差。
    % 7）Cov_Network_Induced.txt - 基因表达协方差的组成部分
    % 由基因表达网络Theta.yy诱导，基因表达协方差分解后。
    % 8）Cov_SNP_Induced.txt - 基因表达协方差分量的诱导
    % 通过SNP扰动，基因表达协方差分解后。
    % %dlmwrite('./data_y/optimal_y.txt', opt.Eycv, '\t');
    % dlmwrite('./results/demo_cv/optimal_Theta_xy.txt', opt.Theta.xy, '\t');
    % dlmwrite('./results/demo_cv/optimal_Theta_yy.txt', opt.Theta.yy, '\t'); 
    % dlmwrite('./results/demo_cv/optimal_intercept.txt', opt.intercept, '\t');
    % dlmwrite('./results/demo_cv/optimal_lambdas.txt', opt.lambdas, '\t');
    % dlmwrite('./results/demo_cv/Beta.txt', Beta, '\t');
    % %dlmwrite(['./results/demo_cv/Beta_', num2str(k), '.txt'], Beta_k, '\t');
    % dlmwrite('./results/demo_cv/Cov_Overall.txt', Cov.Overall, '\t'); 
    % dlmwrite('./results/demo_cv/Cov_Network_Induced.txt', Cov.Network_Induced, '\t');
    % dlmwrite('./results/demo_cv/Cov_SNP_Induced.txt', Cov.SNP_Induced, '\t');
%     save opt.Theta.xy
%     save opt.Theta.yy  
%     save opt.intercept
end
