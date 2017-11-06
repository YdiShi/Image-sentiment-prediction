%--------------------------------------------------------------------------
% Sample run of the sparse CGGM algorithm with cross-validation
%--------------------------------------------------------------------------
%% demo with cross-validation 
% specify the search grid of regularization parameters
clear 
n=0;
lambda1_seq = [0.00001]; %
lambda2_seq = [0.0001]; 
lambda3_seq = [0.05];
phi_seq=[0];
%[opt_lambda1,opt_lambda2,opt_lambda3,opt_phi ] = Training_para( lambda1_seq , lambda2_seq, lambda3_seq,phi_seq); 
 for i=1:20
   i
    n=n+1;
   lambda1 =lambda1_seq; 
   lambda2 = lambda2_seq ;
   lambda3 = lambda3_seq;
    phi=phi_seq;
   

    % performs kcv-fold cross validation, kcv must be >= 3
    kcv = 4;  

    % loading traing data and test data 
    load 'F:\image-sentiment\SLICE_code\data\Emotion6_feat43_normrPCA'
    xtrain1=double(Emotion6_feat43_normrPCA);

    % % xtrain2 = xtrain1.Flickr_LDL_feat39;
    % xtrain2 = dlmread('Flickr_LDL_Sentiment_1200_norm.mat')
    % xtrain = [xtrain1,xtrain2];
    %xtrain = xtrain1 (1+396:1584+396,:);
    %random_generator=randperm(330);
   load 'F:\image-sentiment\SLICE_code\data\random_save'; 
    random_generator=random_save(n,:);
    tran_random=random_generator(1:264);
    test_random=random_generator(264+1:330);
    xtrain_C1=xtrain1(tran_random,:);
    xtrain_C2=xtrain1(tran_random+330,:);
    xtrain_C3=xtrain1(tran_random+330*2,:);
    xtrain_C4=xtrain1(tran_random+330*3,:);
    xtrain_C5=xtrain1(tran_random+330*4,:);
    xtrain_C6=xtrain1(tran_random+330*5,:);
    trainFeature=[xtrain_C1;xtrain_C2;xtrain_C3;xtrain_C4;xtrain_C5;xtrain_C6]; 
    ytrain1 = load('F:\image-sentiment\SLICE_code\data\y_emotion6.txt');
    ytrain1=ytrain1+0.0001*ones(1980,7);
    ytrain_C1=ytrain1(tran_random,:);
    ytrain_C2=ytrain1(tran_random+330,:);
    ytrain_C3=ytrain1(tran_random+330*2,:);
    ytrain_C4=ytrain1(tran_random+330*3,:);
    ytrain_C5=ytrain1(tran_random+330*4,:);
    ytrain_C6=ytrain1(tran_random+330*5,:);
    trainDistribution=[ytrain_C1;ytrain_C2;ytrain_C3;ytrain_C4;ytrain_C5;ytrain_C6];

    xtest1 = xtrain1;
    % xtest2 = xtest1.Flickr_LDL_feat39;
    %xtest = xtest1(1:396,:);
    xtest_C1=xtest1(test_random,:);
    xtest_C2=xtest1(test_random+330,:);
    xtest_C3=xtest1(test_random+330*2,:);
    xtest_C4=xtest1(test_random+330*3,:);
    xtest_C5=xtest1(test_random+330*4,:);
    xtest_C6=xtest1(test_random+330*5,:);
    testFeature=[xtest_C1;xtest_C2;xtest_C3;xtest_C4;xtest_C5;xtest_C6]; 


    ytest1 = load('F:\image-sentiment\SLICE_code\data\y_emotion6.txt');
    ytest1=ytest1+0.0001*ones(1980,7);
    ytest_C1=ytest1(test_random,:);
    ytest_C2=ytest1(test_random+330*1,:);
    ytest_C3=ytest1(test_random+330*2,:);
    ytest_C4=ytest1(test_random+330*3,:);
    ytest_C5=ytest1(test_random+330*4,:);
    ytest_C6=ytest1(test_random+330*5,:);
    testDistribution=[ytest_C1;ytest_C2;ytest_C3;ytest_C4;ytest_C5;ytest_C6];
%     [Beta] = scggm_indirect_SNP_overall(opt.Theta); 
%  testFeature= xtest-repmat(mean(trainFeature),396,1);
%  testDistribution= ytest-repmat(mean(trainDistribution),396,1);

    fprintf('sCGGM demo with %d-fold cross-validation...\n', kcv);

    option.verbose = true; 
    option.maxiter = 20; 
    option.D=150;
    

    opt = scggm_CV(trainFeature, trainDistribution, kcv, lambda1, lambda2, lambda3,phi,option);

    xtests= testFeature;
    ytests= testDistribution;
    ztests=xtests*opt.P;
    %ztests=xtests*eye(300,option.D);
    [pre_y, e] = scggm_predict(opt.Theta, opt.intercept,ztests, ytests);
    fprintf('sCGGM demo  completed, test set prediction error: %g\n', e); 

   for j=1:396
        % Show the comparisons between the predicted distribution
        [disName, distance(n,:)] = computeMeasures(ytests(j,:), pre_y(j,:));
    end
    distance(n,:)
end

