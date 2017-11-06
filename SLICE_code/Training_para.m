function [opt_lambda1,opt_lambda2,opt_lambda3,opt_phi] = Training_para(lambda1_seq, lambda2_seq, lambda3_seq,phi_seq)
cverr  = zeros(length(lambda1_seq), length(lambda2_seq));
count=10;
for i = 1:length(lambda1_seq)
    for j = 1:length(lambda2_seq)
        for k = 1:length(lambda3_seq)
            for m = 1:length(phi_seq)
                for n=11:12
                   n
                   count=count+1;
                   lambda1 = lambda1_seq(i); 
                   lambda2 = lambda2_seq(j);
                   lambda3 = lambda3_seq(k);
                    phi= phi_seq(m);
                    % lambda1_seq = [0.16,0.06]; %lambda_1 / lambda_2 - 正则化参数。 lambda_1控制Theta.yy的Theta.xy和lambda_2的稀疏级别。
                    % lambda2_seq = [0.16,0.08]; 

                    % performs kcv-fold cross validation, kcv must be >= 3
                    kcv = 4;  %交叉验证的次数
;
                    load 'F:\image-sentiment\SLICE_code\data\Emotion6_feat43_normrPCA'
                     xtrain1=double(Emotion6_feat43_normrPCA);
     
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


                    fprintf('sCGGM demo with %d-fold cross-validation...\n', kcv);

                    option.verbose = true; 
                    option.maxiter = 20; 
                    option.D=150;
               

                    opt = scggm_cvgai(trainFeature, trainDistribution, kcv, lambda1, lambda2, lambda3,phi,option);
               
               
                minerr      = 1e99;
                cverr(i,j) = cverr(i,j) + opt.e; 
                end
                cverr(i,j) = cverr(i,j) / kcv/10; 
                if cverr(i,j) < minerr
                    minerr      = cverr(i,j); 
                    opt_lambda1 = lambda1; 
                    opt_lambda2 = lambda2; 
                    opt_lambda3 = lambda3;
                    opt_phi=phi;
                
                end
               
            end
        end
    end
end
 if option.verbose     	   
        fprintf('lambda_1 = %.7f\t lambda_2 = %.7f\t lambda_3 = %.7f\t opt_phi = %.7f\tcross validation error = %.6f\n', opt_lambda1 ,opt_lambda2,opt_lambda3 , opt_phi, cverr(i,j)); 
 end
end
    % compute prediction errors