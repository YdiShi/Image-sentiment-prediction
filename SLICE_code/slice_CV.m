%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Estimates a sparse CGGM with cross validation to select optimal
%  regularization parameters. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ OPT ] = slice( x, y, kcv, lambda1, lambda2,lambda3,phi, option)
default_lambdaseq = [0.32, 0.16, 0.08, 0.04, 0.02, 0.01]; 

if ( nargin <4 || nargin == 5)% 在matlab中定义一个函数时， 在函数体内部， nargin是用来判断输入变量个数的函数。没有运行
    maxiter = 1000;
    tol     = 1e-4;
    verbose = false;
    eta     = 1.5; 
    centered_input = false; 
    ifrefit = true; 
%     Theta0  = scggm_initialize(size(x, 2), size(y, 2));
    Theta0  = scggm_initialize(option.D, size(y, 2));
    if nargin < 4
	lambda1_seq = default_lambdaseq; 
	lambda2_seq = default_lambdaseq; 
    elseif ~isa(lambda1_seq, 'double') || ~isa(lambda2_seq, 'double')
%   K = isa(obj, 'class_name')  判断obj是否为class_name类型。如果是，返回逻辑1（真）；如果不是，返回逻辑0（假）。
	fprintf('sCGGM: error!lambda1_seq lambda2_seq must be vectors\n');
	OPT = {}; return;
    end
else
    if nargin == 4
	if isa(lambda1_seq, 'struct')
		option = lambda1_seq; 
		lambda1_seq = default_lambdaseq; % 保存通过交叉验证选择的最佳λ1/λ2的长度2向量。
		lambda2_seq = default_lambdaseq; 
	else
		fprintf('sCGGM: error! slice must accept lambda1_seq lambda2_seq simultaneously!\n');
		OPT = {}; return; 
	end
    end
    if  isfield(option,'maxiter')
        maxiter = option.maxiter; 
    else
        maxiter = 1000;
    end
    if isfield(option, 'centered_input')
        centered_input = option.centered_input; 
    else
	centered_input = false; %
    end
    if isfield(option, 'ifrefit')
	ifrefit = option.ifrefit; 
    else
	ifrefit = false; 
    end
    if isfield(option, 'Theta0')
        Theta0 = option.Theta0; 
    else
        Theta0= scggm_initialize(option.D, size(y, 2));
%         Theta0.xy=rand(option.D, size(y, 2));
%         Theta0.yy=rand(size(y, 2));
%         Theta0.yy=Theta0.yy'*Theta0.yy+0.01*eye(size(y, 2));
    end
    if isfield(option, 'tol')
        tol = option.tol; 
    else
        tol = 1e-4;
    end
    %一个真/假标志，指定是否将中间结果打印到STDOUT。 默认值为false。
    if isfield(option, 'verbose')
        verbose = option.verbose; 
    else
        verbose = false;
    end
    if isfield(option, 'eta')
        eta  	= option.eta; 
    else
        eta     = 1.5; 
    end
end


%% center the input data if not centered 
N0 = size(x, 1);
if size(y, 1) ~= N0
	fprintf('sCGGM:error! Input data sample size inconsistent!\n');
	OPT = {}; return;  
end
if kcv < 3 || kcv >= N0
	fprintf('sCGGM:error! Cross validation cannot be %d fold \n',kcv); 
	OPT = {}; return; 
end
if ~centered_input
 	y0 = y - repmat(mean(y), N0, 1);
	x0 = x - repmat(mean(x), N0, 1);
else
	x0 = x; 
 	y0 = y; 
end
%y0=y;
J  = size(x, 2);
K  = size(y, 2);



% cross validation index
% cv_indices  = crossvalind('Kfold', N0, kcv);
% %Indices=crossvalind('Kfold',8,4)中‘8’代表元素的个数，‘4’代表分成几类，因为有四类，每类应该有8/4两个元素
% cverr       = zeros(length(lambda1_seq), length(lambda2_seq));
% minerr      = 1e99;

if verbose 
	fprintf('J = %d, K = %d, sample size = %d\n',J, K, N0); 
end
%初始化P，W
% P0=rand(300,option.D);
% W0=rand(option.D);
P0=eye(300,option.D);
W1=normr(rand(size(x,1),option.D));
z0=x0*P0;
% z=x*P0;

%% k-fold cross validation 
% for i = 1:length(lambda1_seq)
%     for j = 1:length(lambda2_seq)
%         for k = 1:length(lambda3_seq)
%             for m = 1:length(phi_seq)
%                 fprintf('i=%d,j=%d\n',i,j,k);
%                 for ff = 1:kcv
%                 % extract centered and uncentered training data
%                 x0_yuan = x0(cv_indices ~= ff, :);%x0中心化之后的数据
%                 x_yuan = x(cv_indices ~= ff, :);
%                 cztr = z0(cv_indices ~= ff, :); 
%                 cytr = y0(cv_indices ~= ff, :); 
%                 ztr  = z(cv_indices ~= ff, :); 
%                 ytr  = y(cv_indices ~= ff, :); 
%                 W0tr  = W1(cv_indices ~= ff, :); 
% 
%                 % extract uncentered cross-validation data
%                 xcv = x(cv_indices == ff, :);
%                 zcv  = z(cv_indices == ff, :);
%                 ycv  = y(cv_indices == ff, :); 
%                 %计算拉普拉斯矩阵
%                 options = [];
%                 options.Metric = 'Euclidean';
%                 options.NeighborMode = 'KNN';
%                 options.k = 5;
%                 options.WeightMode = 'HeatKernel';
%                 S = EuDist2(x0_yuan);
%                 options.t = mean(mean(S));
%                 W = constructW(x0_yuan,options);
%                 D = diag(sum(W,2));%diag(a1,a2,……,an)表示的是对角线元素为a1,a2,……,an的对角矩阵，sum(x,2)以矩阵x的每一行为对象，对一行内的数字求和。
%                 La = D-W;
% 
%                 % estimate a sparse estimate of Theta_xy and Theta_yy
%         % 		[raw_Theta, obj,W0,P]= scggm_sparse_step(lambda1, lambda2, x0_yuan, cztr, cytr, maxiter, tol, verbose, eta, Theta0,W0,La);
%                 [raw_Theta, obj,P,Z,W0]= scggm_sparse_step(lambda1, lambda2, lambda3,phi,x0_yuan, cztr, cytr, maxiter, tol, verbose, eta, Theta0,La,W0tr,P0);
% 
%                 if ifrefit
%                     % refit the parameters
%                     %option.ifrefit - 一个用于指定是否执行重新安装程序的标志。如果option.ifrefit = false，该例程将从优化的L1正则化数据对数返回参数估计值。 
%                     %如果option.ifrefit = true，则例程将执行附加的重新安装步骤，重新估计估计参数的非零项，而不会造成L1惩罚。 
%                     %这种重新拟合步骤的目的是消除由L1惩罚引入的偏差的影响。 默认值为option.ifrefit = true。
%                     zero_theta = scggm_zero_index(raw_Theta);
%                     %cztr=x0_yuan*P;
%                     [Theta, obj,P,Z] = scggm_refit_step(x0_yuan,Z, cytr, zero_theta, maxiter, tol, verbose, eta, raw_Theta,La,lambda3,phi,W0,P); 
%                 else
%                     Theta = raw_Theta; 
%                 end
%                 %P= scggm_refit_Normalization_graph(x,cytr,cxtr,Theta,W0); 
%                 % compute cross-validation error
%                 intercept = mean(ytr) + mean(x_yuan*P) * (Theta.xy * inv(Theta.yy));
%                 [Eycv, e] = scggm_predict(Theta, intercept,xcv*P, ycv); 
%                 cverr(i,j) = cverr(i,j) + e; 
%                 end
% 
%                 cverr(i,j) = cverr(i,j) / kcv; 
%                 if cverr(i,j) < minerr
%                     minerr      = cverr(i,j); 
%                     opt_lambda1 = lambda1; 
%                     opt_lambda2 = lambda2; 
%                     opt_lambda3 = lambda3;
%                      opt_phi = phi;
%                 end
%                 if verbose     	   
%                         fprintf('lambda_1 = %.7f\t lambda_2 = %.7f\t lambda_3 = %.7f\t phi = %.7f\t cross validation error = %.6f\n', lambda1, lambda2,lambda3,phi, cverr(i,j)); 
%                 
%                 end
%             end
%         end
%     end
% end
% 
% if verbose 
%     fprintf('\ntraining sCGGM with optimal regularization parameters: \n'); 
%     fprintf('optimal lambda_1 = %.7f\t optimal lambda_2 = %.6f\t optimal lambda_3 = %.6f\t ... \n', opt_lambda1, opt_lambda2,opt_lambda3); 
% end
% 计算拉普拉斯矩阵
 options = [];
 options.Metric = 'Euclidean';
 options.NeighborMode = 'KNN';
 options.k = 5;
 options.WeightMode = 'HeatKernel';
S = EuDist2(x0);
options.t = mean(mean(S));
W= constructW(x0,options);
D = diag(sum(W,2));%diag(a1,a2,……,an)表示的是对角线元素为a1,a2,……,an的对角矩阵，sum(x,2)以矩阵x的每一行为对象，对一行内的数字求和。
La = D-W;
  [raw_Theta,P,W0]= scggm_sparse_step(lambda1, lambda2, lambda3,phi,x0, z0, y0, maxiter, tol, verbose, eta, Theta0,La,W1,P0);

if ifrefit
	zero_theta = scggm_zero_index(raw_Theta);
	z0=x0*P;
    [Theta,P,Z] = scggm_refit_step(x0,z0, y0, zero_theta, maxiter, tol, verbose, eta, raw_Theta,La,lambda3,phi,W0,P); 
else
	Theta = raw_Theta; 
end

OPT.Theta 	= Theta; 
%OPT.lambdas     = [ opt_lambda1, opt_lambda2 ];
OPT.intercept	= mean(y) + mean(x) *P* (Theta.xy * inv(Theta.yy)); 
%OPT.Eycv  = Eycv;
OPT.P  = P;
%OPT.Z  = Z;

end
