%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Estimates a sparse CGGM with cross validation to select optimal
%  regularization parameters. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ OPT ] = slice( x, y, kcv, lambda1, lambda2,lambda3,phi, option)
default_lambdaseq = [0.32, 0.16, 0.08, 0.04, 0.02, 0.01]; 

if ( nargin <4 || nargin == 5)% ��matlab�ж���һ������ʱ�� �ں������ڲ��� nargin�������ж�������������ĺ�����û������
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
%   K = isa(obj, 'class_name')  �ж�obj�Ƿ�Ϊclass_name���͡�����ǣ������߼�1���棩��������ǣ������߼�0���٣���
	fprintf('sCGGM: error!lambda1_seq lambda2_seq must be vectors\n');
	OPT = {}; return;
    end
else
    if nargin == 4
	if isa(lambda1_seq, 'struct')
		option = lambda1_seq; 
		lambda1_seq = default_lambdaseq; % ����ͨ��������֤ѡ�����Ѧ�1/��2�ĳ���2������
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
    %һ����/�ٱ�־��ָ���Ƿ��м�����ӡ��STDOUT�� Ĭ��ֵΪfalse��
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
% %Indices=crossvalind('Kfold',8,4)�С�8������Ԫ�صĸ�������4������ֳɼ��࣬��Ϊ�����࣬ÿ��Ӧ����8/4����Ԫ��
% cverr       = zeros(length(lambda1_seq), length(lambda2_seq));
% minerr      = 1e99;

if verbose 
	fprintf('J = %d, K = %d, sample size = %d\n',J, K, N0); 
end
%��ʼ��P��W
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
%                 x0_yuan = x0(cv_indices ~= ff, :);%x0���Ļ�֮�������
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
%                 %����������˹����
%                 options = [];
%                 options.Metric = 'Euclidean';
%                 options.NeighborMode = 'KNN';
%                 options.k = 5;
%                 options.WeightMode = 'HeatKernel';
%                 S = EuDist2(x0_yuan);
%                 options.t = mean(mean(S));
%                 W = constructW(x0_yuan,options);
%                 D = diag(sum(W,2));%diag(a1,a2,����,an)��ʾ���ǶԽ���Ԫ��Ϊa1,a2,����,an�ĶԽǾ���sum(x,2)�Ծ���x��ÿһ��Ϊ���󣬶�һ���ڵ�������͡�
%                 La = D-W;
% 
%                 % estimate a sparse estimate of Theta_xy and Theta_yy
%         % 		[raw_Theta, obj,W0,P]= scggm_sparse_step(lambda1, lambda2, x0_yuan, cztr, cytr, maxiter, tol, verbose, eta, Theta0,W0,La);
%                 [raw_Theta, obj,P,Z,W0]= scggm_sparse_step(lambda1, lambda2, lambda3,phi,x0_yuan, cztr, cytr, maxiter, tol, verbose, eta, Theta0,La,W0tr,P0);
% 
%                 if ifrefit
%                     % refit the parameters
%                     %option.ifrefit - һ������ָ���Ƿ�ִ�����°�װ����ı�־�����option.ifrefit = false�������̽����Ż���L1�������ݶ������ز�������ֵ�� 
%                     %���option.ifrefit = true�������̽�ִ�и��ӵ����°�װ���裬���¹��ƹ��Ʋ����ķ�������������L1�ͷ��� 
%                     %����������ϲ����Ŀ����������L1�ͷ������ƫ���Ӱ�졣 Ĭ��ֵΪoption.ifrefit = true��
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
% ����������˹����
 options = [];
 options.Metric = 'Euclidean';
 options.NeighborMode = 'KNN';
 options.k = 5;
 options.WeightMode = 'HeatKernel';
S = EuDist2(x0);
options.t = mean(mean(S));
W= constructW(x0,options);
D = diag(sum(W,2));%diag(a1,a2,����,an)��ʾ���ǶԽ���Ԫ��Ϊa1,a2,����,an�ĶԽǾ���sum(x,2)�Ծ���x��ÿһ��Ϊ���󣬶�һ���ڵ�������͡�
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
