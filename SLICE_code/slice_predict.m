%--------------------------------------------------------------------------
% predict Y (gene-expression) based on X (genotype) and compute
% prediction error when true Y (gene-expression) is given.
%--------------------------------------------------------------------------

function [Ey_ts, e] = scggm_predict(Theta, intercept, x_ts, y_ts)

K = size(Theta.yy, 2); 
N_ts = size(x_ts, 1);

Beta = scggm_indirect_SNP_overall(Theta); 

Ey_ts = x_ts * Beta  + repmat( intercept, N_ts , 1); 
% Ey_ts = x_ts * Beta; 
if nargin > 3
	if size(y_ts, 1) ~=N_ts
		fprintf('sCGGM:error! Genotype and expression test data sample size inconsistent!\n');
		Ey_ts = []; 
		e = nan; 
		return; 
    end
    y_ts=y_ts+0.0001*ones(size(y_ts, 1),size(y_ts, 2));
    e=kldist(y_ts, Ey_ts)
% 	res  = y_ts- Ey_ts; 
% 	prederr = sum(sum(res.^2)) / K / N_ts;
else
    e = nan;
end
