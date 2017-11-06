%--------------------------------------------------------------------------
% evaluate sCGGM objective and/or gradient
%--------------------------------------------------------------------------

function [ value, flag, grad] = scggm_evaluate(cz,theta, Sz, Szy, Sy, N, gradient, verbose,La,phi)

flag        = 0;
[ cyy, p ]  = chol(theta.yy);%[R,p]=chol(X)：这个命令格式将不输出出错信息。当X为对称正定的，则p=0，R与上述格式得到的结果相同；否则p为一个正整数。如

if ( p > 0 )
	if strcmp(gradient, 'y') == 1 && verbose
		fprintf( 'sCGGM: Theta_yy not positive definite!\n' );
	end
	flag        = 1;
	value       = inf;
	grad        = theta;
	return;
end

logdetyy = 2 * sum(log(diag(cyy) ));

if ( isnan(logdetyy) || isinf(logdetyy) )
	if verbose
		fprintf( 'sCGGM: logdet Theta_yy is Nan or Inf!\n' );
	end
	flag = 1;
	value = inf;
	grad = theta;
	return; 
end
Sz=double(Sz);
Szy=double(Szy);
icyy	 = cyy \ eye(size(cyy,2));
ithetayy = icyy * icyy';
txyityy  = theta.xy*ithetayy;
XtXth    = Sz*txyityy;
txyXtXth = theta.xy'*Sz*txyityy;

l1 = trace( theta.yy*Sy );%%
l2 = trace( Szy*theta.xy' );
l3 = trace( txyXtXth );
l4 = phi*trace(cz'*La*cz);
value = 0.5*l1 + l2 + 0.5*l3 - 0.5*N*logdetyy+l4;
value = value / N;%%%目标函数



if strcmp('y',gradient) ==1
	grad.xy = (Szy + XtXth)/N;
	grad.yy = 0.5*(Sy - N*ithetayy - ithetayy*txyXtXth)/N;
else
	grad = [];
end

