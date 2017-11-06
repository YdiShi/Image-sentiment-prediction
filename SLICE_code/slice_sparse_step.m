%--------------------------------------------------------------------------
% estimate a sparse CGGM
%--------------------------------------------------------------------------

function [Theta,P,W0] = scggm_sparse_step( lambda1, lambda2,lambda3,phi,x,cz, cy, maxiter, tol, verbose, eta, Theta0,La,W0,P)

Sz	= cz'*cz; 
Sy 	= cy'*cy; 
Szy	= cz'*cy;
N 	= size(cz, 1);

nobj	= 10;
bconv	= 0;
obj	= zeros(maxiter, 1);
theta	= Theta0; 
L	= 1; 
thk_0 	= 2/3; 
ls_maxiter = 300; 

[obj1, init_flag]  = scggm_evaluate( cz,theta, Sz, Szy, Sy, N, 'n', verbose,La,phi);
if init_flag == 1 && verbose == true
	fprintf('sCGGM: error! initial Theta_yy not positive definite!\n');
end
obj(1) = obj1 + scggm_penalty(theta, lambda1, lambda2,lambda3,P);

xk      = theta;%xk做一种处理，zk做一种处理梯度下降生成theta
zk   	= theta;
thk     = thk_0;

for iter = 2:maxiter
    iter
	thk  = (sqrt( thk^4 + 4 * thk^2 ) - thk^2) / 2; % momentum of acceleration
	y.xy = (1 - thk) * xk.xy + thk * zk.xy;
	y.yy = (1 - thk) * xk.yy + thk * zk.yy;
	[ fyk, flagy, grady] = scggm_evaluate( cz,y, Sz, Szy, Sy, N, 'y', verbose,La,phi); % compute the objective and gradient for y
    
	% line search 原来的目标函数
	ik = 0; 
	while ( 1 )
 	       % gradient step
		zk_grady.xy = zk.xy - 1/(L*thk) * grady.xy;	
		zk_grady.yy = zk.yy - 1/(L*thk) * grady.yy;
        	% proximal step
		zk1 		= scggm_soft_threshold( zk_grady, 2*lambda1/(thk*L), 1*lambda2/(thk*L)) ;
	        % gradient step
		y_grady.xy	= y.xy - 1/L * grady.xy;
		y_grady.yy	= y.yy - 1/L * grady.yy;
       		% proximal step
        xk1         = scggm_soft_threshold( y_grady, 2*lambda1/(L), 1*lambda2/(L));
        
		[fxk1, flagxk1] = scggm_evaluate(cz,xk1, Sz, Szy, Sy, N ,'n', verbose,La,phi);
		[~, flagzk1]    = chol(zk1.yy);

		if ( flagzk1 == 0 && flagy ==0 && flagxk1 ==0 ) % xk1,zk1,y all positive definite
			xk1_y.xy    = xk1.xy - y.xy;
			xk1_y.yy    = xk1.yy - y.yy;	
			lfxk1_y     = fyk + grady.xy(:)'* (xk1_y.xy(:)) + grady.yy(:)'*(xk1_y.yy(:));
			diffxk1y.xy = xk1.xy - y.xy;
			diffxk1y.yy = xk1.yy - y.yy;
			RHS         = lfxk1_y + L/2 *(sum(diffxk1y.xy(:).^2) + sum(diffxk1y.yy(:).^2));%现在的目标函数
			if fxk1 <= RHS + tol%如果此时的目标函数比较小
				xk = xk1;
				zk = zk1;
				bconv = 1;
				break; % line search converged
			end
        	end
        
		ik = ik + 1;
        
		if ( ik > ls_maxiter )
			if verbose
				fprintf( 'sCGGM: line search not converging,ik = %d\n',ik); 
			end
			bconv = 0;
			iter  = max(1, iter - 1); 
			Theta = xk; 
			break;
		end
		L = L * eta;
	end 
	obj(iter)  = fxk1 + scggm_penalty( xk, lambda1, lambda2,lambda3,P);
    [P,Z,W]= scggm_LowRank(x,cz,cy,xk1,La,P,lambda3,phi,W0);  
    W0=W;
    cz=x*P;
    Sz	= cz'*cz; 
    Sy 	= cy'*cy;
    Szy	= cz'*cy;
	if bconv == 0
		break;
	end
    
	if ( iter > nobj + 1)
		value           = obj(iter);
		prevVals        = obj(iter - nobj);
		avgimprovement  = abs( prevVals - value )/nobj;
		relAvgImpr      = avgimprovement / abs( value ) ; % relative average improvement
            
		if ( relAvgImpr < tol )
			bconv = 1;
			break;
		end
	end
end  

Theta = xk; 
obj   = obj(1:iter);
% W0=W;

