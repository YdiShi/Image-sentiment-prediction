                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               function [P,Z,W] =scggm_LowRank(cx,cz,cy,theta,L,Pi,lambda3,phi,W0)
mu=0.0001;
rho = 1.1;
max_mu = 1e6;
% [sample_dim, feature_dim] = size(cx);
% [ cyy, p ]  = chol(theta.yy);
% icyy	 = cyy \ eye(size(cyy,2));
% ithetayy = icyy * icyy';

% A=2*pinv(cx'*cx)*cx'*full(L)*cx;
A=2*phi*full(L);
%A=A+0.0001*eye(size(A,1));
% B=theta.xy*pinv(theta.yy)*theta.xy'-2*W0;
B=theta.xy*pinv(theta.yy)*theta.xy'+mu*eye(size(theta.xy,1));
%C=mu*cx*P-cy*theta.xy'-W;
%C=C+0.0001*eye(size(C));
%P = sylvester(A,B,C);
 %partial_L=Z-X*P;
for K = 1:6
%      W0=W0+mu*partial_L;
     C=-mu*cx*Pi+cy*theta.xy'+W0;
     % = sylvester(A,B,C);
     %Zi = lyap(A,B,C);
     Zi=(-C)*inv(B);
     %partial_L=Z-X*P;
    %% update Pi
    tau = mu*norm(cx,'fro')^2;                                                                            
%     Zip = Y1'*P'*X+mu*X'*P*(P'*X-P'*X*(Zi+Zp)-E);
    Zip = mu*cx'*Zi+cx'*W0;
    temp = tau*(Pi - mu*cx'*cx*Pi+Zip);
    [Ui,sigmai,Vi] = svd(temp,'econ');
    sigmai = diag(sigmai);
    svp = length(find(sigmai>lambda3*tau));
    if svp>=1
        sigmai = sigmai(1:svp)-lambda3*tau;
    else
        svp = 1;
        sigmai = 0;
    end
    Pi = Ui(:,1:svp)*diag(sigmai)*Vi(:,1:svp)';
     %% update parameters
   
    leq1 = Zi-cx*Pi;
    W0 = W0 + mu*leq1;
    mu = min(max_mu,mu*rho);
 end
Z=Zi;
P=Pi;
W=W0;
 end
