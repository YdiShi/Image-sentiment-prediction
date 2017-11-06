function [P0,Z] =slice_LowRank(cx,cy,Z,theta,L,P,lambda3)
mu=0.1;
rho = 1.1;
max_mu = 1e6;
% [sample_dim, feature_dim] = size(cx);
% [ cyy, p ]  = chol(theta.yy);
% icyy	 = cyy \ eye(size(cyy,2));
% ithetayy = icyy * icyy';

% A=2*pinv(cx'*cx)*cx'*full(L)*cx;
A=2*full(L)-mu*size(full(L));
A=A+0.0001*eye(size(A,1));
% B=theta.xy*pinv(theta.yy)*theta.xy'-2*W0;
B=2*theta.xy*pinv(theta.yy)*theta.xy';
%C=mu*cx*P-cy*theta.xy'-W;
%C=C+0.0001*eye(size(C));
%P = sylvester(A,B,C);
 %partial_L=Z-X*P;
for K = 1:50
%      W0=W0+mu*partial_L;
     C=mu*cx*P-cy*theta.xy'-W;
     P = sylvester(A,B,C);
     %partial_L=Z-X*P;
    %% update Zi
    eta = norm(X,F);
%     Zip = Y1'*P'*X+mu*X'*P*(P'*X-P'*X*(Zi+Zp)-E);
    Zip = mu*X'*Z-X'*W;
    temp = Zi - mu*X'*X*Zi+Zip;
    [Ui,sigmai,Vi] = svd(temp,'econ');
    sigmai = diag(sigmai);
    svp = length(find(sigmai>lambda3/(mu*eta)));
    if svp>=1
        sigmai = sigmai(1:svp)-lambda3/(mu*eta);
    else
        svp = 1;
        sigmai = 0;
    end
    Zi = Ui(:,1:svp)*diag(sigmai)*Vi(:,1:svp)';
     %% update parameters
   
    leq1 = Zi-X*P;
    W0 = W0 + mu*leq1;
    mu = min(max_mu,mu*rho);
 end
Z=Zi;
P=P0;
 end
