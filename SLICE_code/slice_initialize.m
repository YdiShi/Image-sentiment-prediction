%--------------------------------------------------------------------------
% randomly initializes slice parameters Theta_xy and Theta_yy
%--------------------------------------------------------------------------

function Theta0 = slice_initialize(J, K )

	if ( K <= 100 )
		Ai = sprandsym(K,0.01);%R = sprandsym(n,density)    %生成n×n的稀疏对称随机矩阵，矩阵元素服从正态分布，分布密度为density
		Theta0.xy = sprand( J, K, 0.01 );%生成一个m×n的服从均匀分布的随机稀疏矩阵，非零元素的分布密度是density，rc是一个约束条件
	else
		Ai = sprandsym(K,0.001);
		Theta0.xy = sprand( J, K, 0.001 );
	end
	Theta0.yy = 0.01*Ai*Ai' + 0.7*speye(K,K);%生成单位稀疏矩阵speye，防止不可逆
end
