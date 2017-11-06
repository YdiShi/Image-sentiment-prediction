%--------------------------------------------------------------------------
% randomly initializes slice parameters Theta_xy and Theta_yy
%--------------------------------------------------------------------------

function Theta0 = slice_initialize(J, K )

	if ( K <= 100 )
		Ai = sprandsym(K,0.01);%R = sprandsym(n,density)    %����n��n��ϡ��Գ�������󣬾���Ԫ�ط�����̬�ֲ����ֲ��ܶ�Ϊdensity
		Theta0.xy = sprand( J, K, 0.01 );%����һ��m��n�ķ��Ӿ��ȷֲ������ϡ����󣬷���Ԫ�صķֲ��ܶ���density��rc��һ��Լ������
	else
		Ai = sprandsym(K,0.001);
		Theta0.xy = sprand( J, K, 0.001 );
	end
	Theta0.yy = 0.01*Ai*Ai' + 0.7*speye(K,K);%���ɵ�λϡ�����speye����ֹ������
end
