function y = Phi2(bvector,c,xvector,N)
    A = 15;
    K = exp(A/2);
    alphainv = [0.5,ones(1,N-1)];
    
    n = 0:N-1;
    y1 = (-1).^(n).*alphainv.*real(LaplacePhi2((A+2*pi*1i*n)/2,c,xvector,bvector));
    y = K.*sum(y1);
    
function y = LaplacePhi2(s,c,x,b)
    
    P = ones(1,length(s));    
    for k = 1:length(b)
        P = P.*((1-x(k)*(s.^(-1))).^(-b(k)));
    end
	y = gamma(c)*(s.^(-c)).*P;