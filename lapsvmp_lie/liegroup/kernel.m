function k = kernel(ker, a, b, sig)
switch lower(ker)
    case 'log'
        k=trace(logm(a)*logm(b));
    case 'rbf'
        d = norm(logm(b)-logm(a), 'fro');
        k=exp(-d*d/sig);
    case 'line'
        k = norm(a*b);
    case 'polym'
        k = (trace(b'*a) + 1 )^2;
    case 'linear'
        k = trace(b'*a);
    case 'tanh'
        bb = 1/1000000000;
        c = 0;
        k = tanh(bb*(trace(b'*a)) - c);
    case 'sig'
        c = -1/1000000000;
        k = 1/( 1+exp(c*(trace(b'*a))) );
    otherwise
end