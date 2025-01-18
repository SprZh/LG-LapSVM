function K = liekernel(options,X1,X2)
%      computes the Gram matrix of a specified kernel function.
% 
%      K = liekernel(options,X1)
%      K = lieckernel(options,X1,X2)
%
%      options: a structure with the following fields
%               options.Kernel: 'linear' | 'poly' | 'rbf' 
%               options.KernelParam: specifies parameters for the kernel 
%                                    functions, i.e. degree for 'poly'; 
%                                    sigma for 'rbf'; can be ignored for 
%                                    linear kernel 
%      X1: N-by-1 data matrix of N k-by-k-dimensional image covariance features
%      X2: (it is optional) M-by-1 data matrix of M k-by-k-dimensional image covariance features
% 
%      K: N-by-N (if X2 is not specified) or M-by-N (if X2 is specified)
%         Gram matrix
%

kernel_type=options.Kernel;
kernel_param=options.KernelParam;

n1=size(X1,1);
if nargin>2
    n2=size(X2,1);
end

if nargin>2
    for l=1:n2
        for m=1:n1
            K(l,m) = kernel(kernel_type, cell2mat(X2(l)), cell2mat(X1(m)), kernel_param);
        end
    end
else
    for l=1:n1
        for m=1:n1
            K(l,m) = kernel(kernel_type, cell2mat(X1(l)), cell2mat(X1(m)), kernel_param);
        end
    end
end



function k = kernel(ker, a, b, sig)
switch lower(ker)
    case 'log'
        k=trace(logm(a)*logm(b));
    case 'rbf'
%         d = norm(logm(inv(a)*b));
        d = norm(logm(b)-logm(a), 'fro');
        k=exp(-d*d/sig);
    case 'line'
        k = norm(a*b);
    case 'polym'
        k = (trace(b'*a) + 1 )^sig;
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
        error('Unknown kernel function.');
end

