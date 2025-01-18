function D = geodesic_distance(A,B)
%
%      D = geodesic_distance(A,B)
%      
%      A: M-by-1 matrix of M k-by-k-dimensional vectors 
%      B: N-by-1 matrix of N k-by-k-dimensional vectors
% 
%      D: M-by-N distance matrix
%


if (size(A,2) ~= size(B,2))
    error('A and B must be of same dimensionality.');
end

m= size(A,1);
n=size(B,1);


% 计算李代数上的测地线距离
D = zeros(m, n);
for i = 1:m
    for j = 1:n
        x = cell2mat(A(i));
        y = cell2mat(B(j));
        D(i, j) = norm(logm(y)-logm(x), 'fro');

    end
end
end