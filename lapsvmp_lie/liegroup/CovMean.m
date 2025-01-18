
function miu = CovMean(ml, dim, index, n)
    miu = zeros(dim);
    for k=0:n-1
        miu = miu + logm(cell2mat(ml(index+k)));
    end
    miu = miu / n;
    miu = expm(miu);