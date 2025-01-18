function L = roboSS_loss(u, a, lambda)
    L = zeros(size(u));
    idx = u > 0;
    L(idx) = lambda * (1 - (a * u(idx) + 1) .* exp(-a * u(idx)));
end