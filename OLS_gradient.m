function [g] = OLS_gradient(beta, X, y, alpha, lambda)
% Computes the gradient of OLS cost function, integrating alpha into the
% function to ensure than the Hessian spectral norm is less than 1.

    m = length(y);
    g = (1/(m*alpha))*(X'*(X*beta-y)) + beta*lambda;

end

