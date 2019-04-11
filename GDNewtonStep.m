function [z_opt] = GDNewtonStep(X, y, beta, eta, T, alpha, lambda, FullHessian)

% Computes gradient descent to retrieve the Newton step
% function is OLS
% T: number of iterations

[~,n] = size(X);
z = rand(n,1);
for (i=1:T)
    g = OLS_gradient(beta, X, y, alpha, lambda);
    z =(eye(n)-FullHessian)*z - eta*g;
end
z_opt = z;
