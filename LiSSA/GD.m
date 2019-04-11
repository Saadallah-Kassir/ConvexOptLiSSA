function [beta_opt, error, error_opt] = GD(X, y, beta, eta, T, alpha, lambda, x_opt)

% Computes gradient descent 
% function is OLS
% T: number of iterations

m = length(y);

error = [(norm((X*beta-y),2)/(2*alpha*m) - (norm((X*x_opt-y), 2)/(2*alpha*m)))];
error_opt = [norm(beta-x_opt, 2)];

for (i=1:T)
    g = OLS_gradient(beta, X, y, alpha, lambda);
    beta = beta - eta*g;
    error = [error; norm((X*beta-y),2)/(2*alpha*m) - (norm((X*x_opt-y), 2)/(2*alpha*m))];
    error_opt = [error_opt; norm(beta-x_opt, 2)];
end
beta_opt = beta;
