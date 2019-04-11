close all
clear variables
clc

% Set the size of the data matrix
m = 1000;
n = 50;

% Generating dataset of side m x n, return beta_opt just to compute the
% optimality gap
[X_data, y_data, beta_opt] = generateDataset2(m,n);

T1 = 10; % number of iterations of warm start first order algorithm
x_0 = zeros(n, 1); % initializing initial point

% Fix the regularizer to be 1/m
lambda = 1/m;

% divide our cost function by a constant set to m to ensure the spectral
% norm of the Hessian is less than 1 so we can sue the approximation for
% the inverse matrix.
alpha = m; 

%Compute the full Hessian
FullHessian = (1/m)*(1/alpha)*(X_data'*X_data) + lambda*eye(n);

% Set the number of epochs 
T = 2e2;
% Call Gradient Descent for warm start
[beta, error, error_opt] = GD(X_data, y_data, x_0, 1e-4, T1, alpha, lambda, beta_opt); % GRADIENT DESCENT

% Saving warm up start point for later
beta2 = beta;
beta3 = beta;

for t=1:T
   z_opt  = GDNewtonStep(X_data, y_data, beta, 1 , 5000, alpha, lambda, FullHessian);
   beta = beta + z_opt;
   error = [error; (norm((X_data*beta-y_data),2)/(2*alpha*m) -(norm((X_data*beta_opt-y_data), 2)/(2*alpha*m)))];
   error_opt = [error_opt; norm(beta-beta_opt,2)];
end
%% Newton's Method

% Just initialing the optimality and suboptimality gap vectors for plots
error2 = norm((X_data*beta2-y_data),2)/(2*alpha*m) -(norm((X_data*beta_opt-y_data), 2)/(2*alpha*m));
error2_opt = norm(beta2-beta_opt);
NewtonToc = [];

% Note: Of course for OLS, one can compute the Hessian and its inverse only
% once outside the loop, but this is not general for any cost function.
% Thus, for fairness in the comparison, we compute the Hessian and its
% inverse inside the for loop.

for (i=1:T)
    tic
    eta2 = 1; % skip BTLS and use step size of 1
    
    % Compute Hessian & its inverse
    
    HessianInv = FullHessian^(-1);
    
    % Update beta
    beta2 = beta2-eta2*HessianInv*OLS_gradient(beta2, X_data, y_data, alpha, lambda);
    
    % Save the gaps and time per iteration
    error2 = [error2; (norm((X_data*beta2-y_data),2)/(2*alpha*m) - (norm((X_data*beta_opt-y_data), 2)/(2*alpha*m)))];
    error2_opt = [error2_opt; norm(beta2-beta_opt)];
    NewtonToc = [toc NewtonToc];
end
NewtonToc = mean(NewtonToc);

%% Vanilla GD, just of the sake of the comparison
[beta3, error3, error3_opt] = GD(X_data, y_data, beta3, 4, T, alpha, lambda, beta_opt); % GRADIENT DESCENT

%% Plots

figure()
hold on
plot(error2_opt, 'LineWidth', 2);
plot(error_opt(T1:end), 'LineWidth', 2);
plot(error3_opt, 'LineWidth', 2);
legend('Newton', 'LISSA', 'GD')
title('Optimality Gap')
set(gca, 'YScale', 'log')
ylabel('$||x-x^*||_2$', 'interpreter', 'latex')
xlabel('Epochs')
grid on
hold off

figure()
hold on
plot(error2, 'LineWidth', 2);
plot(error(T1:end), 'LineWidth', 2);
plot(error3, 'LineWidth', 2);
legend('Newton', 'LISSA', 'GD')
title('Subptimality Gap')
set(gca, 'YScale', 'log')
ylabel('$f(x)-f(x^*)$', 'interpreter', 'latex')
xlabel('Epochs')
grid on
hold off
