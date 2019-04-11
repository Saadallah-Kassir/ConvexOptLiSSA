function [X_data, y_data, x_opt] = generateDataset2(m,n)

X_data = (rand(m,n));
x_opt = (rand(n, 1));
n = rand(m, 1);
y_data = (X_data*x_opt + n);

end

