function [W, Q, pi_0, V, C] = train_gaussian_state_poisson_rate_params(X, Y, k)
% Returns MLE estimate of parameters in a state space model with 
% the state model being nonlinear Gaussian: x_t ~ N(f(x_{t-1}), Q),
% f(x) = (1-k)x + k*W*erf(x)
% the observation model being Poisson: y_t^i ~ Poisson(\lambda_i(x_t)\delta t), 
% \lambda_i(x_t) = log(1+e^{c_i^T*x+d_i}) using
% Inputs:
% X: cell matrix holding state variable matrix for each trial. 
% Each one is N-by-d. Sample in row, timestamp increasing
% If X has known M categories (e.g. reaching directions), they should be put
% in columns of the cell matrix, i.e. X is K-by-M. 
% Y: cell matrix holding observation matrix for each trial, 
% Each one is N-by-p. Sample in row, timestamp increasing
% k: memory factor in the state model, scaler
% Outputs:
% W: transformation matrix of the nonlinear state model, d-by-d
% Q: noise covariance matrix of state model, d-by-d
% pi_0: mean of initial state, d-by-1, or d-by-M
% V: covariance of initial state, d-by-d, or d-by-d-by-M
% C: transformation matrix of observation model, p-by-(d+1)
    X_erf = cellfun(@(X_one_trial) erf(X_one_trial(1:end-1, :)), X, 'UniformOutput', false);
    X_X_erf_onestep_corr = cellfun(@(X_one_trial, X_erf_one_trial) X_one_trial(2:end, :)'*X_erf_one_trial, X, X_erf, 'UniformOutput', false);
    X_X_erf_auto_corr = cellfun(@(X_one_trial, X_erf_one_trial) X_one_trial(1:end-1, :)'*X_erf_one_trial, X, X_erf, 'UniformOutput', false);
    X_erf_X_erf_auto_corr = cellfun(@(X_erf_one_trial) X_erf_one_trial'*X_erf_one_trial, X_erf, 'UniformOutput', false);
    W = (sum(cat(3, X_X_erf_onestep_corr{:}), 3)./k + sum(cat(3, X_X_erf_auto_corr{:}), 3).*((1-k)/k))/sum(cat(3, X_erf_X_erf_auto_corr{:}), 3);
    
    num_of_timestamps = sum(cellfun(@(X_one_trial) size(X_one_trial, 1), X));
    X_diff = cellfun(@(X_one_trial) X_one_trial(2:end, :) - ((1-k).*X_one_trial(1:end-1, :) + k*erf(X_one_trial(1:end-1, :))*W'), X, 'UniformOutput', false);
    X_cov = cellfun(@(X_diff_one_trial) X_diff_one_trial'*X_diff_one_trial, X_diff, 'UniformOutput', false);
    Q = sum(cat(3, X_cov{:}), 3)./(num_of_timestamps - numel(X));
    
    X_initial = cellfun(@(X_one_trial) X_one_trial(1, :), X, 'UniformOutput', false);
    X_initial = vertcat(X_initial{:});
    pi_0 = mean(X_initial);
    V = cov(X_initial);
    
    X_augmented = [cat(1, X{:}), ones(num_of_timestamps, 1)];
    Y_cell_by_dimension = num2cell(cat(1, Y{:}), 1);
    %C_cell = cellfun(@(Y_one_dim) X_augmented\log(exp(Y_one_dim)-1), Y_cell_by_dimension, 'UniformOutput', false);
    C_cell = cellfun(@(Y_one_dim) X_augmented\log(exp(Y_one_dim)), Y_cell_by_dimension, 'UniformOutput', false);
    C = cat(2, C_cell{:});
end