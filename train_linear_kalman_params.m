function [A, Q, pi_0, V, C, R] = train_linear_kalman_params(X, Y)
% Returns MLE estimate of parameters in a linear Kalman filter using
% Inputs:
% X: cell matrix holding state variable matrix for each trial. 
% Each one is N-by-d. Sample in row, timestamp increasing
% If X has known M categories (e.g. reaching directions), they should be put
% in columns of the cell matrix, i.e. X is K-by-M. 
% Y: cell matrix holding observation matrix for each trial, 
% Each one is N-by-p. Sample in row, timestamp increasing
% Outputs:
% A: transformation matrix of state model, d-by-d
% Q: noise covariance matrix of state model, d-by-d
% pi_0: mean of initial state, d-by-1, or d-by-M
% V: covariance of initial state, d-by-d, or d-by-d-by-M
% C: transformation matrix of observation model, p-by-d
% R: noise covariance matrix of observation model, p-by-p

    [~, ~, pi_0_cell, ~, ~, ...
        state_auto_corr_cell, state_onestep_corr_cell, state_auto_corr_full_cell, state_rate_crosscorr_cell] = ...
        cellfun(@(X_onetrial,Y_onetrial) train_linear_kalman_params_one_trial(X_onetrial, Y_onetrial, [], []), X, Y, ...
        'UniformOutput', false);
    
    total_num_of_timestamps = sum(sum(cellfun(@(x) size(x,1), X)));
    A = sum(cat(3,state_onestep_corr_cell{:}),3)/sum(cat(3,state_auto_corr_cell{:}),3);
    C = sum(cat(3, state_rate_crosscorr_cell{:}),3)/sum(cat(3, state_auto_corr_full_cell{:}),3); 
    
    pi_0_all = cat(3, pi_0_cell{:});
    pi_0 = mean(reshape(pi_0_all, size(pi_0_all,1), size(pi_0_all,2), size(pi_0_cell,1), size(pi_0_cell,2)), 3);
    pi_0 = squeeze(pi_0);
    if size(pi_0,1) == 1
        pi_0 = pi_0';
    end
    V = cov(squeeze(pi_0_all)', 1);
    
    [~, Q_cell, ~, ~, R_cell, ~, ~, ~, ~] = ...
        cellfun(@(X_onetrial,Y_onetrial) train_linear_kalman_params_one_trial(X_onetrial, Y_onetrial, A, C), X, Y, ...
        'UniformOutput', false);
    Q = sum(cat(3, Q_cell{:}), 3)./(total_num_of_timestamps-numel(X)); 
    R = sum(cat(3, R_cell{:}), 3)./total_num_of_timestamps;
end