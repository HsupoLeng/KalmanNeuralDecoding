function [A, Q, pi_0, C, R, state_auto_corr, state_onestep_corr, state_auto_corr_full, state_rate_crosscorr] ...
    = train_linear_kalman_params_one_trial(X, Y, A, C)
% Returns MLE estimate of parameters in a linear Kalman filter for one trial using
% Inputs:
% X: state variable matrix, N-by-d. Sample in row, timestamp increasing
% Y: cell matrix, N-by-p. Sample in row, timestamp increasing
% Outputs:
% A: transformation matrix of state model, d-by-d
% Q: noise covariance matrix of state model, d-by-d
% pi_0: mean of initial state, 1-by-d
% V: covariance of initial state, d-by-d
% C: transformation matrix of observation model, p-by-d
% R: noise covariance matrix of observation model, p-by-p
    if isempty(A) && isempty(C)
        state_auto_corr = X(1:end-1,:)'*X(1:end-1,:);
        state_onestep_corr = X(2:end,:)'*X(1:end-1,:);
        A = state_onestep_corr/state_auto_corr; 

        state_auto_corr_full = X'*X;
        state_rate_crosscorr = Y'*X;
        C = state_rate_crosscorr/state_auto_corr_full; 
        pi_0 = X(1, :);
        
        Q=[];R=[];
    else
        state_one_step_pred = X(1:end-1,:)*A';
        state_one_step_diff = X(2:end,:)-state_one_step_pred;
        Q = state_one_step_diff'*state_one_step_diff;

        rate_current_pred = X*C';
        rate_current_diff = Y - rate_current_pred;
        R = rate_current_diff'*rate_current_diff;
        
        A=[];pi_0=[]; C=[];
        state_auto_corr=[]; state_onestep_corr=[];
        state_auto_corr_full=[];state_rate_crosscorr=[];
    end
end