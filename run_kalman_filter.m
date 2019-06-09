function [X_mse_mean, X_estimate_cell, X_estimate_var_cell, X_mse_cell, switch_state_prob_cell, observ_likelihood_history] = ...
    run_kalman_filter(train_state, train_observ, test_state, test_observ, filter_str, Ns, rng_seed)
        if strcmp(filter_str, 'switching_kalman_gaussian_mixture_gaussian')
            if ~isempty(rng_seed)
                rng(rng_seed);
            end
            [A, Q, pi_0, V, T, C_cell, R_cell, switch_state_init, observ_likelihood_history] = train_switching_kalman_params(train_state, train_observ, Ns);
            [X_estimate_cell, X_estimate_var_cell, X_mse_cell, switch_state_prob_cell] = ...
                    cellfun(@(X_onetrial,Y_onetrial) ...
                    predict_switching_kalman_filter(X_onetrial, Y_onetrial, A, Q, pi_0', V, T, C_cell, R_cell, switch_state_init, Ns), test_state, test_observ, ...
                    'UniformOutput', false);

            
%             for i=1:length(C_cell)
%                 figure();
%                 imagesc(C_cell{i});
%                 colorbar;
%                 title(sprintf('Transformation matrix C for switching state %d', i));
%                 xticks(1:4);
%                 xticklabels({'x', 'y', 'v_x', 'v_y'});
%                 xlabel('Kinematic states');
%                 ylabel('Neural activity principal directions');
%             end

        elseif strcmp(filter_str, 'nonlinear_kalman_gaussian_poisson')
            memory_factor = 0.1;
            [W, Q, pi_0, V, C] = train_gaussian_state_poisson_rate_params(train_state, train_observ, memory_factor);
            [X_estimate_cell, X_estimate_var_cell, X_mse_cell] = ...
                    cellfun(@(X_onetrial,Y_onetrial) ...
                    predict_nonlinear_kalman_gaussian_poisson(X_onetrial, Y_onetrial, W, Q, pi_0', V, C, memory_factor), test_state, test_observ, ...
                    'UniformOutput', false);

        elseif strcmp(filter_str, 'linear_kalman_gaussian_gaussian')
            [A, Q, pi_0, V, C, R] = train_linear_kalman_params(train_state, train_observ);
            [X_estimate_cell, X_estimate_var_cell, X_mse_cell] = ...
                    cellfun(@(X_onetrial,Y_onetrial) ...
                    predict_linear_kalman_filter(X_onetrial, Y_onetrial, A, Q, pi_0', V, C, R), test_state, test_observ, ...
                    'UniformOutput', false);
                
            switch_state_prob_cell = {};
            observ_likelihood_history = [];

        else % Default to linear Gaussian-Gaussian Kalman filter
            [A, Q, pi_0, V, C, R] = train_linear_kalman_params(train_state, train_observ);
            [X_estimate_cell, X_estimate_var_cell, X_mse_cell] = ...
                    cellfun(@(X_onetrial,Y_onetrial) ...
                    predict_linear_kalman_filter(X_onetrial, Y_onetrial, A, Q, pi_0', V, C, R), test_state, test_observ, ...
                    'UniformOutput', false);
                
            switch_state_prob_cell = {};
            observ_likelihood_history = [];
        end
        
        X_mse_mean = mean(cell2mat(X_mse_cell(:)));
end

