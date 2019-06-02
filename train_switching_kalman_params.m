function [A, Q, pi_0, V, T, C_cell, R_cell, switch_state_init, observ_likelihood_history] = train_switching_kalman_params(X, Y, Ns)
    total_num_of_timestamps = sum(sum(cellfun(@(x) size(x,1), X)));
     
    % Compute A and Q using the same approach as in linear Kalman filter
    state_auto_corr_cell = cellfun(@(X_onetrial) X_onetrial(1:end-1, :)'*X_onetrial(1:end-1, :), X, 'UniformOutput', false);
    state_onestep_corr_cell = cellfun(@(X_onetrial) X_onetrial(2:end, :)'*X_onetrial(1:end-1, :), X, 'UniformOutput', false);
    A = sum(cat(3, state_onestep_corr_cell{:}), 3)/sum(cat(3, state_auto_corr_cell{:}), 3);
    
    state_onestep_diff_cell = cellfun(@(X_onetrial) X_onetrial(2:end,:) - X_onetrial(1:end-1,:)*A', X, 'UniformOutput', false);
    Q_cell = cellfun(@(state_onestep_diff) state_onestep_diff'*state_onestep_diff, state_onestep_diff_cell, 'UniformOutput', false);
    Q = sum(cat(3, Q_cell{:}), 3)./(total_num_of_timestamps - numel(X));
    
    % Compute pi_0 and V using the same approach as in linear Kalman filter
    pi_0_cell = cellfun(@(X_onetrial) X_onetrial(1, :), X, 'UniformOutput', false);
    pi_0_all = cat(3, pi_0_cell{:});
    pi_0 = mean(reshape(pi_0_all, size(pi_0_all,1), size(pi_0_all,2), size(pi_0_cell,1), size(pi_0_cell,2)), 3);
    pi_0 = squeeze(pi_0);
    if size(pi_0,1) == 1
        pi_0 = pi_0';
    end
    V = cov(squeeze(pi_0_all)', 1);
    
    % Compute discrete latent state transition matrix T, 
    % kinematic state-neural activity transform matrices C_cell, one for each switching state
    % and corresponding covariance matrices R_cell, one for each switching
    % state
    T_prev = ones(Ns, Ns)./Ns;
    C_curr = cell(Ns, 1);
    R_curr = cell(Ns, 1);
    T_curr = rand(Ns, Ns);
    T_curr = T_curr./sum(T_curr, 2);
    C_init = (vertcat(X{:})\vertcat(Y{:}))';
    R_init = cov(vertcat(Y{:}));
    for j=1:Ns
        C_curr{j} = C_init + mean(C_init(:)).*rand(size(Y{1}, 2), size(X{1}, 2));
        %R_curr{j} = eye(size(Y{1}, 2));
        R_curr{j} = R_init + mean(diag(R_init)).*diag(rand(1, size(R_init, 1)));
    end
 
    iter = 1;
    observ_likelihood_prev = -inf;
    observ_likelihood_curr = -realmax;
    observ_likelihood_history = [];
    switch_state_init = ones(1, Ns)./Ns;
    while(observ_likelihood_curr - observ_likelihood_prev > 10^(-3)) 
        observ_likelihood_prev = observ_likelihood_curr;
        T_prev = T_curr;
        C_prev = C_curr;
        R_prev = R_curr;
        func_h1 = @(X_onetrial, Y_onetrial) run_switching_inference_one_trial(X_onetrial, Y_onetrial, T_prev, C_prev, R_prev, Ns, [], switch_state_init);
        [T_nume_cell, T_deno_cell, C_cross_cell, C_auto_cell, ~, R_deno_cell, switch_state_cond_global_cell, observ_likelihood_cell] = ...
            cellfun(func_h1, X, Y, 'UniformOutput', false);
        T_curr = sum(cat(3, T_nume_cell{:}), 3)./sum(horzcat(T_deno_cell{:}), 2);

        switch_state_init_cell = cellfun(@(x) x(1, :), switch_state_cond_global_cell, 'UniformOutput', false);
        switch_state_init = mean(vertcat(switch_state_init_cell{:}));
        C_cross_alltrial = sum(cat(4, C_cross_cell{:}), 4);
        C_auto_alltrial = sum(cat(4, C_auto_cell{:}), 4);
        for j=1:Ns
            C_curr{j} = C_cross_alltrial(:, :, j)/C_auto_alltrial(:, :, j);
        end
        R_deno_alltrial = sum(vertcat(R_deno_cell{:}));
        func_h2 = @(X_onetrial, Y_onetrial, switch_state_cond_global) run_switching_inference_one_trial(X_onetrial, Y_onetrial, T_curr, C_curr, R_prev, Ns, switch_state_cond_global, switch_state_init);
        [~, ~, ~, ~, R_nume_cell, ~, ~] = ...
            cellfun(func_h2, X, Y, switch_state_cond_global_cell, 'UniformOutput', false);
        R_nume_alltrial = sum(cat(4, R_nume_cell{:}), 4);
        for j=1:Ns
            R_curr{j} = R_nume_alltrial(:, :, j)./R_deno_alltrial(j);
        end
        observ_likelihood_curr = sum([observ_likelihood_cell{:}]);
        observ_likelihood_history(iter) = observ_likelihood_curr;
        fprintf('Iteration %d: total observation likelihood is %f\n', iter, observ_likelihood_curr);
        iter = iter + 1;
    end
    T = T_curr; 
    C_cell = C_curr;
    R_cell = R_curr;
end