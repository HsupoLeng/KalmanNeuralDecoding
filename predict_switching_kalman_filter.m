function [X_estimate, X_estimate_var, X_mse, switch_state_prob_all] = predict_switching_kalman_filter(X, Y, A, Q, pi_0, V, T, C_cell, R_cell, switch_state_init, Ns)
    X_estimate = zeros(size(X));
    X_estimate_var = zeros([size(X,1), size(A)]);
    
    X_estimate(1,:) = X(1,:)';
    X_estimate_var(1,:,:) = V;
    
    x_estimate_switch_collapse = cell(Ns, 1);
    V_estimate_switch_collapse = cell(Ns, 1);
    for j=1:Ns
        x_estimate_switch_collapse{j} = [X(1,[1,2]), pi_0([3,4])]';
        V_estimate_switch_collapse{j} = zeros(size(V));
    end
    
    x_estimate_switch_pair = cell(Ns, Ns);
    V_estimate_switch_pair = cell(Ns, Ns);
    y_likelihood_switch_pair = zeros(Ns, Ns);
    
    switch_state_prob = switch_state_init;
    switch_state_prob_all = zeros(size(X, 1), length(switch_state_prob));
    switch_state_prob_all(1, :) = switch_state_prob; 
    for t=2:size(X,1)
        y_curr = Y(t, :)';
        
        % Propagate from each possible prev. state to each current state 
        for i=1:Ns
            for j=1:Ns
                x_prev = x_estimate_switch_collapse{i};
                V_prev = V_estimate_switch_collapse{i};
                
                [x_curr, V_curr, y_likelihood] = update_linear_kalman_filter(x_prev, y_curr, A, Q, V_prev, C_cell{j}, R_cell{j});
                
                x_estimate_switch_pair{i, j} = x_curr;
                V_estimate_switch_pair{i, j} = V_curr;
                y_likelihood_switch_pair(i, j) = y_likelihood;
            end
        end
        
        % Compute switching state conditional probabilities
        switch_state_pair_prob = y_likelihood_switch_pair.*T.*(switch_state_prob)';
        switch_state_pair_prob = switch_state_pair_prob./sum(switch_state_pair_prob(:));
        switch_state_prob = sum(switch_state_pair_prob);
        switch_state_backward_prob = switch_state_pair_prob./(switch_state_prob+eps);
        switch_state_prob_all(t, :) = switch_state_prob; 
        
        % Collapse all Gaussians with the same current state by moment
        % matching up to the 2nd order
        for j=1:Ns
            x_estimate_switch_collapse{j} = sum(horzcat(x_estimate_switch_pair{:, j}).*switch_state_backward_prob(:,j)', 2);
            V_temp_cell = cellfun(@(x, V, w) w.*(V+(x_estimate_switch_collapse{j}-x)*(x_estimate_switch_collapse{j}-x)'), ...
                x_estimate_switch_pair(:, j), V_estimate_switch_pair(:, j), num2cell(switch_state_backward_prob(:, j)), ...
                'UniformOutput', false);
            V_estimate_switch_collapse{j} = sum(cat(3, V_temp_cell{:}), 3);
            if any(isnan(x_estimate_switch_collapse{j}))
                pause(1);
            end
        end
        
        X_estimate(t, :) = sum(horzcat(x_estimate_switch_collapse{:}).*switch_state_prob, 2)';
        V_temp_cell = cellfun(@(x, V, w) w.*(V+(X_estimate(t, :)'-x)*(X_estimate(t, :)'-x)'), ...
                x_estimate_switch_collapse, V_estimate_switch_collapse, num2cell(switch_state_prob'), ...
                'UniformOutput', false);
        X_estimate_var(t, :, :) = sum(cat(3, V_temp_cell{:}), 3);
    end
    
    X_mse = mean(sum((X(:,1:2) - X_estimate(:,1:2)).^2, 2));
end