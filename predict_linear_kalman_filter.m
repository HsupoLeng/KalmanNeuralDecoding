function [X_estimate, X_estimate_var, X_mse] = predict_linear_kalman_filter(X, Y, A, Q, pi_0, V, C, R)
    X_estimate = zeros(size(X));
    X_estimate_var = zeros([size(X,1), size(A)]);
    
    x_prev = [X(1,[1,2]), pi_0([3,4])]';
    V_prev = zeros(size(V));
    
    X_estimate(1,:) = x_prev;
    X_estimate_var(1,:,:) = V; 
    for i=2:size(X,1)
        y_curr = Y(i, :)';
        
        [x_curr, V_curr, ~] = update_linear_kalman_filter(x_prev, y_curr, A, Q, V_prev, C, R);
        X_estimate(i, :) = x_curr';
        X_estimate_var(i, :, :) = V_curr;
        
        x_prev = x_curr;
        V_prev = V_curr;
    end
    
    X_mse = mean(sum((X(:,1:2) - X_estimate(:,1:2)).^2, 2));
end