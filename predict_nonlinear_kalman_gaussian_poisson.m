function [X_estimate, X_estimate_var, X_mse] = predict_nonlinear_kalman_gaussian_poisson(X, Y, W, Q, pi_0, V, C, k)
    X_estimate = zeros(size(X));
    X_estimate_var = zeros([size(X,1), size(Q)]);
    
    x_prev = X(1,:)';
    V_prev = zeros([size(V,1), size(V,2)]);

    X_estimate(1,:) = x_prev;
    X_estimate_var(1,:,:) = V; 
    for i=2:size(X,1)
        y_curr = Y(i, :)';
        
        [x_curr, V_curr] = update_nonlinear_kalman_gaussian_poisson(x_prev, y_curr, W, Q, V_prev, C, k);
        X_estimate(i, :) = x_curr';
        X_estimate_var(i, :, :) = V_curr;
        
        x_prev = x_curr;
        V_prev = V_curr;
    end
    
    X_mse = mean(sum((X(:,1:2) - X_estimate(:,1:2)).^2, 2));
end