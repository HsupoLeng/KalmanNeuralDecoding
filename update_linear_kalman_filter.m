function [x_curr, V_curr, y_likelihood] = update_linear_kalman_filter(x_prev, y_curr, A, Q, V_prev, C, R)
    
    x_one_step = A*x_prev; 
    x_one_step_cov = A*V_prev*A' + Q; 
        
    K = (x_one_step_cov*C')/(C*x_one_step_cov*C'+R);
    x_curr = x_one_step + K*(y_curr - C*x_one_step);
    V_curr = (eye(size(V_prev)) - K*C)*x_one_step_cov;
    
    y_likelihood = mvnpdf((y_curr - C*x_one_step)', zeros(size(y_curr))', C*x_one_step_cov*C'+R);
end