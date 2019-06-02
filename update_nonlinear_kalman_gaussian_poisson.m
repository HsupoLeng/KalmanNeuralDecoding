function [x_curr, V_curr] = update_nonlinear_kalman_gaussian_poisson(x_prev, y_curr, W, Q, V_prev, C, k)
    x_onestep = (1-k).*x_prev + k*W*erf(x_prev);
    
    x_curr = x_prev; 
    x_temp = inf(size(x_prev));
    while norm(x_curr - x_temp) > 0.01
        x_temp = x_curr;
        y_curr_mean_pred = log(1+ exp(C'*[x_temp;1]));
        x_curr = x_temp + Q\(x_temp - x_onestep) ...
            - C(1:end-1, :)*((y_curr./y_curr_mean_pred).*(exp(y_curr_mean_pred)-1)./exp(y_curr_mean_pred)) ...
            + C(1:end-1, :)*((exp(y_curr_mean_pred)-1)./exp(y_curr_mean_pred));
    end
    
    y_curr_mean_pred = log(1+ exp(C'*[x_curr;1]));
    scale_factor = y_curr.*((1-1./exp(y_curr_mean_pred))./y_curr_mean_pred.^2 + 1./(y_curr_mean_pred.*exp(y_curr_mean_pred))).*exp(y_curr_mean_pred-1)./exp(y_curr_mean_pred);
    V_curr = -pinv(-pinv(Q) + C.*scale_factor*C');
end