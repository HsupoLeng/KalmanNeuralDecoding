function [T_onetrial_nume, T_onetrial_deno, C_onetrial_cross, C_onetrial_auto, R_onetrial_nume, R_onetrial_deno, switch_state_cond_global, observ_likelihood_onetrial] = ...
    run_switching_inference_one_trial(X, Y, T, C_cell, R_cell, Ns, switch_state_cond_global, switch_state_init)
    if isempty(switch_state_cond_global)
        switch_state_cond_causal = zeros(size(X, 1), Ns);

        observ_loglikelihood = zeros(size(X, 1), Ns);
        for j=1:Ns
            observ_loglikelihood(:, j) = logmvnpdf(Y-X*C_cell{j}', zeros(1, size(Y, 2)), R_cell{j}+10^(-3).*eye(size(R_cell{j})));
        end
        switch_state_onestep_pred = zeros(size(X, 1), Ns);
        for t = 1:size(switch_state_cond_causal, 1)
            if t == 1
                switch_state_onestep_pred(t, :) = switch_state_init*T;
            else
                switch_state_onestep_pred(t, :) = switch_state_cond_causal(t-1, :)*T;
            end
            switch_state_cond_causal(t, :) = exp(observ_loglikelihood(t, :)).*switch_state_onestep_pred(t, :);
            switch_state_cond_causal(t, :) = switch_state_cond_causal(t, :)./sum(switch_state_cond_causal(t, :));
        end

        switch_state_cond_global = zeros(size(switch_state_cond_causal));
        switch_state_cond_global(end, :) = switch_state_cond_causal(end, :);
        for t=size(X, 1)-1:-1:1
            switch_state_cond_global(t, :) = switch_state_cond_causal(t, :).*(T*(switch_state_cond_global(t+1, :)./switch_state_onestep_pred(t+1, :))')';
        end
        
        T_onetrial_nume = zeros(size(T));
        for j=1:Ns
            for k=1:Ns
                T_onetrial_nume(j,k) = sum(switch_state_cond_global(1:end-1, j).*switch_state_cond_global(2:end, k));
            end
        end
        %T_onetrial_deno = sum(switch_state_cond_global(1:end-1, :))';
        T_onetrial_deno = sum(T_onetrial_nume, 2);
        
        C_onetrial_cross = zeros(size(Y, 2), size(X, 2), Ns);
        C_onetrial_auto = zeros(size(X, 2), size(X, 2), Ns);
        for j=1:Ns
            C_onetrial_cross(:, :, j) = (Y.*switch_state_cond_global(:,j))'*X;
            C_onetrial_auto(:, :, j) = (X.*switch_state_cond_global(:,j))'*X;
        end
        
        R_onetrial_nume = [];
        R_onetrial_deno = sum(switch_state_cond_global);
        
        observ_likelihood_onetrial = sum(sum(observ_loglikelihood.*switch_state_cond_global, 2));
        
    else
        R_onetrial_nume = zeros(size(Y, 2), size(Y, 2), Ns);
        for j=1:Ns
            R_onetrial_nume(:, :, j) = (Y' - C_cell{j}*X')*(Y.*switch_state_cond_global(:,j));
        end
        
        T_onetrial_nume = [];
        T_onetrial_deno = [];
        C_onetrial_cross = [];
        C_onetrial_auto = [];
        R_onetrial_deno = [];
        switch_state_cond_global = [];
        observ_likelihood_onetrial = nan;
    end
end