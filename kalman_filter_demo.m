close all;
load('MM_S1_processed.mat');

%% Prepare training and test data
train_size = uint16(0.8*length(Data.kinematics));
train_state = Data.kinematics(1:train_size);
test_state = Data.kinematics(train_size+1:end);
train_M1 = Data.neural_data_M1(1:train_size);
test_M1 = Data.neural_data_M1(train_size+1:end);
train_PMd = Data.neural_data_PMd(1:train_size);
test_PMd = Data.neural_data_PMd(train_size+1:end);

target_on_idx = unique(cellfun(@find,Data.target_on));
% time_lag_options = 0:10:200; Optimal lag is 150~170ms, tested with M1 data
lag_idx_delta = uint8(150/10);
state_mask_str = 'position_and_velocity';
if strcmp(state_mask_str, 'position')
    state_mask = 1:2;
elseif strcmp(state_mask_str, 'all') % Position, velocity and acceleration
    state_mask = 1:6;
else
    state_mask = 1:4;
end

train_state_motion = cellfun(@(s) s(target_on_idx+lag_idx_delta:end, state_mask), train_state, 'UniformOutput', false);
test_state_motion = cellfun(@(s) s(target_on_idx+lag_idx_delta:end, state_mask), test_state, 'UniformOutput', false);
train_M1_motion = cellfun(@(s) s(:,target_on_idx:end-lag_idx_delta)', train_M1, 'UniformOutput', false);
test_M1_motion = cellfun(@(s) s(:,target_on_idx:end-lag_idx_delta)', test_M1, 'UniformOutput', false);
train_PMd_motion = cellfun(@(s) s(:,target_on_idx:end)', train_PMd, 'UniformOutput', false);
test_PMd_motion = cellfun(@(s) s(:,target_on_idx:end)', test_PMd, 'UniformOutput', false);

% Reduce spike count to 50 dimensions, explaining approx. 92% of variability
coeffs = pca(vertcat(train_M1_motion{:}), 'NumComponents', 50); 
train_M1_motion = cellfun(@(data) data*coeffs, train_M1_motion, 'UniformOutput', false);
test_M1_motion = cellfun(@(data) data*coeffs, test_M1_motion, 'UniformOutput', false);
%{
[~, psi, ~, ~, factoran_preds] = factoran(vertcat(train_M1_motion{:}, test_M1_motion{:}), 20, 'maxit', 100000);
train_row_dist = cellfun(@(x) size(x,1), train_M1_motion);
test_row_dist = cellfun(@(x) size(x,1), test_M1_motion); 
train_M1_motion = mat2cell(factoran_preds(1:sum(train_row_dist), :), train_row_dist);
test_M1_motion = mat2cell(factoran_preds(end-sum(test_row_dist)+1:end, :), test_row_dist);
%}
%% Apply Kalman filter
filter_str = 'switching_kalman_gaussian_mixture_gaussian';
Ns = 4; 
[X_rms_mean, X_estimate_cell, X_estimate_var_cell, X_mse_cell, switch_state_prob_cell, observ_likelihood_history] = run_kalman_filter(train_state_motion, train_M1_motion, test_state_motion, test_M1_motion, filter_str, Ns, 0);

%% Visualize dataset and the prediction result
figure();
hold on;
for i=1:size(train_state_motion,1)
    plot(train_state_motion{i}(:,1), train_state_motion{i}(:,2));
end
hold off;

trial_to_visualize = 3;
X_estimate_linear = X_estimate_cell_linear{trial_to_visualize};
X_estimate_var_linear = X_estimate_var_cell_linear{trial_to_visualize};
X_estimate_switching_4 = X_estimate_cell{trial_to_visualize};
X_estimate_var_switching_4 = X_estimate_var_cell{trial_to_visualize};
switch_state_prob = switch_state_prob_cell{trial_to_visualize};
figure();
hold on;
%plot(X_estimate_linear(:,1),X_estimate_linear(:,2), 'b.-');
color_opts = colormap('lines');
for t=1:size(X_estimate_switching_4, 1)-1
    [~, switch_state] = max(switch_state_prob(t, :));
    plot(X_estimate_switching_4(t:t+1,1),X_estimate_switching_4(t:t+1,2), '.-', 'Color', color_opts(switch_state, :), 'LineWidth', 3);
end
%for i=2:size(X_estimate_linear,1)
    %plot_gaussian_ellipsoid(X_estimate_linear(i,[1,2]), squeeze(X_estimate_var_linear(i,[1,2],[1,2])));
    %plot_gaussian_ellipsoid(X_estimate_switching_4(i,[1,2]), squeeze(X_estimate_var_switching_4(i,[1,2],[1,2])));
%end
plot(test_state_motion{trial_to_visualize}(:,1), test_state_motion{trial_to_visualize}(:,2), 'k.-');
hold off;
xlabel('x (cm)');
ylabel('y (cm)');
%legend('Switching Kalman with N_s = 4', 'Groundtruth');

figure();
hold on; 
for j=1:size(switch_state_prob, 2)
    plot(switch_state_prob(:, j));
end
hold off; 
ylim([0, 1.2]);
xlabel('Time (s)');
xticklabels(cellstr(num2str(xticks'.*10./1000)));
ylabel('Switching state posterior probability');
legend({'S1', 'S2', 'S3', 'S4'});

for i=1:length(C_cell)
    figure();
    imagesc(C_cell{i});
    colorbar;
    title(sprintf('Transformation matrix C for switching state %d', i));
    xticks(1:4);
    xticklabels({'x', 'y', 'v_x', 'v_y'});
    xlabel('Kinematic states');
    ylabel('Neural activity principal directions');
end

%% Repeat tests for switching Kalman
num_of_switch_state_opts = 2.^(1:3);
repeat_em = 10; 
filter_str = 'switching_kalman_gaussian_mixture_gaussian';

mse_mean_mat = zeros(repeat_em, length(num_of_switch_state_opts));
observ_likelihood_history_mat = cell(repeat_em, length(num_of_switch_state_opts));
for i=1:length(num_of_switch_state_opts)
    for j=1:repeat_em
        [X_mse_mean, X_estimate_cell, X_estimate_var_cell, X_mse_cell, switch_state_prob_cell, observ_likelihood_history] = run_kalman_filter(train_state_motion, train_M1_motion, test_state_motion, test_M1_motion, filter_str, ...
            num_of_switch_state_opts(i), j);
        mse_mean_mat(j, i) = X_mse_mean; 
        observ_likelihood_history_mat{j, i} = observ_likelihood_history;
    end
end
save('switching_kalman_test_result.mat', 'mse_mean_mat', 'observ_likelihood_history_mat');
%%
color_opts = colormap('lines');
figure();
hold on;
errorbar(1:size(mse_mean_mat, 2), mean(mse_mean_mat), ...
    mean(mse_mean_mat) - min(mse_mean_mat, [], 1), ...
    max(mse_mean_mat, [], 1) - mean(mse_mean_mat), 'LineWidth', 3);
p_obj = line([0.5, 3.5], [4.6894, 4.6894], 'Color', 'red', 'LineStyle', '--', 'LineWidth', 3);
xlim([0.5, 3.5]);
ylim([4.3, 4.8]);
xticks(1:3);
xticklabels({'2,', '4', '8'});
xlabel('Number of switching Kalman latent states');
ylabel('MSE (cm^2)');
legend(p_obj, 'Linear Kalman baseline');

p_objs = cell(size(observ_likelihood_history_mat, 2));
figure(); 
hold on; 
for i=1:size(observ_likelihood_history_mat, 2)
    for j=1:size(observ_likelihood_history_mat, 1)
        p_obj = plot(observ_likelihood_history_mat{j, i}, 'Color', color_opts(i, :), 'LineWidth', 2);
    end
    p_objs{i} = p_obj;
end
hold off;
legend([p_objs{end:-1:1}], {'N_s=8', 'N_s=4', 'N_s=2'});