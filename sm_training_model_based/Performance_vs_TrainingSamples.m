clear
close all
clc

%%

M = 256;
mode = 'interf';

network = 'model'; % model or fc

y_true = load(['./datasets/surrogate_test_dataset_M_', num2str(M), '_', mode, '.mat']);
y_true = y_true.gain_vec;
y_true = y_true.';

%%

tr_size = 1000:1000:10000;
repeat_ind = 0:99;

%%

model_based_pred = cell(1, length(repeat_ind));
fc_based_pred = cell(1, length(repeat_ind));

for jj = 1:length(repeat_ind)
    
    repeat_idx = repeat_ind(jj);
    model_based_pred{jj} = cell(1, length(tr_size));
    fc_based_pred{jj} = cell(1, length(tr_size));
    
    for ii = 1:length(tr_size)
        
        num_sample = tr_size(ii);
        
        if strcmp(network, 'model')
            tmp = load(['./pred_results/model_based_params_M_', num2str(M), '_', mode, '_tr_size_', num2str(num_sample), '_repeat_', num2str(repeat_idx), '.mat']);
            model_based_pred{jj}{ii} = tmp.y_pred;
        else
            tmp = load(['./pred_results/fully_connected_params_M_', num2str(M), '_', mode, '_tr_size_', num2str(num_sample), '_repeat_', num2str(repeat_idx), '.mat']);
            fc_based_pred{jj}{ii} = tmp.y_pred;
        end
        
    end
end

%% MSE calculation

model_based_acc = zeros(length(repeat_ind), length(tr_size));
model_based_acc_NMSE = zeros(length(repeat_ind), length(tr_size));
fc_based_acc = zeros(length(repeat_ind), length(tr_size));

for jj = 1:length(repeat_ind)
    for ii = 1:length(tr_size)
        
        if strcmp(network, 'model')
            model_based_acc(jj, ii) = mean( (model_based_pred{jj}{ii}-y_true).^2 );
            model_based_acc_NMSE(jj, ii) = mean( ((model_based_pred{jj}{ii}-y_true).^2) ./ y_true.^2 );
        else
            fc_based_acc(jj, ii) = mean( (fc_based_pred{jj}{ii}-y_true).^2 );
        end
    end
end

%% plotting

figure(1)

% for jj = 1:length(repeat_ind)
%     if strcmp(network, 'model')
%         p1 = semilogy(tr_size, model_based_acc(jj, :), '-');
%         p2 = semilogy(tr_size, model_based_acc_NMSE(jj, :), '-');
%     else
%         p1 = semilogy(tr_size, fc_based_acc(jj, :), '-');
%     end
%     hold on
% %     p1.MarkerFaceColor = 'w';
% %     p1.Color = [0.00,0.45,0.74];
% %     p1.LineWidth = 1.0;
% end

model_based_acc_db = 10*log10(model_based_acc);
model_based_acc_std = std(model_based_acc_db, 1);
model_based_acc_U = 10*log10(mean(model_based_acc)) + model_based_acc_std;
model_based_acc_L = 10*log10(mean(model_based_acc)) - model_based_acc_std;

x2 = [tr_size, fliplr(tr_size)];
inBetween = [model_based_acc_L, fliplr(model_based_acc_U)];
p = fill(x2, inBetween, 'g');
p.FaceAlpha = 0.3;
p.EdgeColor = "none";
hold on

if strcmp(network, 'model')
    p = plot(tr_size, 10*log10(mean(model_based_acc)), '-s', 'LineWidth', 1.5);
    p.Color = [0.64,0.08,0.18];
    hold on
%     plot(tr_size, 10*log10(mean(model_based_acc_NMSE)), '-ks', 'LineWidth', 2.0);
%     hold on
else
    plot(tr_size, mean(fc_based_acc), '-ks', 'LineWidth', 2.0);
end

% hold on
% 
% p2 = plot(log10(tr_size), 10*log10(fc_based_acc), '-s');
% p2.Color = [0.64,0.08,0.18];
% p2.MarkerFaceColor = 'w';
% p2.LineWidth = 1.0;

% xlim([log10(tr_size(1)), log10(tr_size(end))])
% 
% xticks([2, 3, 4])
% xticklabels({'10^2', '10^3', '10^4', 'Interpreter', 'latex'})

xlabel('Number of training samples', 'FontSize', 12)
ylabel('MSE (dB)', 'FontSize', 12)

% legend('Model-based architecture', ...
%        'Fully-connected layer based architecture', ...
%        'FontSize', 11)

grid on
box on
