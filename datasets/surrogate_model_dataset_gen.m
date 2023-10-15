% clear
% close all
% clc

%% System configuration

M = 8; % number of antennas

num_bit = 3; % analog phase-shifter resolution
ph_conf_set = linspace(-pi, pi-(2*pi/2^num_bit), 2^num_bit);

%% Channel generation

% dir_list = [103.48]; % unit: degree; signal direction
dir_list = [66.4, 82.55]; % unit: degree; interference directions

over_sampling_y = 1000;
My = M;
[F,~] = UPA_codebook_generator(1,My,1,1,over_sampling_y,1,.5); %F: (#ant, #sampled_directions)
theta_s = 0:pi/(over_sampling_y*My):pi-1e-6; %exclude pi

beam_id = floor(dir_list/180*(M*over_sampling_y));

channels = F(:, beam_id);
% plot_pattern(channels) % check the channels

%% Dataset collection

num_of_data_point = 10000;

ph_vec = zeros(M, num_of_data_point);
gain_vec = zeros(1, num_of_data_point);

for ii = 1:num_of_data_point
    
%     ph = -pi + 2*pi*rand(M, 1);
%     ph_vec(:, ii) = ph;
    
    ph = randsample(ph_conf_set, M, true);
    ph_vec(:, ii) = ph;
    
    w = (1/sqrt(M)) * exp(1j * ph.');
    
    gain_vec(ii) = 10*log10(sum(abs(w'*channels).^2));
    
    if mod(ii, 1000) == 0
        fprintf("Dataset generation: %d / %d.\n", ii, num_of_data_point)
    end
    
end

%%

% save(['surrogate_test_dataset_M_', num2str(M), '_interf.mat'], 'ph_vec', 'gain_vec')

