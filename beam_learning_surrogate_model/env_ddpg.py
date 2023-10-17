import os
import torch
import numpy as np


class envCB:

    def __init__(self, num_ant, num_bits, idx, options, model_interf, model_signal, exp_idx):

        self.idx = idx
        self.exp_idx = exp_idx
        self.num_ant = num_ant
        self.num_bits = num_bits
        self.cb_size = 2 ** self.num_bits
        self.codebook = self.codebook_gen()

        # self.ch_t = torch.from_numpy(options['ch_t']).float().cuda()
        # self.ch_i = torch.from_numpy(options['ch_i']).float().cuda()

        self.state = torch.zeros((1, self.num_ant)).float().cuda()
        self.bf_vec = self.init_bf_vec()
        self.previous_gain = 0
        self.previous_gain_pred = 0
        self.th_step = 0.01
        self.threshold = torch.tensor([-100]).float().cuda()
        self.count = 1
        self.record_freq = 10
        self.record_decay_th = 1000
        self.achievement = torch.tensor([0]).float().cuda()
        self.gain_record = [np.array(-100.)]
        self.N_count = 1
        self.best_bf_vec = self.init_best()
        # self.opt_bf_gain()
        self.options = options

        self.model_interf = model_interf
        self.model_signal = model_signal

    def step(self, input_action):  # input_action: (1, num_ant), rep: phase vector
        self.state = input_action
        self.bf_vec = self.phase2bf(self.state)
        reward, bf_gain = self.reward_fn()
        terminal = 0
        return self.state.clone(), reward, bf_gain, terminal

    def reward_fn(self):

        bf_gain_t = self.model_signal(self.state)
        bf_gain_i = self.model_interf(self.state)
        bf_gain = bf_gain_t - bf_gain_i

        if bf_gain > self.previous_gain:
            if bf_gain > self.threshold:
                reward = torch.from_numpy(np.array([1]).reshape((1, 1))).float()
                # reward = bf_gain - self.previous_gain
                # reward = reward.reshape((1, 1))
                self.threshold = self.threshold_modif(self.state, bf_gain)
                # print('threshold reset to: %.1f.' % self.threshold)
            else:
                reward = torch.from_numpy(np.array([1]).reshape((1, 1))).float()
                # reward = bf_gain - self.previous_gain
                # reward = reward.reshape((1, 1))
        else:
            if bf_gain > self.threshold:
                reward = torch.from_numpy(np.array([-1]).reshape((1, 1))).float()
                # reward = bf_gain - self.previous_gain
                # reward = reward.reshape((1, 1))
                self.threshold = self.threshold_modif(self.state, bf_gain)
                # print('threshold reset to: %.1f.' % self.threshold)
            else:
                reward = torch.from_numpy(np.array([-1]).reshape((1, 1))).float()
                # reward = bf_gain - self.previous_gain
                # reward = reward.reshape((1, 1))
        self.previous_gain = self.previous_gain_pred
        return reward, bf_gain

    def get_reward(self, input_action):
        inner_state = input_action

        # Quantization Processing
        # self.options['ph_table_rep'].cuda()
        mat_dist = torch.abs(inner_state.reshape(self.num_ant, 1) - self.options['ph_table_rep'])
        action_quant = self.options['ph_table_rep'][range(self.num_ant), torch.argmin(mat_dist, dim=1)].reshape(1, -1)

        bf_gain_t = self.model_signal(action_quant)
        bf_gain_i = self.model_interf(action_quant)
        bf_gain = bf_gain_t - bf_gain_i

        if bf_gain > self.previous_gain_pred:
            if bf_gain > self.threshold:
                reward = torch.from_numpy(np.array([1]).reshape((1, 1))).float()
                # reward = bf_gain - self.previous_gain_pred
                # reward = reward.reshape((1, 1))
                self.threshold = self.threshold_modif_get_reward(action_quant, bf_gain)
                # print('threshold reset to: %.1f.' % self.threshold)
            else:
                reward = torch.from_numpy(np.array([1]).reshape((1, 1))).float()
                # reward = bf_gain - self.previous_gain_pred
                # reward = reward.reshape((1, 1))
        else:
            if bf_gain > self.threshold:
                reward = torch.from_numpy(np.array([-1]).reshape((1, 1))).float()
                # reward = bf_gain - self.previous_gain_pred
                # reward = reward.reshape((1, 1))
                self.threshold = self.threshold_modif_get_reward(action_quant, bf_gain)
                # print('threshold reset to: %.1f.' % self.threshold)
            else:
                reward = torch.from_numpy(np.array([-1]).reshape((1, 1))).float()
                # reward = bf_gain - self.previous_gain_pred
                # reward = reward.reshape((1, 1))
        # if self.count % self.record_freq == 0:
        #     self.gain_vs_iter()
        #     if self.count == self.record_decay_th:
        #         self.record_freq = 1000
        self.previous_gain_pred = bf_gain
        self.count += 1
        return reward, bf_gain, action_quant.clone(), action_quant.clone()

    def threshold_modif(self, ph_vec, bf_gain):
        self.achievement = bf_gain
        self.gain_recording(ph_vec, self.idx)
        # self.threshold += self.th_step
        self.threshold = bf_gain
        return self.threshold

    def threshold_modif_get_reward(self, inner_bf, bf_gain):
        self.achievement = bf_gain
        self.gain_recording(inner_bf, self.idx)
        # self.threshold += self.th_step
        self.threshold = bf_gain
        return self.threshold

    def phase2bf(self, ph_vec):
        bf_vec = torch.zeros((1, 2 * self.num_ant)).float().cuda()
        for kk in range(self.num_ant):
            bf_vec[0, 2 * kk] = torch.cos(ph_vec[0, kk])
            bf_vec[0, 2 * kk + 1] = torch.sin(ph_vec[0, kk])
        bf_vec = torch.divide(torch.tensor(1).float(), torch.sqrt(torch.tensor(self.num_ant).float())) * bf_vec
        return bf_vec

    def BF_GAIN(self, bf_vec, ch_set):

        bf_r = bf_vec[0, ::2].clone().reshape(1, -1)
        bf_i = bf_vec[0, 1::2].clone().reshape(1, -1)

        ch_r = torch.squeeze(ch_set[:, :self.num_ant].clone())
        ch_i = torch.squeeze(ch_set[:, self.num_ant:].clone())

        bf_gain_1 = torch.matmul(bf_r, torch.t(ch_r))
        bf_gain_2 = torch.matmul(bf_i, torch.t(ch_i))
        bf_gain_3 = torch.matmul(bf_r, torch.t(ch_i))
        bf_gain_4 = torch.matmul(bf_i, torch.t(ch_r))

        bf_gain_r = (bf_gain_1 + bf_gain_2) ** 2
        bf_gain_i = (bf_gain_3 - bf_gain_4) ** 2
        bf_gain_pattern = bf_gain_r + bf_gain_i

        bf_gain = torch.sum(bf_gain_pattern)

        return bf_gain

    def gain_recording(self, bf_vec, idx):
        new_gain = torch.Tensor.cpu(self.achievement).detach().numpy().reshape((1, 1))
        bf_print = torch.Tensor.cpu(bf_vec).detach().numpy().reshape(1, -1)
        if new_gain > max(self.gain_record):
            self.gain_record.append(new_gain)
            self.best_bf_vec = torch.Tensor.cpu(bf_vec).detach().numpy().reshape(1, -1)
            if os.path.exists('beams/beams_' + str(idx) + '_exp_' + str(self.exp_idx) + '_max.txt'):
                with open('beams/beams_' + str(idx) + '_exp_' + str(self.exp_idx) + '_max.txt', 'ab') as bm:
                    np.savetxt(bm, new_gain, fmt='%.2f', delimiter='\n')
                with open('beams/beams_' + str(idx) + '_exp_' + str(self.exp_idx) + '_max.txt', 'ab') as bm:
                    np.savetxt(bm, bf_print, fmt='%.5f', delimiter=',')
            else:
                np.savetxt('beams/beams_' + str(idx) + '_exp_' + str(self.exp_idx) + '_max.txt', new_gain, fmt='%.2f', delimiter='\n')
                # with open('beams/beams_' + str(idx) + '_max.txt', 'ab') as bm:
                #     np.savetxt(bm, new_gain, fmt='%.2f', delimiter='\n')
                with open('beams/beams_' + str(idx) + '_exp_' + str(self.exp_idx) + '_max.txt', 'ab') as bm:
                    np.savetxt(bm, bf_print, fmt='%.5f', delimiter=',')

    def codebook_gen(self):
        angles = np.linspace(0, 2 * np.pi, self.cb_size, endpoint=False)
        cb = np.exp(1j * angles)
        codebook = torch.zeros((self.cb_size, 2))  # shape of the codebook
        for ii in range(cb.shape[0]):
            codebook[ii, 0] = torch.tensor(np.real(cb[ii]))
            codebook[ii, 1] = torch.tensor(np.imag(cb[ii]))
        return codebook

    def init_bf_vec(self):
        bf_vec = torch.empty((1, 2 * self.num_ant))
        bf_vec[0, ::2] = torch.tensor([1])
        bf_vec[0, 1::2] = torch.tensor([0])
        bf_vec = bf_vec.float().cuda()
        return bf_vec

    def init_best(self):
        ph_book = np.linspace(-np.pi, np.pi, 2 ** self.num_bits, endpoint=False)
        ph_vec = np.array([[ph_book[np.random.randint(0, len(ph_book))] for ii in range(self.num_ant)]])
        bf_complex = np.exp(1j * ph_vec)
        bf_vec = np.empty((1, 2 * self.num_ant))
        for kk in range(self.num_ant):
            bf_vec[0, 2 * kk] = np.real(bf_complex[0, kk])
            bf_vec[0, 2 * kk + 1] = np.imag(bf_complex[0, kk])
        return bf_vec
