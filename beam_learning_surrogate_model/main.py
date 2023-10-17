import os
import torch
import numpy as np
import argparse

from train_ddpg import train
from DataPrep import load_ch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(dest='exp_id')

    args = parser.parse_args()

    options = {
        'gpu_idx': 0,
        'num_ant': 8,
        'num_bits': 3,
        'pf_print': 100,

        'path_target': './user_target_4600_new.mat',
        'path_interf_1': './user_interf_3670_new.mat',
        'path_interf_2': './user_interf_2952_new.mat',

        'save_freq': 50000
    }

    train_opt = {
        'state': 0,
        'best_state': 0,
        'num_iter': 1000,
        'tau': 1e-2,
        'overall_iter': 1,
        'replay_memory': [],
        'replay_memory_size': 1024,
        'minibatch_size': 512,
        'gamma': 0
    }

    if not os.path.exists('beams/'):
        os.mkdir('beams/')

    if not os.path.exists('pfs/'):
        os.mkdir('pfs/')

    ch_t = load_ch(options['path_target'], 'user_t')  # numpy.ndarray: (#users, 2*#ant)
    ch_i_1 = load_ch(options['path_interf_1'], 'user_i')
    ch_i_2 = load_ch(options['path_interf_2'], 'user_i_2')

    ch_i = np.concatenate((ch_i_1, ch_i_2), axis=0)

    options['ch_t'] = ch_t
    options['ch_i'] = ch_i

    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------

    # ch = ch[:1, :]

    # Quantization settings
    options['num_ph'] = 2 ** options['num_bits']
    options['multi_step'] = torch.from_numpy(
        np.linspace(int(-(options['num_ph'] - 2) / 2),
                    int(options['num_ph'] / 2),
                    num=options['num_ph'],
                    endpoint=True)).type(dtype=torch.float32).reshape(1, -1)
    options['pi'] = torch.tensor(np.pi)
    options['ph_table'] = (2 * options['pi']) / options['num_ph'] * options['multi_step']
    options['ph_table_rep'] = options['ph_table'].repeat(options['num_ant'], 1)

    train(options, train_opt, int(args.exp_id), int(args.exp_id))
