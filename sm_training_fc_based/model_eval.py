import os
import time

import torch
import numpy as np
import scipy.io as scio

from surrogate_model import GainPred_Dense_v2


cuda = torch.device('cuda')
cpu = torch.device('cpu')


if __name__ == '__main__':

    M = 8
    mode = 'interf'

    file_name = './datasets/surrogate_test_dataset_M_' + str(M) + '_' + mode + '.mat'

    data = scio.loadmat(file_name)  # 'gain_vec', 'ph_vec'

    gain_vec = data['gain_vec']
    ph_vec = data['ph_vec']

    # ----------------------------------------------------------------------------- #

    x = ph_vec.transpose()  # shape: (num_of_sample, num_of_ant)
    y = gain_vec.transpose()  # shape: (num_of_sample, 1)

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    # ----------------------------------------------------------------------------- #

    num_of_sample = x.shape[0]

    y_pred = torch.zeros((num_of_sample, 1))

    # train_ratio_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    #                     0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    # ratio_exp_point = np.linspace(-3, -1, num=10, endpoint=True)

    # ratio_exp_point = np.array([-1.])

    # train_ratio_list = 10 ** ratio_exp_point

    train_num_list = np.linspace(1000, 10000, num=10).astype("int")
    # train_num_list = [1000, 3000, 5000, 7000, 9000]

    for repeat_idx in range(10):

        for tr_size in train_num_list:

            # tr_size = int(np.rint(100000 * train_ratio))

            model = GainPred_Dense_v2(M)

            model_name = './trained_model/fully_connected_params_M_' + str(M) + '_' + mode + '_tr_size_' + str(tr_size) + '_repeat_' + str(repeat_idx) + '.pt'
            model.load_state_dict(torch.load(model_name))

            model.eval()
            model.to(cpu)

            batch_size = 100

            batch_per_epoch = np.ceil(np.divide(num_of_sample, batch_size))
            batch_count = 0

            while batch_count < batch_per_epoch:

                start = batch_count * batch_size
                end = np.minimum(start + batch_size, num_of_sample)
                batch_count += 1

                X = x[start:end, :].to(cpu)
                Y = y[start:end, 0:1].to(cpu)

                # Y_hat = 10 * torch.log10(model(X))
                Y_hat = model(X)

                y_pred[start:end, :] = Y_hat

            y_pred_numpy = torch.Tensor.cpu(y_pred.detach()).numpy()

            if not os.path.exists('./pred_results'):
                os.mkdir('./pred_results')

            file_name = './pred_results/fully_connected_params_M_' + str(M) + '_' + mode + '_tr_size_' + str(tr_size) + '_repeat_' + str(repeat_idx) + '.mat'
            scio.savemat(file_name,
                         {'y_true': gain_vec.transpose(),
                          'y_pred': y_pred_numpy})

            # torch.cuda.empty_cache()
            print("Train size: %d; Repeat: %d." % (tr_size, repeat_idx))
            # time.sleep(1.0)
