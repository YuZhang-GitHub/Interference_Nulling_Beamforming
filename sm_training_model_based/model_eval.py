import os
import torch
import numpy as np
import scipy.io as scio
import time

from surrogate_model import GainPred

cuda = torch.device('cuda')
cpu = torch.device('cpu')

device = cpu

if __name__ == '__main__':

    M = 256
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

    train_num_list = np.linspace(1000, 10000, num=10).astype("int")

    for repeat_idx in range(0, 100):

        for tr_size in train_num_list:

            t1 = time.time()

            model = GainPred(M)

            model_name = './trained_model/model_based_params_M_' + str(M) + '_' + mode + '_tr_size_' + str(tr_size) + '_repeat_' + str(repeat_idx) + '.pt'
            model.load_state_dict(torch.load(model_name, map_location=device))

            model.eval()
            # model.to(cpu)

            batch_size = 100

            batch_per_epoch = np.ceil(np.divide(num_of_sample, batch_size))
            batch_count = 0

            while batch_count < batch_per_epoch:
                start = batch_count * batch_size
                end = np.minimum(start + batch_size, num_of_sample)
                batch_count += 1

                X = x[start:end, :].to(device)
                Y = y[start:end, 0:1].to(device)

                Y_hat = model(X)

                y_pred[start:end, :] = Y_hat

            y_pred_numpy = torch.Tensor.cpu(y_pred.detach()).numpy()

            if not os.path.exists('./pred_results'):
                os.mkdir('./pred_results')

            file_name = './pred_results/model_based_params_M_' + str(M) + '_' + mode + '_tr_size_' + str(tr_size) + '_repeat_' + str(repeat_idx) + '.mat'
            a = scio.savemat(file_name,
                             {'y_true': gain_vec.transpose(),
                              'y_pred': y_pred_numpy})
            # time.sleep(1.0)
            t2 = time.time()
            print("Train size: %d, Repeat: %d, Time: %.5f" % (tr_size, repeat_idx, t2 - t1))
