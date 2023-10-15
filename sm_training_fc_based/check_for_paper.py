import torch
import numpy as np
import scipy.io as scio

from surrogate_model import GainPred_Dense_v2


def model_pre_eval(model, data_path):

    data = scio.loadmat(data_path)  # 'gain_vec', 'ph_vec'

    gain_vec = data['gain_vec']
    ph_vec = data['ph_vec']

    # ----------------------------------------------------------------------------- #

    x = ph_vec.transpose()  # shape: (num_of_sample, num_of_ant)
    y = gain_vec.transpose()  # shape: (num_of_sample, 1)

    y_true_numpy = y

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    # ----------------------------------------------------------------------------- #

    num_of_sample = x.shape[0]

    y_pred = torch.zeros((num_of_sample, 1))

    model.eval()
    model.cuda()

    batch_size = 100

    batch_per_epoch = np.ceil(np.divide(num_of_sample, batch_size))
    batch_count = 0

    while batch_count < batch_per_epoch:
        start = batch_count * batch_size
        end = np.minimum(start + batch_size, num_of_sample)
        batch_count += 1

        X = x[start:end, :].cuda()
        Y = y[start:end, 0:1].cuda()

        Y_hat = model(X)

        y_pred[start:end, :] = Y_hat

    y_pred_numpy = torch.Tensor.cpu(y_pred.detach()).numpy()

    performance = 10 * np.log10(np.mean(np.abs(y_pred_numpy - y_true_numpy) ** 2))

    return performance


if __name__ == '__main__':

    num_ant = 8
    mode = 'signal'

    val_list = []

    for ii in range(10):
        model = GainPred_Dense_v2(num_ant)
        model.load_state_dict(torch.load('./trained_model/fully_connected_params_M_8_signal_tr_size_10000_repeat_' + str(ii) + '.pt'))
        data_path = './datasets/surrogate_test_dataset_M_' + str(num_ant) + '_' + mode + '.mat'
        val_list.append(model_pre_eval(model, data_path))

    print("FC-based interference prediction accuracy (NMSE): {:5f} dB.".format(np.nanmean(val_list)))
