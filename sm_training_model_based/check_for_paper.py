import torch
import numpy as np
import scipy.io as scio


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
