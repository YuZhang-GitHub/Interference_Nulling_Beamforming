import os
import random
import torch
import torch.nn as nn
import torch.optim as optimizer
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

from surrogate_model import GainPred_Dense_v2
from check_for_paper import model_pre_eval

if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    options = {
        'gpu_idx': 0,
        'num_ant': 256,
        'mode': 'signal',
    }

    # ----------------------------------------------------------------------------- #

    options['path'] = './datasets/surrogate_tr_dataset_M_' + str(options['num_ant']) + '_' + options['mode'] + '.mat'

    data = scio.loadmat(options['path'])  # 'gain_vec', 'ph_vec'

    gain_vec = data['gain_vec']
    ph_vec = data['ph_vec']

    # ----------------------------------------------------------------------------- #

    x = ph_vec.transpose()  # shape: (num_of_sample, num_of_ant)

    y_dB = gain_vec.transpose()  # shape: (num_of_sample, 1)
    y_linear = np.power(10, np.divide(y_dB, 10))

    # ----------------------------------------------------------------------------- #

    num_of_sample = x.shape[0]

    # ----------------------------------------------------------------------------- #

    train_num_list = [100000]  # M=256
    # train_num_list = [10000]  # M=8

    for repeat_idx in range(0, 10):

        shuffle_ind = np.random.permutation(num_of_sample)
        x_shuffled = x[shuffle_ind]
        y_shuffled_dB = y_dB[shuffle_ind]
        y_shuffled_linear = y_linear[shuffle_ind]

        for num_tr in train_num_list:

            # ----------------------------------------------------------------------------- #

            dataset = {'x_train': torch.from_numpy(x_shuffled[:num_tr, :]).float(),
                       'y_train': torch.from_numpy(y_shuffled_linear[:num_tr, 0:1]).float(),
                       'y_train_db': torch.from_numpy(y_shuffled_dB[:num_tr, 0:1]).float(),
                       'x_test': torch.from_numpy(x_shuffled[num_tr:, :]).float(),
                       'y_test': torch.from_numpy(y_shuffled_linear[num_tr:, 0:1]).float(),
                       'y_test_db': torch.from_numpy(y_shuffled_dB[num_tr:, 0:1]).float()}

            # %% deep learning model

            model = GainPred_Dense_v2(options['num_ant'])

            # %% model training parameters

            opt = optimizer.Adam(model.parameters(), lr=1e-2)
            scheduler = optimizer.lr_scheduler.MultiStepLR(opt, [50, 100, 200], gamma=0.1, last_epoch=-1)
            criterion = nn.MSELoss()

            batch_size = 1024
            num_of_epoch = 500
            val_loss_list = []

            # fig, ax = plt.subplots(1, figsize=(4, 4))
            fig, ax = plt.subplots(1, figsize=(8, 4))

            if not os.path.exists('results/'):
                os.mkdir('results/')

            with torch.cuda.device('cuda:' + str(options['gpu_idx'])):

                model.cuda()

                for epoch in range(num_of_epoch):

                    batch_count = 0
                    batch_per_epoch = np.ceil(np.divide(dataset['x_train'].size(0), batch_size)).astype('int32')

                    model.train()

                    while batch_count < batch_per_epoch:

                        start = batch_count * batch_size
                        end = np.minimum(start + batch_size, dataset['x_train'].size(0))
                        batch_count += 1

                        X = dataset['x_train'][start:end, :].cuda()
                        Y = dataset['y_train_db'][start:end, 0:1].cuda()

                        Y_hat = model(X)
                        loss = criterion(Y, Y_hat)

                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        if batch_count % 50 == 0:
                            print("Epoch: {:d}/{:d}, Batch: {:d}/{:d}, Training loss: {:.5f}.".format(epoch + 1,
                                                                                                      num_of_epoch,
                                                                                                      batch_count,
                                                                                                      batch_per_epoch,
                                                                                                      loss.item()))

                    model.eval()
                    batch_per_epoch = np.ceil(np.divide(dataset['x_test'].size(0), batch_size))
                    batch_count = 0
                    val_loss = 0
                    while batch_count < batch_per_epoch:
                        start = batch_count * batch_size
                        end = np.minimum(start + batch_size, dataset['x_test'].size(0))
                        batch_count += 1

                        X = dataset['x_test'][start:end, :].cuda()
                        Y = dataset['y_test_db'][start:end, 0:1].cuda()

                        # Y_hat = 10 * torch.log10(model(X))
                        Y_hat = model(X)
                        loss = criterion(Y, Y_hat)
                        val_loss += loss.item()

                    val_loss_list.append(val_loss / batch_per_epoch)
                    print("Repeat: {:d}, TrSize: {:d}, Epoch: {:d}, Validation loss: {:.5f}.".
                          format(repeat_idx, num_tr, epoch + 1, val_loss / batch_per_epoch))

                    ax.plot(range(epoch + 1), 10 * np.log10(np.array(val_loss_list)), '-k', alpha=0.7, label='MSE Loss')

                    ax.grid(True)
                    ax.legend()
                    # plt.draw()
                    # plt.pause(0.001)

                    if (epoch + 1) == num_of_epoch:
                        plt.savefig('./results/record_FC_M_' + str(options['num_ant']) + '_' + options['mode'] + '_tr_size_' + str(num_tr) + '_repeat_' + str(repeat_idx) + '.png', bbox_inches='tight')

                    ax.cla()

                    current_lr = scheduler.get_last_lr()[0]
                    scheduler.step()
                    new_lr = scheduler.get_last_lr()[0]
                    if current_lr != new_lr:
                        print("Learning rate reduced to {0:.5f}.".format(new_lr))

            data_path = './datasets/surrogate_test_dataset_M_' + str(options['num_ant']) + '_' + options['mode'] + '.mat'
            val = model_pre_eval(model, data_path)

            if not os.path.exists('./trained_model'):
                os.mkdir('./trained_model')

            model_save_path = './trained_model/fully_connected_params_M_' + str(options['num_ant']) + '_' + options['mode'] + '_tr_size_' + str(num_tr) + '_repeat_' + str(repeat_idx) + '.pt'

            torch.save(model.state_dict(), model_save_path)
