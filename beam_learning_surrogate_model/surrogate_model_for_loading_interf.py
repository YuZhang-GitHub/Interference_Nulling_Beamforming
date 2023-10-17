import torch
import torch.nn as nn
import numpy as np


def phase2bf(ph_mat):

    # ph_mat: (i) a tensor, (ii) B x M
    # bf_mat_: (i) a tensor, (ii) B x 2M
    # B stands for batch size and M is the number of antenna

    M = torch.tensor(ph_mat.shape[1]).to(ph_mat.device)

    bf_mat = torch.exp(1j * ph_mat)
    bf_mat_r = torch.real(bf_mat)
    bf_mat_i = torch.imag(bf_mat)

    bf_mat_ = (1 / torch.sqrt(M)) * torch.cat((bf_mat_r, bf_mat_i), dim=1)

    return bf_mat_


class GainPred_interf(nn.Module):

    def __init__(self, M):
        super(GainPred_interf, self).__init__()

        self.M = M
        self.pi = torch.tensor(np.pi).float().cuda()

        self.H_r = nn.Parameter(torch.randn(self.M, 4))
        self.H_i = nn.Parameter(torch.randn(self.M, 4))

    def forward(self, ph):

        bf_vec = phase2bf(ph)
        bf_vec_r, bf_vec_i = bf_vec[:, :self.M], bf_vec[:, self.M:]

        z_r = bf_vec_r @ self.H_r + bf_vec_i @ self.H_i
        z_i = bf_vec_r @ self.H_i - bf_vec_i @ self.H_r

        z = z_r ** 2 + z_i ** 2

        out = 10 * torch.log10(torch.sum(z, dim=1)).reshape(-1, 1)

        return out


class GainPred_Dense(nn.Module):

    def __init__(self, M):
        super(GainPred_Dense, self).__init__()

        self.M = M
        self.K = 32

        self.fc1 = nn.Linear(2 * M, self.K)
        self.bn1 = nn.BatchNorm1d(self.K)
        self.fc2 = nn.Linear(self.K, self.K)
        self.bn2 = nn.BatchNorm1d(self.K)
        self.fc3 = nn.Linear(self.K, 1)

    def forward(self, ph):

        bf_vec = phase2bf(ph)
        bf_vec_r, bf_vec_i = bf_vec[:, :self.M], bf_vec[:, self.M:]
        bf_vec_cat = torch.cat((bf_vec_r, bf_vec_i), dim=1)

        x = torch.relu(self.bn1(self.fc1(bf_vec_cat)))
        x = torch.relu(self.bn2(self.fc2(x)))
        out = self.fc3(x)

        # out = 10 * torch.log10(torch.pow(out, 2)).reshape(-1, 1)

        return out
