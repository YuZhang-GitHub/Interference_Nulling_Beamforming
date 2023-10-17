import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Critic_(nn.Module):

    def __init__(self, input_size, output_size):
        super(Critic_, self).__init__()

        self.scaling_factor = 16
        self.fc1 = nn.Linear(input_size, self.scaling_factor * input_size)
        self.bn1 = nn.BatchNorm1d(self.scaling_factor * input_size)
        self.fc2 = nn.Linear(self.scaling_factor * input_size, self.scaling_factor * output_size)
        self.bn2 = nn.BatchNorm1d(self.scaling_factor * output_size)
        self.fc3 = nn.Linear(self.scaling_factor * output_size, output_size)

    def forward(self, state, action):
        x = torch.cat((state, action), 1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x


class Critic(nn.Module):

    def __init__(self, input_size, output_size, ch, p_factor):
        super(Critic, self).__init__()

        self.M = int(input_size / 2)
        self.output_size = output_size
        self.penalty_factor = torch.tensor(p_factor).float().cuda()

        # just for debugging
        self.ch_r = torch.from_numpy(ch[:, :self.M].transpose()).float().cuda()
        self.ch_i = torch.from_numpy(ch[:, self.M:].transpose()).float().cuda()

        # self.H_r = nn.Parameter(torch.randn(self.M, 16))
        self.H_r = torch.from_numpy(ch[:, :self.M].transpose()).float().cuda()
        # self.H_i = nn.Parameter(torch.randn(self.M, 16))
        self.H_i = torch.from_numpy(ch[:, self.M:].transpose()).float().cuda()

        self.Thetas = nn.Parameter(torch.randn(self.M, 32))
        self.R = nn.Parameter(torch.rand(self.M, 32))

        self.fc1 = nn.Linear(522, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        # x = torch.cat((state, action), 1)

        # self.H_r = (1 / torch.sqrt(torch.tensor(self.M))) * torch.cos(self.Thetas) * self.R
        # self.H_i = (1 / torch.sqrt(torch.tensor(self.M))) * torch.sin(self.Thetas) * self.R

        x_state = phase2bf(state)
        x_state_r, x_state_i = x_state[:, :self.M], x_state[:, self.M:]

        z_r = x_state_r @ self.H_r + x_state_i @ self.H_i
        z_i = x_state_r @ self.H_i - x_state_i @ self.H_r

        z = z_r ** 2 + z_i ** 2
        z = z ** self.penalty_factor
        # z_min = torch.min(z, dim=1).values.reshape(-1, 1)

        x_action = phase2bf(action)
        x_action_r, x_action_i = x_action[:, :self.M], x_action[:, self.M:]

        u_r = x_action_r @ self.H_r + x_action_i @ self.H_i
        u_i = x_action_r @ self.H_i - x_action_i @ self.H_r

        u = u_r ** 2 + u_i ** 2
        u = u ** self.penalty_factor
        # u_min = torch.min(u, dim=1).values.reshape(-1, 1)

        # ----------- up to this point ----------- #
        # "z" is of size (batch, S)
        # "u" is of size (batch, S)
        # ---------------------------------------- #

        feature = torch.cat((z, u), dim=1)
        out = torch.relu(self.fc1(feature))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)

        # out = 10 * torch.log10(torch.mean(u, dim=1).reshape(-1, 1)) - 10 * torch.log10(torch.mean(z, dim=1).reshape(-1, 1))

        # ----------- this one works ----------- #
        # out = torch.mean(10 * torch.log10(u), dim=1).reshape(-1, 1) - torch.mean(10 * torch.log10(z), dim=1).reshape(-1, 1)
        # -------------------------------------- #

        # out = -torch.mean(torch.divide(torch.tensor(1.), u), dim=1).reshape(-1, 1) + \
        #       torch.mean(torch.divide(torch.tensor(1.), z), dim=1).reshape(-1, 1)

        return out


class Actor(nn.Module):

    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.pi = torch.tensor(np.pi).float().cuda()
        self.scaling_factor = 16
        self.fc1 = nn.Linear(input_size, self.scaling_factor * input_size)
        self.bn1 = nn.BatchNorm1d(self.scaling_factor * input_size)
        self.fc2 = nn.Linear(self.scaling_factor * input_size, self.scaling_factor * output_size)
        self.bn2 = nn.BatchNorm1d(self.scaling_factor * output_size)
        self.fc3 = nn.Linear(self.scaling_factor * output_size, output_size)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x)) * self.pi

        return x


class Actor_(nn.Module):

    def __init__(self, input_size, output_size):
        super(Actor_, self).__init__()
        self.pi = torch.tensor(np.pi).float().cuda()
        self.scaling_factor = 1
        self.fc1 = nn.Linear(input_size, self.scaling_factor * input_size)
        # self.bn1 = nn.BatchNorm1d(self.scaling_factor * input_size)
        # self.fc2 = nn.Linear(self.scaling_factor * input_size, self.scaling_factor * output_size)
        # self.bn2 = nn.BatchNorm1d(self.scaling_factor * output_size)
        # self.fc3 = nn.Linear(self.scaling_factor * output_size, output_size)

    def forward(self, state):
        # x = F.relu(self.bn1(self.fc1(state)))
        # x = F.relu(self.bn2(self.fc2(x)))
        # x = torch.tanh(self.fc3(x)) * self.pi
        x = self.fc1(state)

        return x


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


class OUNoise(object):
    def __init__(self, action_shape, mu=0.0, theta=0.15, max_sigma=1, min_sigma=0.08, decay_period=10000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_shape
        self.low = -np.pi
        self.high = np.pi
        self.state = self.reset()

    def reset(self):
        state = torch.ones(self.action_dim) * self.mu
        return state.float().cuda()

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.normal(0, 1, size=self.action_dim).cuda()
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        # return torch.clamp(action + ou_state, self.low, self.high)
        return action + ou_state


def phase2bf(ph_mat):
    # ph_mat: (i) a tensor, (ii) B x M
    # bf_mat: (i) a tensor, (ii) B x 2M
    # B stands for batch size and M is the number of antenna

    M = torch.tensor(ph_mat.shape[1]).to(ph_mat.device)
    bf_mat = torch.exp(1j * ph_mat)
    bf_mat_r = torch.real(bf_mat)
    bf_mat_i = torch.imag(bf_mat)

    bf_mat_ = (1 / torch.sqrt(M)) * torch.cat((bf_mat_r, bf_mat_i), dim=1)

    return bf_mat_
