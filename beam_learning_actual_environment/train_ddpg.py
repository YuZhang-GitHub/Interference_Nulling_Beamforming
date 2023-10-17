import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as scio

from DDPG_classes import Actor, Critic_, OUNoise, init_weights
from env_ddpg import envCB


def train(options, train_options, beam_id, exp_id):

    with torch.cuda.device(options['gpu_idx']):
        print('Beam', beam_id, 'training begins. GPU being used:', torch.cuda.current_device())

        options['ph_table_rep'] = options['ph_table_rep'].cuda()
        options['multi_step'] = options['multi_step'].cuda()
        options['ph_table'] = options['ph_table'].cuda()

        real_time_perf = np.zeros((train_options['num_iter'],))
        real_time_perf_n = np.zeros((train_options['num_iter'],))

        actor_net = Actor(options['num_ant'], options['num_ant'])
        critic_net = Critic_(2 * options['num_ant'], 1)
        ounoise = OUNoise((1, options['num_ant']))
        env = envCB(options['num_ant'], options['num_bits'], beam_id, options, exp_id)

        actor_net.cuda()
        critic_net.cuda()
        actor_net.apply(init_weights)
        critic_net.apply(init_weights)

        CB_Env = env
        critic_optimizer = optim.Adam(critic_net.parameters(), lr=1e-3)
        actor_optimizer = optim.Adam(actor_net.parameters(), lr=1e-3, weight_decay=1e-2)
        critic_criterion = nn.MSELoss()

        if train_options['overall_iter'] == 1:
            state = torch.zeros((1, options['num_ant'])).float().cuda()  # vector of phases
            print('Initial State Activated.')
        else:
            state = train_options['state']

        # fig, ax = plt.subplots(1, figsize=(8, 4))

        # -------------- training -------------- #
        replay_memory = train_options['replay_memory']
        iteration = 0
        num_of_iter = train_options['num_iter']
        while iteration < num_of_iter:

            # Proto-action
            actor_net.eval()
            action_pred = actor_net(state)
            reward_pred, bf_gain_pred, action_quant_pred, state_1_pred = CB_Env.get_reward(action_pred)
            # reward_pred = torch.from_numpy(reward_pred).float().cuda()

            critic_net.eval()
            q_pred = critic_net(state, action_quant_pred)

            real_time_perf[iteration] = torch.Tensor.cpu(bf_gain_pred.detach()).numpy()

            # Exploration and Quantization Processing
            action_pred_noisy = ounoise.get_action(action_pred, t=train_options['overall_iter'])  # torch.Size([1, action_dim])
            mat_dist = torch.abs(action_pred_noisy.reshape(options['num_ant'], 1) - options['ph_table_rep'])
            action_quant = options['ph_table_rep'][range(options['num_ant']), torch.argmin(mat_dist, dim=1)].reshape(1, -1)
            # action_quant = action_pred

            state_1, reward, bf_gain, terminal = CB_Env.step(action_quant)  # get next state and reward
            # reward = torch.from_numpy(reward).float().cuda()
            action = action_quant.reshape((1, -1)).float().cuda()  # best action accordingly

            real_time_perf_n[iteration] = torch.Tensor.cpu(bf_gain.detach()).numpy()

            replay_memory.append((state, action, reward, state_1, terminal))
            replay_memory.append((state, action_quant_pred, reward_pred, state_1_pred, terminal))
            while len(replay_memory) > train_options['replay_memory_size']:
                replay_memory.pop(0)

            # -------------- Experience Replay -------------- #
            minibatch = random.sample(replay_memory, min(len(replay_memory), train_options['minibatch_size']))

            # unpack minibatch, since torch.cat is by default dim=0, which is the dimension of batch
            state_batch = torch.cat(tuple(d[0] for d in minibatch))  # torch.Size([*, state_dim])
            action_batch = torch.cat(tuple(d[1] for d in minibatch))  # torch.Size([*, action_dim])
            reward_batch = torch.cat(tuple(d[2] for d in minibatch))  # torch.Size([*, 1])
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))  # torch.Size([*, state_dim])

            state_batch = state_batch.detach()
            action_batch = action_batch.detach()
            reward_batch = reward_batch.detach()
            state_1_batch = state_1_batch.detach()

            if torch.cuda.is_available():  # put on GPU if CUDA is available
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                state_1_batch = state_1_batch.cuda()

            # loss calculation for Critic Network
            critic_net.train()
            Q_prime = reward_batch
            Q_pred = critic_net(state_batch, action_batch)
            critic_loss = critic_criterion(Q_pred, Q_prime.detach())

            # Update Critic Network
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # loss calculation for Actor Network
            actor_net.train()
            critic_net.eval()
            actor_loss = torch.mean(-critic_net(state_batch, actor_net(state_batch)))

            # Update Actor Network
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # UPDATE state, epsilon, target network, etc.
            state = state_1_pred
            iteration += 1
            train_options['overall_iter'] += 1  # global counter

            if train_options['overall_iter'] % options['save_freq'] == 0:
                if not os.path.exists('pretrained_model/'):
                    os.mkdir('pretrained_model/')
                PATH = 'pretrained_model/beam' + str(beam_id) + '_iter' + str(train_options['overall_iter']) + '.pth'
                torch.save(critic_net.state_dict(), PATH)
                torch.save(actor_net.state_dict(), PATH)

            # store: best beamforming vector so far
            if train_options['overall_iter'] % options['pf_print'] == 0:
                iter_id = np.array(train_options['overall_iter']).reshape(1, 1)
                best_state = CB_Env.best_bf_vec.reshape(1, -1)
                if os.path.exists('pfs/pf_' + str(beam_id) + '.txt'):
                    with open('pfs/pf_' + str(beam_id) + '.txt', 'ab') as bm:
                        np.savetxt(bm, iter_id, fmt='%d', delimiter='\n')
                    with open('pfs/pf_' + str(beam_id) + '.txt', 'ab') as bm:
                        np.savetxt(bm, best_state, fmt='%.5f', delimiter=',')
                else:
                    np.savetxt('pfs/pf_' + str(beam_id) + '.txt', iter_id, fmt='%d', delimiter='\n')
                    with open('pfs/pf_' + str(beam_id) + '.txt', 'ab') as bm:
                        np.savetxt(bm, best_state, fmt='%.5f', delimiter=',')

            # plotting
            # ax.plot(range(iteration), real_time_perf[:iteration], '-k', alpha=0.7, label='DRL Performance')
            # ax.plot(range(iteration), real_time_perf_n[:iteration], '-b', alpha=0.3, label='Noise Performance')
            # # ax.plot(range(iteration), np.ones(iteration) * options['target'], '--m', alpha=1.0, label='Victory Claimed')
            # # ax.set_xscale('log')
            # ax.grid(True)
            # ax.legend()
            # plt.draw()
            # plt.pause(0.001)
            # ax.cla()

            print(
                "Beam: %d, Iteration: %d, Q value: %.4f, Reward: %.4f, BF Gain pred: %.2f, BF Gain: %.2f, Critic Loss: %.2f, Policy Loss: %.2f" % \
                (beam_id, train_options['overall_iter'],
                 torch.Tensor.cpu(q_pred.detach()).numpy().squeeze(),
                 torch.Tensor.cpu(reward_pred).numpy().squeeze(),
                 torch.Tensor.cpu(bf_gain_pred.detach()).numpy().squeeze(),
                 torch.Tensor.cpu(bf_gain.detach()).numpy().squeeze(),
                 torch.Tensor.cpu(critic_loss.detach()).numpy().squeeze(),
                 torch.Tensor.cpu(actor_loss.detach()).numpy().squeeze()))

        # Training Communication Interface
        train_options['replay_memory'] = replay_memory  # used for the next loop
        train_options['state'] = state  # used for the next loop
        train_options['best_state'] = CB_Env.best_bf_vec  # used for clustering and assignment

        file_name = 'perf_vs_iter_exp_' + str(exp_id) + '.mat'
        scio.savemat(file_name,
                     {'agent': real_time_perf,
                      'noise': real_time_perf_n})

    return train_options
