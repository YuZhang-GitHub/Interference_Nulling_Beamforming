import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

num_realization = 10
num_iteration = 1000

data_all_agent = np.zeros((num_realization, num_iteration))
data_all_noise = np.zeros((num_realization, num_iteration))

for ii in range(num_realization):

    data = scio.loadmat('./perf_vs_iter_exp_' + str(ii) + '.mat')  # 'agent', 'noise'

    data_all_agent[ii, :] = data['agent']
    data_all_noise[ii, :] = data['noise']

    data_all = np.maximum(data_all_agent, data_all_noise)

data_all_mean = np.nanmean(data_all, axis=0)
data_all_std = np.std(data_all, axis=0)

plt.plot(range(num_iteration), data_all_mean, '-', color='green', alpha=0.8, label='Actual environment based')

plt.fill_between(range(num_iteration), data_all_mean - data_all_std, data_all_mean + data_all_std, color='green', alpha=0.2)

plt.xlabel('Number of iteration')
plt.ylabel('SIR (dB)')

plt.ylim((-15.500842081835165, 27.671106027487284))

# bottom, top = plt.ylim()

plt.grid(True)
plt.legend(loc='lower right')

plt.show()
