import numpy as np
import h5py as h5
import scipy.io as scio


def dataPrep(inputName=None):
    with h5.File(inputName, 'r') as f:
        fields = [k for k in f.keys()]  # fields = ['ch_grid']
        nested = [k for k in f[fields[0]]]
        data_channels = np.squeeze(np.array(nested))
        decoup = data_channels.view(np.float64).reshape(data_channels.shape + (2,))
        # shape: (#users, #ant, 2), decoup[0,0,0]=real, decoup[0,0,1]=imag
        X_real = decoup[:, 0]  # shape: (#users, #ant), all real parts of channels
        X_imag = decoup[:, 1]  # shape: (#users, #ant), all imag parts of channels
        X = np.concatenate((X_real, X_imag), axis=-1)

    return X


def load_ch(file_path, key_str):

    X = scio.loadmat(file_path)[key_str]
    X = X.transpose()
    X_r = np.real(X)
    X_i = np.imag(X)

    X_ = np.concatenate((X_r, X_i), axis=1)

    return X_


def bf_gain_calc(f, h):

    # f: (i) real-valued beamforming or combining vector, (ii) 1 x 2N
    # ch: (i) real-valued channel vector, (ii) 1 x 2N

    half = int(f.shape[1] / 2)
    f_r, f_i = f[0:1, :half], f[0:1, half:]
    h_r, h_i = h[:, :half], h[:, half:]

    gain_1 = f_r @ h_r.transpose()
    gain_2 = f_i @ h_i.transpose()
    gain_3 = f_r @ h_i.transpose()
    gain_4 = f_i @ h_r.transpose()

    gain_r = (gain_1 + gain_2) ** 2
    gain_i = (gain_3 - gain_4) ** 2
    gain = gain_r + gain_i

    return gain
