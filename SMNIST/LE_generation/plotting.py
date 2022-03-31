import torch
from torch.autograd import Variable
from torch.nn import RNN, GRU, LSTM
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pkl
import gc
import math
import pickle

def LE_stats(LE, save_file=False, file_name='LE.p'):
    mean, std = (torch.mean(LE, dim=0), torch.std(LE, dim=0))
    if save_file:
        pkl.dump((mean, std), open(file_name, "wb"))
    return mean, std


def plot_spectrum(LE, model_name, k_LE=100000, plot_size=(10, 7), legend=[], show=False):
    k_LE = max(min(LE.shape[1], k_LE), 1)
    LE_mean, LE_std = LE_stats(LE)
    f = plt.figure(figsize=plot_size)
    x = range(1, k_LE + 1)
    plt.title('Mean LE Spectrum for ' + model_name)
    f = plt.errorbar(x, LE_mean[:k_LE].to(torch.device('cpu')), yerr=LE_std[:k_LE].to(torch.device('cpu')), marker='.',
                     linestyle=' ', markersize=7, elinewidth=2)
    plt.xlabel('Exponent #')
    if show:
        plt.show()


def main():
    model_type = 'lstm'
    a_s = [0.1, 0.5, 1.0, 1.5, 2.0]
    N_s = [8, 16, 32]
    c_s = ['b', 'g', 'r']
    plt.figure()
    for i, a in enumerate(a_s):
        for j, N in enumerate(N_s):
            trials = pickle.load(open(f'trials/{model_type}/{model_type}_{N}_uni_{a}.pickle', 'rb'))
            val_loss = np.zeros([len(trials), ])
            for k in range(len(trials)):
                val_loss[k] = trials[k][9]['val_loss'].cpu()
            plt.scatter(np.ones_like(val_loss) * i + 0.15 * (j-1), val_loss, c=c_s[j], alpha=0.5)

    plt.xlim([-0.5, 4.5])
    plt.ylim([-0.05, 0.85])
    plt.xticks([0, 1, 2, 3, 4], labels=['', '', '', '', ''])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8], labels=['', '', '', '', ''])
    plt.show()


def val_N(threshold=0.1):
    model_type = 'lstm'
    a_s = [0.1, 0.5, 1.0, 1.5, 2.0]
    N_s = [8, 16, 32]
    val_loss_N_s = {}
    good_count_N_s = np.zeros([len(N_s), ])
    bad_count_N_s = np.zeros([len(N_s), ])
    for j, N in enumerate(N_s):
        val_loss_N = torch.tensor([])
        for i, a in enumerate(a_s):
            trials = pickle.load(open(f'trials/{model_type}/{model_type}_{N}_uni_{a}.pickle', 'rb'))
            for k in range(len(trials)):
                val_loss_N = torch.cat([val_loss_N, torch.unsqueeze(trials[k][9]['val_loss'].cpu(), dim=0)], dim=0)
        good_count_N_s[j] = len(np.where(val_loss_N <= threshold)[0])
        bad_count_N_s[j] = len(np.where(val_loss_N > threshold)[0])
        val_loss_N_s[N] = val_loss_N
    print(good_count_N_s, bad_count_N_s)

    fig = plt.figure()
    ax = fig.add_axes([0.025, 0.025, 0.95, 0.95])
    ax.bar([0, 1, 2], good_count_N_s + bad_count_N_s, color='r', width=0.7)
    ax.bar([0, 1, 2], good_count_N_s, color='lime', width=0.7)
    plt.xticks([0, 1, 2], ['', '', ''])
    plt.yticks([0, 100, 200, 300, 400, 500, 550], ['', '', '', '', '', '', ''])
    plt.show()


def val_a(threshold = 0.1):
    model_type = 'lstm'
    a_s = [0.1, 0.5, 1.0, 1.5, 2.0]
    N_s = [8, 16, 32]
    val_loss_a_s = {}
    good_count_a_s = np.zeros([len(N_s), ])
    bad_count_a_s = np.zeros([len(N_s), ])
    fig = plt.figure()
    for i, a in enumerate(a_s):
        val_loss_a = torch.tensor([])
        for j, N in enumerate(N_s):

            trials = pickle.load(open(f'trials/{model_type}/{model_type}_{N}_uni_{a}.pickle', 'rb'))
            for k in range(len(trials)):
                val_loss_a = torch.cat([val_loss_a, torch.unsqueeze(trials[k][9]['val_loss'].cpu(), dim=0)], dim=0)
        good_idx_a = np.where(val_loss_a <= threshold)[0]
        bad_idx_a = np.where(val_loss_a > threshold)[0]
        plt.scatter(i * np.ones([len(good_idx_a), ]), val_loss_a[good_idx_a], c='lime', alpha=0.5)
        plt.scatter(i * np.ones([len(bad_idx_a), ]), val_loss_a[bad_idx_a], c='red', alpha=0.5)
    plt.xticks([0, 1, 2, 3, 4], ['', '', '', '', ''])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['', '', '', '', '', ''])
    plt.xlim([-0.1, 4.1])
    plt.ylim([-0.05, 1.05])
    plt.show()
    #     good_count_N_s[j] = len(np.where(val_loss_N <= threshold)[0])
    #     bad_count_N_s[j] = len(np.where(val_loss_N > threshold)[0])
    #     val_loss_N_s[N] = val_loss_N
    # print(good_count_N_s, bad_count_N_s)


if __name__ == '__main__':
    # main()
    # val_N(threshold=0.2)
    val_a(threshold=0.2)