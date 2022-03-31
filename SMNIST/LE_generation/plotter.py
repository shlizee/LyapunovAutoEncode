import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from torch.nn import functional as F
import numpy as np
import pickle
from dataloader import MNIST_dataloader
import torch.nn as nn
from model import LSTM, GRU
from tl_lyapunov import calc_LEs_an
import copy
from PIL import Image
import torchvision
from torchvision.transforms import ToPILImage


def logit_plot(input_sample,  indices=[1, 9, 15], model_type='lstm', hidden_size=64,):
    losses = torch.zeros([len(indices), ])
    outputs_logits = np.zeros([len(indices), 10])
    predictions = torch.zeros([len(indices, )])
    predictions_targets = torch.zeros([len(indices, )])
    full_outputs = np.zeros([len(indices) * 2, 10])
    for i, index in enumerate(indices):
        models = pickle.load(open(f'trials/{model_type}/models/{model_type}_{hidden_size}_trials_0.pickle', 'rb'))
        model, model_loss, model_acc = models[index][19]['model'], models[index][19]['loss'], models[index][19]['accuracy']
        print(f'model general performance: loss {model_loss:.3f}, accuracy {model_acc:.3f}')
        device = model.device
        inputs, targets = input_sample
        inputs = inputs.to(device)
        targets = torch.tensor([targets]).to(device)
        outputs = model(inputs)[0]
        outputs_logits[i] = F.softmax(outputs, dim=1).detach().cpu().numpy()
        criterion = nn.CrossEntropyLoss()
        losses[i] = criterion(outputs, targets)
        predictions[i] = torch.argmax(outputs[0])
        predictions_targets[i] = targets
        full_outputs[2 * i] = outputs_logits[i]
        print(outputs_logits[i], targets)
        print(f"Prediction: {torch.argmax(outputs[0])}, target: {targets}, loss: {losses[i]}")

    f = plt.figure(figsize=(4, 3))
    ax = f.add_axes((0.02, 0.1, 0.73, 0.8))


    keep_chars = 10
    norm = colors.LogNorm(vmin=2e-3, vmax=1)
    cmap = plt.get_cmap('gray_r')
    pc = ax.pcolor(full_outputs, cmap=cmap, norm=norm)
    ax.grid(False)
    ax.set_xticks(np.array(range(keep_chars)) + 0.5)
    ax.set_xticklabels(labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    # ax.set_xticklabels(labels=char_labels[new_max_inds.numpy()][:keep_chars], Fontsize=16)
    ax.set_yticks([])
    # ax.set_yticklabels([f'Error=\n{loss:0.3f}  ' for loss in losses])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    for i in range(3):
        loss = losses[i]
        max_ind = predictions_targets[i]
        # print(max_ind)
        ax.text(max_ind.item() + 0.5, 2 * i + 0.5, f'{full_outputs[2* i, int(max_ind.item())]:.2f}', color='w',
                horizontalalignment='center', rotation='vertical', verticalalignment='center')
        ax.text(keep_chars / 2, 2 * i + 1.2, f'Error={loss:0.2f}', horizontalalignment='center')
    cax = f.add_axes([0.77, 0.1, 0.05, 0.70])
    f.colorbar(pc, cax=cax, label='Probability')
    # plt.savefig('CharRNNPredictions2', dpi=200, hbbox_inches='tight')
    plt.show()


def LE_plotter(model_type='lstm', dir='lstm/', LE_idx=[]):
    x_data = torch.load(f'Processed/{dir}{model_type}_allLEs.p')
    LE_plotted = x_data[LE_idx]
    plt.figure(figsize=(4, 3))
    plt.gray()
    plt.style.use('grayscale')
    for LEs in LE_plotted:
        # print(LEs.shape)
        plt.scatter(list(range(1024)), LEs, s=8, alpha=0.2)
    plt.axis('off')
    plt.savefig('SampleLE_spectra', dpi=200)
    plt.show()

def loading_samples(hidden_size, model_type='lstm', num_trials=2):
    # losses = torch.tensor([])
    # a_s = torch.tensor([])
    first_time = True
    threshold_a = 2.0
    total_trials = 0
    for i in range(num_trials):
        trials = pickle.load(open(f'trials/{model_type}/models/{model_type}_{hidden_size}_trials_{i}.pickle', 'rb'))
        total_trials += len(trials)
        for j in trials.keys():
            model, model_loss, model_acc, model_a = trials[j][19]['model'], trials[j][19]['loss'], trials[j][19]['accuracy'], trials[j]['a']
            if model_a > threshold_a:
                if first_time:
                    losses = np.array([model_loss.detach().cpu().numpy()])
                    a_s = np.array([model_a])
                    first_time = False
                else:
                    losses = np.concatenate((losses, [model_loss.detach().cpu().numpy()]), axis=0)
                    a_s = np.concatenate((a_s, [model_a]), axis=0)
        print(f"The number of total trials: {total_trials}, number of remain trials: {len(a_s)}")
    return a_s, losses
def plot_distribution(hidden_sizes, model_type='lstm'):
    threshold = 0.6
    good_performance_count = np.zeros([len(hidden_sizes), 1])
    bad_performance_count = np.zeros([len(hidden_sizes), 1])
    for i, hidden_size in enumerate(hidden_sizes):
        if hidden_size == 32:
            num_trials = 3
        elif hidden_size == 64:
            num_trials = 7
        if hidden_size == 128:
            num_trials = 3
        a, loss = loading_samples(hidden_size, model_type, num_trials)

        good_performance_count[i] = len(np.where(loss <= threshold)[0])

        bad_performance_count[i] = len(np.where(loss > threshold)[0])
        if i == 0:
            a_s = a
            losses = loss
        else:
            losses = np.concatenate((losses, loss), axis=0)
            a_s = np.concatenate((a_s, a), axis=0)

        print(a_s.shape, losses.shape)
    print(good_performance_count, bad_performance_count)

    good_performance_idx = np.where(losses <= threshold)[0]
    bad_performance_idx = np.where(losses > threshold)[0]
    plt.figure()
    plt.scatter(a_s[bad_performance_idx], losses[bad_performance_idx], s = 50, c='red', alpha=0.5)
    plt.scatter(a_s[good_performance_idx], losses[good_performance_idx], s = 50, c='lime', alpha=0.5)
    plt.xticks([2.0, 2.5, 3], labels=[ '', '', ''])
    plt.yticks([0, 0.5, 1, 1.5, 2.0, 2.5], labels=['', '', '', '', '', ''])
    plt.show()

    if len(hidden_sizes) == 2:
        plt.figure()
        plt.bar([1, 2], height=(good_performance_count[:, 0] + bad_performance_count[:, 0]), color='red', width=[0.5, 0.5])
        plt.bar([1, 2], height=good_performance_count[:, 0], color='lime', width=[0.5, 0.5])
        # plt.xticks([1, 2],['', ''])
        # plt.yticks([0, 100, 200], ['', '', ''])
        plt.show()
    elif len(hidden_sizes) == 3:
        plt.figure()
        plt.bar([1, 2, 3], height=(good_performance_count[:, 0] + bad_performance_count[:, 0]), color='red', width=[0.5, 0.5, 0.5])
        plt.bar([1, 2, 3], height=good_performance_count[:, 0], color='lime', width=[0.5, 0.5, 0.5])
        # plt.xticks([1, 2],['', ''])
        # plt.yticks([0, 100, 200], ['', '', ''])
        plt.show()

if __name__ == "__main__":
    model_type = 'lstm'
    # hidden_sizes = [64]
    hidden_size = 64
    batch_size = 100
    test_dataloader = MNIST_dataloader(batch_size, train=False)
    # data = pickle.load(open(f'trials/lstm/models/lstm_128_trials_2.pickle', 'rb'))
    # print(len(data))
    # plot_distribution(hidden_sizes, model_type)
    indices = [948]
    test_sample = test_dataloader.dataset[indices[0]]
    # for index in indices:
    #     test_sample, test_target = test_dataloader.dataset[index]
    #
    #     plt.figure()
    #     to_pil = torchvision.transforms.ToPILImage()
    #     img = to_pil(test_sample)
    #     plt.imshow(img)
    #     plt.show()
    logit_plot(test_sample, hidden_size=hidden_size)
