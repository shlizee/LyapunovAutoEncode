import torch
from AEPredNet import AEPredNet
from AE_utils import mini_batch_ae, train_val_split
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Visualizing g vs. target_learning loss(and its average)
def plotting(gs, targets_avg, targets_all):
    plt.figure()

    x_axis = gs[0] * torch.ones_like(targets_all)
    plt.scatter(x_axis, targets_all,s = 1)
    plt.scatter(gs, targets_avg)
    plt.plot(gs, targets_avg, 'k-', linewidth=3)
    plt.axis([min(gs)-0.1, max(gs) + 0.1, -.1, 1.1])
    plt.show()

# Fetch the LEs and target_learning loss as inputs and targets for auto-encoder network
def pre_preparation(inputs_epoch, target_epoch, N, gs):

    targets_avg = torch.zeros((len(gs), 1))
    for i, g in enumerate(gs):
        first_time = True
        for trial in range(0, 11):
            file_path = '../lyapunov-hyperopt-master/trials/N_{}/4sine_learner_N_{}_g_{}_trial_{}.p'.format(N, N, g, trial)
            trials = pickle.load(open(file_path, 'rb'))
            count = 0
            for key in trials.keys():
                if (count == 0):
                    inputs = np.expand_dims(trials[key]['LEs_stats'][inputs_epoch]['LEs'], axis=0)
                    targets = np.expand_dims(np.array(trials[key]['LEs_stats'][target_epoch]['val_loss']), axis=0)
                    # print(type(targets))
                else:
                    inputs = np.concatenate((inputs,np.expand_dims(trials[key]['LEs_stats'][inputs_epoch]['LEs'], axis=0)), axis = 0)
                    targets = np.concatenate((targets, np.expand_dims(np.array(trials[key]['LEs_stats'][target_epoch]['val_loss']), axis = 0)), axis=0)
                count += 1
            # print(g, targets.shape)
            # print(targets)
            inputs = torch.tensor(inputs)
            targets = torch.tensor(targets)
            inputs = interpolate(inputs)
            # targets_avg[i] = torch.mean(targets)
            # print(targets_avg[i])
            device = inputs.device
            if (first_time):
                inputs_g = inputs
                targets_g = targets
                first_time = False
            else:
                inputs_g = torch.cat((inputs_g, inputs), dim=0).to(device)
                targets_g = torch.cat((targets_g, targets), dim=0).to(device)
                print(g, targets_g.shape)
        targets_avg[i] = torch.mean(targets_g)
        print(targets_avg[i])
        if i == 0:
            inputs_all = inputs_g
            targets_all = targets_g
        else:
            inputs_all = torch.cat((inputs_all, inputs_g), dim=0).to(device)
            targets_all = torch.cat((targets_all, targets_g), dim=0).to(device)
    print(inputs_all.shape)
    print(targets_all.shape)

    # Visualizing
    plotting(gs, targets_avg, targets_all)

    # Save data for AE network
    data_path = "training_data/g_1.5/4sine_epoch_{}_N_{}".format(inputs_epoch, N)
    data = {"inputs": inputs_all, "targets": targets_all}
    pickle.dump(data, open(data_path, 'wb'))

# interpolate the inputs so that its dimension increases to twice
def interpolate(inputs, inputs_dim = 512, target_dim = 1024):
    m, n = inputs.shape
    device = inputs.device
    shift = torch.cat((inputs[:, 1:], torch.zeros((m, 1), dtype=inputs.dtype)), dim=1).to(device)

    diffs = (inputs - shift) / 2;
    diffs[:, -1] = diffs[:, -2]

    new_inputs = torch.zeros(m, 0).to(device)
    for col, diff in zip(inputs.T, diffs.T):
        new_inputs = torch.cat((new_inputs, col.unsqueeze(1)), dim=1)
        new_inputs = torch.cat((new_inputs, (col - diff).unsqueeze(1)), dim=1)

    # check if it works
    # print(inputs[0,:4])
    # print(new_inputs[0,:4])
    return new_inputs

def main():
    inputs_epoch = 13
    target_epoch = 14
    N = 512
    gs = [1.5]
    pre_preparation(inputs_epoch, target_epoch, N, gs)
    data_path = "training_data/g_1.5/4sine_epoch_{}_N_{}".format(inputs_epoch, N)
    data = pickle.load(open(data_path, 'rb'))

    inputs, targets = data['inputs'], data['targets']


    plotting(gs, torch.mean(targets), targets)
    # print(targets)
if __name__ == "__main__":
    main()