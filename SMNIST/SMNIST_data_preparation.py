import torch
from AEPredNet import AEPredNet
from AE_utils import mini_batch_ae, train_val_split
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

# Visualizing g vs. target_learning loss(and its average)
def LEs_plotting(inputs, a):
    plt.figure()
    num_trials, N = inputs.shape
    for i in range(5):
        x_axis = range(N)
        plt.scatter(x_axis, inputs[i,:])
    plt.title("a: {}, N: {}".format(a, N))
    # x_axis = gs[0] * torch.ones_like(targets_all)
    # plt.scatter(x_axis, targets_all,s = 1)
    # plt.scatter(gs, targets_avg)
    # plt.plot(gs, targets_avg, 'k-', linewidth=3)
    # plt.axis([min(gs)-0.1, max(gs) + 0.1, -.1, 1.1])
    plt.show()

# Fetch the LEs and target_learning loss as inputs and targets for auto-encoder network
def pre_preparation(inputs_epoch, target_epoch, N, a_s, interpreted=False):

    targets_avg = torch.zeros((len(a_s), 1))
    first_time = True
    for i, a in enumerate(a_s):

        file_path = '../lyapunov-hyperopt-master/SMNIST/trials/lstm_{}_uni_{}.pickle'.format(N, a)
        trials = pickle.load(open(file_path, 'rb'))
        count = 0
        for key in trials.keys():
            if (count == 0):
                inputs = np.expand_dims(trials[key][inputs_epoch]['LEs'].detach().cpu(), axis=0)
                targets = np.expand_dims(np.array(trials[key][target_epoch]['val_loss'].detach().cpu()), axis=0)
                # print(type(targets))
            else:
                inputs = np.concatenate((inputs,np.expand_dims(
                    trials[key][inputs_epoch]['LEs'].detach().cpu(), axis=0)), axis = 0)
                targets = np.concatenate((targets, np.expand_dims(np.array(
                    trials[key][target_epoch]['val_loss'].detach().cpu()), axis = 0)), axis=0)
            count += 1
        # print(g, targets.shape)
        # print(targets)
        inputs = torch.tensor(inputs)
        inputs = torch.mean(inputs, dim=1)
        targets = torch.tensor(targets)
        # print(inputs.shape, targets.shape)
        LEs_plotting(inputs, a)

        if interpreted:
            inter_times = int(math.log(512 / N, 2))
            for _ in range(inter_times):
                inputs = interpolate(inputs)
        # print(inputs.shape, targets.shape)
        targets_avg[i] = torch.mean(targets)
        print(targets_avg[i])
        device = inputs.device
        if (first_time):
            inputs_as = inputs
            targets_as = targets
            first_time = False
            # print("inputs_g, targets_g shape: ", inputs_as.shape, targets_as.shape)
        else:
            inputs_as = torch.cat((inputs_as, inputs), dim=0).to(device)
            targets_as = torch.cat((targets_as, targets), dim=0).to(device)
            # print(g, targets_g.shape)
            # print("inputs_g, targets_g shape: ",inputs_as.shape, targets_as.shape)

        # Save data for AE network
        if interpreted:
            if len(a_s) > 1:
                data_path = "SMNIST/TrainingData/interpreted/a_mixed/"
            else:
                data_path = "SMNIST/TrainingData/interpreted/a_{}/".format(a)
        else:
            if len(a_s) > 1:
                data_path = "SMNIST/TrainingData/non_interpreted/a_mixed/"
            else:
                data_path = "SMNIST/TrainingData/non_interpreted/a_{}/".format(a)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        data = {"inputs": inputs_as, "targets": targets_as}
        pickle.dump(data, open(data_path + 'epoch_{}_N_{}.pickle'.format(inputs_epoch, N), 'wb'))
    # print(inputs_all.shape)
    # print(targets_all.shape)


    # Visualizing
    # plotting(gs, targets_avg, targets_all)



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
    for inputs_epoch in range(0, 9):
        a_s = [0.1, 0.5, 1, 2, 5]
        target_epoch = 9
        N = 32
        pre_preparation(inputs_epoch, target_epoch, N, a_s, True)
        # data_path = "training_data/{}/g_{}/{}_epoch_{}_N_{}".format(
        #     function_type,g,function_type, inputs_epoch, N)
        # data = pickle.load(open(data_path, 'rb'))
        #
        # inputs, targets = data['inputs'], data['targets']

        # plotting(gs, torch.mean(targets), targets)
        # print(targets)
if __name__ == "__main__":
    main()