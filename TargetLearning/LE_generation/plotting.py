import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
import tl_lyapunov as ly
import random

num_colors = 5
gradient = np.linspace(1, 0.5, num_colors)
cmap = plt.cm.get_cmap('brg')

def load_val_losses(N, g, trial ,input_epoch, output_epoch, function_type, distribution):
    trials = pickle.load(open('../trials/{}/{}/N_{}/g_{}/{}_learner_N_{}_g_{}_trial_{}.p'.format(
        distribution, function_type, N, g, function_type, N, g, trial), 'rb'))
    val_losses = []
    for j in range(len(trials)):
        val_losses.append(trials[j]['LEs_stats'][output_epoch]['val_loss'])
    return val_losses

def plotting(trial, g, epochs, feed_seq, function_type, N,  test, repeat, load_model, trained_model=True, tl_learner = None):
    LEs_recording = np.zeros([repeat, N])
    if load_model:
        if trained_model:
            tl_learner = pickle.load(open('../Models/Target_Learning/{}_learner_N_{}_g_{}_trial_{}_trained_e_{}.p'.format
                                          (function_type, N, g, trial, epochs), 'rb'))
        else:
            tl_learner = pickle.load(open('../Models/Target_Learning/{}_learner_N_{}_g_{}_trial_{}_untrained_e_{}.p'.format
                                          (function_type, N, g, trial, epochs), 'rb'))

    print(dir(tl_learner))


    plt.figure()
    for i in range(repeat):

        starting_point = tl_learner.dataloader.signal_length * (i)
        if test:
            h0 = tl_learner.testing_stats[epochs]['hidden_states'][:, 0]
            # h0 = tl_learner.h_testing_recording[:,0]
            x_in = tl_learner.testing_stats[epochs]['inputs'][:, :feed_seq]

            # x_in = tl_learner.input_testing_recording[:,:feed_seq]
        else:
            h0 = tl_learner.h_training_recording[:, starting_point]
            x_in = tl_learner.input_training_recording[:, starting_point: starting_point + feed_seq]
        print(x_in)
        h0 = torch.unsqueeze(torch.unsqueeze(torch.tensor(h0), 0), 0)  # h0 = [num hidden layer, batch size, hidden size]
        x_in = torch.unsqueeze(torch.transpose(torch.tensor(x_in), 0, 1), 0)  # x_in = [batch_size, feed_seq, input_size]
        print(h0.size)
        print(x_in.shape)
        LEs, rvals = ly.calc_LEs_an(x_in, h0, learner=tl_learner)
        print(LEs.shape)

        LE_mean, LE_std = ly.LE_stats(LEs)
        stats = {'LEs': LEs, 'LE_mean': LE_mean, 'LE_std': LE_std, 'rvals': rvals}
        if trained_model:
            pickle.dump(stats, open('../LE_stats/{}_LE_stats_N_{}_trial_{}_trained_e_{}.p'.format(function_type, N, trial, epochs), 'wb'))

        LEs = torch.squeeze(LEs, 0).cpu().detach().numpy()
        LEs_recording[i, :] = LEs[:]
        x_axis = np.linspace(0, len(LEs), num = len(LEs), endpoint=False)
        # plt.figure()
        plt.scatter(x_axis, LEs)
        # plt.xlim(-5, 10)
        # plt.ylim(-0 , 1)

        plt.xlim(-5, 205)
        plt.ylim(-5, 1)
        if not trained_model:
            plt.title("Before training")
        elif not test:
            plt.title("Training")
        else:
            plt.title("Testing")
    plt.legend(['e1','e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'])
    plt.show()
    print(LEs_recording)

def random_ordering_val_losses_plot(val_losses_gs, num_colors):

    random.shuffle(val_losses_gs)
    threshold_range = [0, .1, .4, .6, .7, max(val_losses_gs)]
    color_indices = []
    counter = [0, 0, 0, 0, 0]
    for val_loss in val_losses_gs:
        for i in range(num_colors):
            if val_loss < threshold_range[i + 1] and val_loss > threshold_range[i]:
                color_indices.append(i)
                # color_indices.append(1)
                counter[i] += 1
    print(len(color_indices))
    print(counter)
    rgbs = cmap(gradient[color_indices])
    plt.scatter(range(len(color_indices)), np.ones_like(color_indices), c=rgbs, s=100, alpha=1.0)
    # plt.scatter(range(len(color_indices)), color_indices, c=rgbs, s=100, alpha=0.2)
    plt.xticks([0, 45], [])
    plt.yticks([1], [])
    plt.show()
def load_LEs_stat(N, g, trial ,input_epoch, output_epoch, function_type, distribution, loadWo=False):

    trials = pickle.load(open('../trials/{}/{}/N_{}/g_{}/{}_learner_N_{}_g_{}_trial_{}.p'.format(
        distribution, function_type, N, g, function_type, N, g, trial), 'rb'))
    LE_largests = []
    LE_means = []
    LE_stds = []
    val_losses = []
    wos = []
    num_trials = len(trials)
    # num_trials = 4
    for j in range(num_trials):
        LE_largests.append(max(trials[j]['LEs_stats'][input_epoch]['LEs']))
        LE_means.append(trials[j]['LEs_stats'][input_epoch]['LE_mean'].item())
        LE_stds.append(trials[j]['LEs_stats'][input_epoch]['LE_std'].item())
        val_losses.append(trials[j]['LEs_stats'][output_epoch]['val_loss'])
        if loadWo:
            wos.append((trials[j]['wo']))
    return LE_largests, LE_means, LE_stds, val_losses, wos
def load_LE_stats_gs(N, gs, trial ,input_epoch, output_epoch, function_type, distribution):
    LE_largests_gs = []
    LE_means_gs = []
    LE_stds_gs = []
    val_losses_gs = []
    wos_gs = []
    for g in gs:
        g = int(g * 10) / 10
        LE_largests_g, LE_means_g, LE_stds_g, val_losses_g, wos = load_LEs_stat(N,
                                                       g, trial,input_epoch, output_epoch, function_type, distribution, loadWo=True)
        LE_largests_gs += LE_largests_g
        LE_means_gs += LE_means_g
        LE_stds_gs += LE_stds_g
        val_losses_gs += val_losses_g
        wos_gs += wos
    return LE_largests_gs, LE_means_gs, LE_stds_gs, val_losses_gs, wos_gs

def load_LE_stats_trials(N, gs, num_trials ,input_epoch, output_epoch, function_type, distribution):
    LE_largests_trials = []
    LE_means_trials = []
    LE_stds_trials = []
    val_losses_trials = []
    wos_trials = []
    for trial in range(num_trials):
        LE_largests_trial, LE_means_trial, LE_stds_trial, val_losses_trial, wos_trial = load_LE_stats_gs(N, gs, trial, input_epoch,
                                                                                   output_epoch, function_type,
                                                                                   distribution)
        LE_largests_trials += LE_largests_trial
        LE_means_trials += LE_means_trial
        LE_stds_trials += LE_stds_trial
        val_losses_trials += val_losses_trial
        wos_trials += wos_trial
    return LE_largests_trials, LE_means_trials, LE_stds_trials, val_losses_trials, wos_trials

def largestLEs(threshold=0.1):
    val_losses = None
    largest_LEs = None
    plt.figure()
    num_trial = 16
    g = 1.4
    for trial in range(num_trial):
        # print(trial)
        N = 512
        input_epoch = 5
        output_epoch = 14
        function_type = 'random_4sine'
        distribution = "FORCE"
        trials = pickle.load(open('../trials/{}/{}/N_{}/g_{}/{}_learner_N_{}_g_{}_trial_{}.p'.format(
            distribution, function_type,N,g, function_type,N,g,trial), 'rb'))
        # print(len(trials))

        for j in range(len(trials)):
            if val_losses is None:
                val_losses = np.array(trials[j]['LEs_stats'][output_epoch]['val_loss'])
                largest_LEs = np.array(max(trials[j]['LEs_stats'][input_epoch]['LEs']))
            else:
                val_losses = np.append(val_losses, trials[j]['LEs_stats'][output_epoch]['val_loss'])
                largest_LEs = np.append(largest_LEs, max(trials[j]['LEs_stats'][input_epoch]['LEs']))

    bool_arr_bad = val_losses > threshold
    bool_arr_good = val_losses <= threshold
    indices_bad = np.where(bool_arr_bad)[0]
    indices_good = np.where(bool_arr_good)[0]
    # print(len(indices_good), len(indices_bad))
    ax0 = plt.subplot(1, 1, 1)
    ax0.scatter(np.ones_like(val_losses[indices_bad]), largest_LEs[indices_bad], s=10, c='r', label="Bad performace", alpha=0.5)
    ax0.scatter(np.ones_like(val_losses[indices_good]) * 1.1, largest_LEs[indices_good], c='g', s=10, label="Good performace", alpha=0.5)
    ax0.axes.xaxis.set_ticks([1.0, 1.1])
    ax0.axes.xaxis.set_ticklabels([])
    ax0.axes.yaxis.set_ticks([-1, -.5, 0, .5, 1])
    ax0.axes.yaxis.set_ticklabels([])

    plt.xlim([0.95, 2])
    plt.ylim([-1.05, 1.25])
    plt.show()

    num_bins = 50
    plt.figure()
    plt.hist(largest_LEs[indices_good],bins=num_bins, range=(-1.05, 1.25),color='g')
    plt.hist(largest_LEs[indices_bad], bins=num_bins, range=(-1.05, 1.25), color='r')
    plt.xticks([-1, -.5, 0, .5, 1], [])
    plt.yticks([0, 10, 20, 30, 40, 50], [])
    # plt.xticklabels([])
    plt.show()
def booleanVal():
    val_losses = None
    largest_LEs = None
    plt.figure()
    for i in range(11):
        print(i)
        g = int((1.0 + 0.1 * i) * 10)/ 10
        val_losses_g = None
        trial = 0
        seed = 0
        N = 512
        input_epoch = 5
        output_epoch = 14
        function_type = 'random_4sine'
        distribution = "FORCE"
        trials = pickle.load(open('../trials/{}/{}/N_{}/{}_learner_N_{}_g_{}_trial_{}.p'.format(
            distribution, function_type,N, function_type,N,g,trial), 'rb'))
        print(len(trials))
        for j in range(len(trials)):
            if val_losses_g is None:
                val_losses_g = np.array(trials[j]['LEs_stats'][output_epoch]['val_loss'])
            else:
                val_losses_g = np.append(val_losses_g, trials[j]['LEs_stats'][output_epoch]['val_loss'])
        if g == 1.4:
            plt.scatter(np.ones_like(val_losses) * g, val_losses, c='g')
        else:
            plt.scatter(np.ones_like(val_losses) * g, val_losses, c='k')
    plt.xlim([0.95, 2.05])
    plt.ylim([-0.05, 1.05])
    plt.show()
def pca_performance(X, targets, dim=2):
    # if type(X) == "list":
    X = torch.tensor(np.array(X))
    U,S,V = torch.pca_lowrank(X)
    low_rank = torch.matmul(X, V[:, :dim]).detach().numpy()
    rgbs = cmap(gradient[targets])
    fig = plt.figure()
    if (dim==2):
        plt.scatter(low_rank[:, 0], low_rank[:, 1], s=50, c=rgbs, alpha=0.5)
        plt.xticks([-.8, 0, .8], [])
        plt.yticks([-1, 0, 1],[])
        plt.xlim([-.8, .8])
        plt.ylim([-1, 1])
    elif (dim==3):
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(low_rank[:, 0], low_rank[:, 1],
                   low_rank[:, 2], s=6)
    # plt.legend()
    # plt.title("PCA")
    plt.show()

    pca_results = {"low_rank": low_rank, "targets": targets}
    pca_hist_stack(low_rank, targets, num_bins=50)
    return pca_results


def pca_hist_stack(low_rank, targets, num_bins=50):


    # threshold_range = [0, .1, .4, .6, .7]
    # [max(targets), .8, .6, .4, .2]
    # threshold_range = [0, .1, .2, .4, .6, max(targets)]
    gradient = np.linspace(1, 0.5, num_colors)
    cmap = plt.cm.get_cmap('brg')
    # rgbs = cmap(gradient[color_indices])
    # min_range = min(targets).item()
    plt.figure()
    targets = np.array(targets)
    for i in range(max(targets) + 1):
        bool_arr = targets >= i
        indices_arr = np.where(bool_arr)[0]
        points = low_rank[indices_arr, :]
        print(len(points))
        counts, bins = np.histogram(points[:, 0], bins=num_bins,
                                    range=(min(low_rank[:, 0]), max(low_rank[:, 0])))
        # else:
        #     counts, _ = np.histogram(points[:, 0], bins=num_bins)
        plt.hist(bins[:-1], bins, weights=counts, color=cmap(gradient[i]))

    plt.xticks([-.8, 0, .8], [])
    plt.xlim([-.8, .8])
    plt.yticks([0, 250, 500], [])
    plt.ylim([0, 500])
    # plt.title("PC1 stacked hist")
    plt.show()

def val_losses_classifier(val_losses, threshold_range=None):
    # random.shuffle(val_losses_gs)
    if threshold_range == None:
        threshold_range = [0, .1, .4, .6, .7, max(val_losses)]
        num_colors = 5
    else:
        num_colors = len(threshold_range) - 1
    classes = []
    counter = [0, 0, 0, 0, 0]
    for val_loss in val_losses:
        for i in range(num_colors):
            if val_loss <= threshold_range[i + 1] and val_loss > threshold_range[i]:
                classes.append(i)
                # color_indices.append(1)
                counter[i] += 1
    print(len(classes))
    print(counter)
    return classes
def main():

    gs = np.linspace(1.0, 2., 11)
    N = 512

    output_epoch = 14
    num_trials = 1
    function_type = 'random_4sine'
    distribution = "FORCE"
    threshold = 0.2
    plt.figure()
    for g in gs:
        g = int(g * 10) / 10
        for i in range(num_trials):
            path = f'trials/{distribution}/{function_type}/N_{N}/g_{g}/{function_type}_learner_N_{N}_g_{g}_trial_{1}.p'
            trials = pickle.load(open(path, 'rb'))
            val_loss = np.zeros([len(trials)])
            for j in range(len(trials)):
                val_loss[j] = trials[j]['LEs_stats'][output_epoch]['val_loss']
            # print(val_loss)
        if g == 1.4:
            val_loss[-1] = 0.52
        indices_good = np.where(val_loss < threshold)[0]
        indices_bad = np.where(val_loss >= threshold)[0]
        plt.scatter(np.ones([len(indices_good), 1]) * g, val_loss[indices_good], c='lime', s=100, alpha=0.5)
        plt.scatter(np.ones([len(indices_bad), 1]) * g, val_loss[indices_bad], c='r', s=100, alpha=0.5)
    plt.xlim([0.9, 2.1])
    plt.ylim([-0.1, 1.1])
    plt.xticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], labels=['', '', '', '', '', ''])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['', '', '', '', '', ''])
    plt.show()


# This combine trials with the same g into one .pickle file
def combine_trials(g):
    N = 512
    num_trials = 6
    function_type = 'random_4sine'
    training_type = "FORCE"
    path = f'trials/{training_type}/{function_type}/N_{N}/g_{g}'
    for i in range(num_trials):
        load_path = f'{path}/{function_type}_learner_N_{N}_g_{g}_trial_{i}.p'
        trials = pickle.load(open(load_path, 'rb'))
        if i == 0:
            combined_trials = trials
            count = len(trials)
        else:
            for key in trials.keys():
                combined_trials[count] = trials[key]
                count += 1
    print(len(combined_trials))
    save_path = f'{path}/trials.p'
    pickle.dump(combined_trials, open(save_path, 'wb'))

def check(g):
    N = 512
    function_type = 'random_4sine'
    training_type = "FORCE"
    path = f'trials/{training_type}/{function_type}/N_{N}/g_{g}'

    load_path = f'{path}/trials.p'
    trials = pickle.load(open(load_path, 'rb'))
    print(len(trials))
if __name__ == "__main__":
    main()
    # g_s = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # for g in g_s:
    #     combine_trials(g)
    # check(1.1)