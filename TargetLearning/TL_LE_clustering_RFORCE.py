import torch
from AEPredNet import AEPredNet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import colors
import pickle
import seaborn as sns
import pandas as pd
import numpy as np

def tsne(X, targets, dim=2):
    tsne_model = TSNE(perplexity=10, n_components=dim, random_state=1)

    x_embedded = tsne_model.fit_transform(X.detach().numpy())

    # x_embedded = tsne(hidden_outputs.detach(), dim=tsne_dim)
    indices = np.argsort(x_embedded[:, 0])
    a =x_embedded[indices, 0]
    b = x_embedded[indices, 1]
    fig = plt.figure()
    if (dim==2):
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=6)
        # plt.xlim([-60, 40])
        # plt.ylim([-60, 80])
    elif (dim==3):
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_embedded[:, 0], x_embedded[:, 1],
                   x_embedded[:, 2], s=6)
    plt.legend()
    plt.title("TSNE")
    plt.show()
    print(x_embedded.shape)

    tsne_results = {'x_embedded':x_embedded, 'targets':targets}
    # plt.savefig('../lyapunov-hyperopt-master/Figures/AEPredNet_tsne.png', dpi=200)

    return tsne_results

def tsne_distribution(dim, path, index, x_embedded=None):
    if not x_embedded:
        tsne_results = torch.load(path + 'tsne.p')
        x_embedded, targets = tsne_results['x_embedded'], tsne_results['targets']
    fig = plt.figure()
    if (dim == 2):
        ax = fig.add_subplot(111)
        p_F = ax.scatter(x_embedded[index[0]: index[1], 0], x_embedded[index[0]: index[1], 1], s=10)
        p_R = ax.scatter(x_embedded[index[1]: index[2], 0], x_embedded[index[1]: index[2], 1], s=10)

        # fig.colorbar(p, label='Val Loss')
        plt.xlabel('TSNE 1')
        plt.ylabel('TSNE 2')
    plt.title("TSNE of FORCE and RFORCE")
    plt.show()

def tsne_perf(dim, path, x_embedded=None):
    if not x_embedded:
        tsne_results = torch.load(path + 'tsne.p')
        x_embedded, targets = tsne_results['x_embedded'], tsne_results['targets']
    fig = plt.figure()
    if (dim == 2):
        ax = fig.add_subplot(111)
        p = ax.scatter(x_embedded[:, 0], x_embedded[:, 1], s=10, c = targets,
                       norm=colors.LogNorm(vmin=targets.min(), vmax=targets.max()))
        fig.colorbar(p, label='Val Loss')
        plt.xlabel('TSNE 1')
        plt.ylabel('TSNE 2')
    elif (dim==3):
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(x_embedded[:, 0], x_embedded[:, 1],
                   x_embedded[:, 2], s=6, c = targets,
                       norm=colors.LogNorm(vmin=targets.min(), vmax=targets.max()))
        fig.colorbar(p, label='Val Loss')
        ax.set_xlabel('TSNE 1')
        ax.set_ylabel('TSNE 2')
        ax.set_zlabel('TSNE 3')
    plt.title("TSNE Performance")
    plt.show()
    # plt.savefig('../lyapunov-hyperopt-master/Figures/AEPredNet_tsne_performance.png', dpi=200)

def tsne_binary(dim, path, x_embedded=None, threshold=0.1):
    if not x_embedded:
        tsne_results = torch.load(path + 'tsne.p')
        x_embedded, targets = tsne_results['x_embedded'], tsne_results['targets']
    # targets_binary = np.ones_like(targets)
    bool_arr_bad = targets > threshold
    bool_arr_good = targets <= threshold
    indices_bad = np.where(bool_arr_bad)[0]
    indices_good = np.where(bool_arr_good)[0]
    if dim == 2:
        ax0 = plt.subplot(3, 2, 2)
        ax0.scatter(x_embedded[indices_bad, 0], x_embedded[indices_bad, 1],  s=15, c='r', label="Bad performace")
        ax0.scatter(x_embedded[indices_good, 0], x_embedded[indices_good, 1], c='g',s=6, label="Good performace")
        # ax0.legend()
        ax0.set_title(threshold)
    elif dim==3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_embedded[indices_bad, 0], x_embedded[indices_bad, 1], x_embedded[indices_bad, 2],
                    s=15, c='r', label="Bad performace")
        ax.scatter(x_embedded[indices_good, 0], x_embedded[indices_good, 1], x_embedded[indices_good, 2],
                    s=6, c='g', label="Good performace")
        # plt.legend()
        plt.title(threshold)
    # plt.savefig('../lyapunov-hyperopt-master/Figures/tsne/binary/threshol_{}.png'.format(threshold), dpi=200)

    # plt.show()

def tsne_hist(dim, path, x_embedded=None, threshold=0.1, num_bins = 50 ):
    if not x_embedded:
        tsne_results = torch.load(path + 'tsne.p')
        x_embedded, targets = tsne_results['x_embedded'], tsne_results['targets']
    # targets_binary = np.ones_like(targets)
    bool_arr_bad = targets > threshold
    bool_arr_good = targets <= threshold
    indices_bad = np.where(bool_arr_bad)[0]
    indices_good = np.where(bool_arr_good)[0]

    ax1 = plt.subplot(3, 2, 4)
    # First Dimension
    good_points = x_embedded[indices_good, :]
    counts, bins = np.histogram(good_points[:, 0], bins= num_bins)
    ax1.hist(bins[:-1], bins, weights=counts, color='g')

    bad_points = x_embedded[indices_bad, :]
    counts, bins = np.histogram(bad_points[:, 0], bins= num_bins)
    ax1.hist(bins[:-1], bins, weights=counts, color='r')
    ax1.set_ylim([0, 50])
    ax1.set_title("dim 0")

    # Second Dimension
    ax2 = plt.subplot(3, 2, 1)
    good_points = x_embedded[indices_good, :]
    counts, bins = np.histogram(good_points[:, 1], bins= num_bins)
    ax2.hist(bins[:-1], bins, weights=counts, orientation="horizontal", color='g')

    bad_points = x_embedded[indices_bad, :]
    counts, bins = np.histogram(bad_points[:, 1], bins= num_bins)
    ax2.hist(bins[:-1], bins, weights=counts,orientation="horizontal", color='r')
    ax2.set_title("dim 1")
    # ax2.set_xlim([0, 50])

    ax3 = plt.subplot(3, 2, 3)
    ax3.hist2d(good_points[:,0], good_points[:,1 ], bins=20)
    # ax3.set_xlim([-80,80])
    # ax3.set_ylim([-55, 55])
    ax3.set_title("Good performance")

    ax4 = plt.subplot(3, 2, 5)
    ax4.hist2d(bad_points[:,0], bad_points[:,1 ], bins=20)
    # ax4.set_xlim([-80,80])
    # ax4.set_ylim([-55, 55])
    ax4.set_title("Bad Performance")

    # plt.savefig('../lyapunov-hyperopt-master/Figures/tsne/combined/threshold_{}.png'.format(threshold), dpi=200)

    plt.show()

def pca(X, targets, dim=2):
    U,S,V = torch.pca_lowrank(X)
    low_rank = torch.matmul(X, V[:, :dim]).detach().numpy()
    # fig = plt.figure()
    # if (dim==2):
    #     plt.scatter(low_rank[:, 0], low_rank[:, 1], s=6)
    # elif (dim==3):
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(low_rank[:, 0], low_rank[:, 1],
    #                low_rank[:, 2], s=6)
    # plt.legend()
    # plt.title("PCA")
    # plt.show()
    pca_results = {"low_rank": low_rank, "targets": targets}
    return pca_results

def pca_distribution(dim, path, index, low_rank=None):
    if not low_rank:
        pca_results = torch.load(path + 'PCA_dim{}.p'.format(dim))
        low_rank, targets = pca_results['low_rank'], pca_results['targets']
    fig = plt.figure()
    if (dim == 2):
        ax = fig.add_subplot(111)
        p_F = ax.scatter(low_rank[index[0]: index[1], 0], low_rank[index[0]: index[1], 1], s=10, label='FORCE')
        p_R = ax.scatter(low_rank[index[1]: index[2], 0], low_rank[index[1]: index[2], 1], s=10, label='RFORCE')

        # fig.colorbar(p, label='Val Loss')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend()
    plt.title("PCA of FORCE and RFORCE")
    plt.show()

def pca_perf(dim, path, low_rank=None, targets=None):
    if not low_rank:
        pca_results = torch.load(path + 'PCA_dim{}.p'.format(dim))
        low_rank, targets = pca_results['low_rank'], pca_results['targets']
    fig = plt.figure()
    if (dim == 2):
        # ax = fig.add_subplot(111)
        p = plt.scatter(low_rank[:, 0], low_rank[:, 1], s=6, c = targets,
                       norm=colors.LogNorm(vmin=targets.min(), vmax=targets.max()))
        fig.colorbar(p, label='Val Loss')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
    elif (dim==3):
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(low_rank[:, 0], low_rank[:, 1],
                   low_rank[:, 2], s=6, c = targets,
                       norm=colors.LogNorm(vmin=targets.min(), vmax=targets.max()))
        fig.colorbar(p, label='Val Loss')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
    plt.title("PCA Performance")
    plt.show()

def pca_binary(dim, path, low_rank=None, targets=None, threshold=0.1):
    if not low_rank:
        pca_results = torch.load(path + 'PCA_dim{}.p'.format(dim))
        low_rank, targets = pca_results['low_rank'], pca_results['targets']

    bool_arr_bad = targets > threshold
    bool_arr_good = targets <= threshold
    indices_bad = np.where(bool_arr_bad)[0]
    indices_good = np.where(bool_arr_good)[0]

    if (dim==2):
        ax0 = plt.subplot(1,1,1)
        # ax0 = plt.subplot(3, 2, 2)

        ax0.scatter(low_rank[indices_bad, 0], low_rank[indices_bad, 1],  s=25, c='r', label="Bad performace", alpha=0.5)
        ax0.scatter(low_rank[indices_good, 0], low_rank[indices_good, 1], c='g',s=25, label="Good performace", alpha=0.5)
        ax0.axis("off")
        # ax0.set_xlabel('PC 1')
        # ax0.set_ylabel('PC 2')
        # plt.legend()
        # ax0.set_title(threshold)
    elif (dim==3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(low_rank[indices_bad, 0], low_rank[indices_bad, 1], low_rank[indices_bad, 2],
                    s=15, c='r', label="Bad performace")
        ax.scatter(low_rank[indices_good, 0], low_rank[indices_good, 1], low_rank[indices_good, 2],
                    s=6, c='g', label="Good performace")
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        # plt.legend()
        plt.title(threshold)
    # plt.savefig('../lyapunov-hyperopt-master/Figures/pca/binary/threshol_{}.png'.format(threshold), dpi=200)
    plt.show()

def pca_hist(dim, path, low_rank=None, threshold=0.1, num_bins = 50 ):
    if not low_rank:
        pca_results = torch.load(path + 'PCA_dim{}.p'.format(dim))
        low_rank, targets = pca_results['low_rank'], pca_results['targets']

    bool_arr_bad = targets > threshold
    bool_arr_good = targets <= threshold
    indices_bad = np.where(bool_arr_bad)[0]
    indices_good = np.where(bool_arr_good)[0]

    # ax1 = plt.subplot(3, 2, 4)
    plt.figure()
    ax1 = plt.subplot(1,1,1)
    # First Dimension
    good_points = low_rank[indices_good, :]
    counts, bins = np.histogram(good_points[:, 0], bins= num_bins)
    ax1.hist(bins[:-1], bins, weights=counts, color='g')


    bad_points = low_rank[indices_bad, :]
    counts, bins = np.histogram(bad_points[:, 0], bins= num_bins)
    ax1.hist(bins[:-1], bins, weights=counts, color='r')
    ax1.set_ylim([0, 50])
    ax1.set_title("PC 1")
    plt.show()

    # plt.savefig('../lyapunov-hyperopt-master/Figures/pca/dim_0/threshol_{}.png'.format(threshold), dpi=200)
    # plt.show()
    # Second Dimension
    ax2 = plt.subplot(3, 2, 1)
    # good_points = low_rank[indices_good, :]
    good_points = low_rank
    counts, bins = np.histogram(good_points[:, 1], bins= num_bins)
    ax2.hist(bins[:-1], bins, weights=counts, orientation="horizontal", color='g')

    bad_points = low_rank[indices_bad, :]
    counts, bins = np.histogram(bad_points[:, 1], bins= num_bins)
    ax2.hist(bins[:-1], bins, weights=counts, orientation="horizontal", color='r')
    ax2.set_xlim([0, 50])
    ax2.set_title("PC 2")

    ax3 = plt.subplot(3, 2, 3)
    ax3.hist2d(good_points[:,0], good_points[:,1 ], bins=20)
    ax3.set_xlim([-8.5, 2.5])
    ax3.set_ylim([0, 5.5])
    ax3.set_title("Good performance")

    ax4 = plt.subplot(3, 2, 5)
    ax4.hist2d(bad_points[:,0], bad_points[:,1 ], bins=20)
    ax4.set_xlim([-8.5, 2.5])
    ax4.set_ylim([0, 5.5])
    ax4.set_title("Bad Performance")
    # plt.savefig('../lyapunov-hyperopt-master/Figures/pca/combined/threshold_{}.png'.format(threshold), dpi=200)
    plt.show()

def pca_scatter_stack(dim, path, low_rank=None):
    if not low_rank:
        pca_results = torch.load(path + 'PCA_dim{}.p'.format(dim))
        low_rank, targets = pca_results['low_rank'], pca_results['targets']

    num_colors = 5
    threshold_range = [max(targets), .8, .6, .4, .2, 0]
    # threshold_range = [0, .1, .2, .4, .6, max(targets)]
    gradient = np.linspace(1, 0.5, num_colors)
    cmap = plt.cm.get_cmap('brg')
    plt.figure()
    print(len(targets), len(low_rank[:,0]))
    color_indices = []
    for target in targets:
        for i in range(num_colors):
            # target = target.item()
            if (target <= threshold_range[i]) and (target > threshold_range[i+1]):
                color_indices.append(i)
    print(color_indices)
    gradient = np.linspace(0.5, 1, num_colors)
    cmap = plt.cm.get_cmap('brg')
    rgbs = cmap(gradient[color_indices])
    plt.scatter(low_rank[:,1], low_rank[:, 0], c=rgbs, s=100, alpha=0.5)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()


    # for i in range(num_colors):
    #
    #     bool_arr = (targets < threshold_range[i] and targets > threshold_range[i + 1])
    #     indices_arr = np.where(bool_arr)[0]
    #     points = low_rank[indices_arr, :]
    #     counts, bins = np.histogram(points[:, 0], bins=num_bins)
    #     plt.hist(bins[:-1], bins, weights=counts, color=cmap(gradient[num_colors - i - 1]))
    # plt.xticks([],[])
    # plt.yticks([],[])
    # plt.show()


def pca_hist_stack(dim, path, low_rank=None, num_bins=50):
    if not low_rank:
        pca_results = torch.load(path + 'PCA_dim{}.p'.format(dim))
        low_rank, targets = pca_results['low_rank'], pca_results['targets']

    num_colors = 5
    # [0, .1, .4, .6, .7, max(val_losses_gs)]
    threshold_range = [0, .1, .4, .6, .7]
        # [max(targets), .8, .6, .4, .2]
    # threshold_range = [0, .1, .2, .4, .6, max(targets)]
    gradient = np.linspace(1, 0.5, num_colors)
    cmap = plt.cm.get_cmap('brg')
    # rgbs = cmap(gradient[color_indices])
    # min_range = min(targets).item()
    plt.figure()
    for i in range(num_colors):
        bool_arr = targets > threshold_range[i]
        indices_arr = np.where(bool_arr)[0]
        points = low_rank[indices_arr, :]
        # if i == 0:
        # print(min_range, max(targets))
        counts, bins = np.histogram(points[:, 0], bins=num_bins, range=(min(low_rank[:,0]), max(low_rank[:,0])))
        # else:
        #     counts, _ = np.histogram(points[:, 0], bins=num_bins)
        plt.hist(bins[:-1], bins, weights=counts, color=cmap(gradient[i]))
    plt.xticks([],[])
    # plt.yticks([],[])
    # plt.xticks([-.8, 0, .8], [])
    # plt.xlim([-.8, .8])
    plt.yticks([0, 250, 500], [])
    plt.ylim([0, 500])
    plt.title("PC1 stacked hist")
    plt.show()

    plt.figure()
    for i in range(num_colors):
        bool_arr = targets > threshold_range[i]
        indices_arr = np.where(bool_arr)[0]
        points = low_rank[indices_arr, :]
        # if i == 0:
        # print(min_range, max(targets))
        counts, bins = np.histogram(points[:, 1], bins=num_bins, range=(min(low_rank[:,1]), max(low_rank[:,1])))
        # else:
        #     counts, _ = np.histogram(points[:, 0], bins=num_bins)
        plt.hist(bins[:-1], bins, weights=counts, color=cmap(gradient[i]))
    plt.xticks([],[])
    plt.yticks([0, 250, 500], [])
    plt.ylim([0, 500])
    plt.title("PC2 stacked hist")
    plt.show()





def visualization(targets, predictions):
    plt.figure()
    plt.scatter(torch.ones_like(targets) , targets)
    plt.scatter(torch.ones_like(predictions) * 1.1, predictions.detach())
    plt.legend(["Targets", 'Predictions'])
    # plt.axis([0.95, 1.15, -.1, 1.])
    plt.show()

def main():
    gs = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # g = 1.4
    N = 512
    inputs_epoch = 5
    val_split = 0.2
    dim = 2
    isRFORCE = False
    if isRFORCE:
        distribution = 'RFORCE'
    else:
        distribution = "FORCE"

    function_type = "random_4sine"
    prediction_loss_type = 'BCE'
    if len(gs) > 1:
        Results_path = 'Results/{}/N_{}_g_mixed/epoch_{}/'.format(function_type, N, inputs_epoch)
        data_path = "training_data/{}/{}/non_interpreted/g_mixed/{}_epoch_{}_N_{}".format(
            distribution, function_type, function_type, inputs_epoch, N)
        data = pickle.load(open(data_path, 'rb'))

    else:
        g = gs[0]
        Results_path = 'Results/{}/N_{}_g_{}/epoch_{}/'.format(function_type, N, g, inputs_epoch)
        data_path = "training_data/{}/{}/non_interpreted/g_{}/{}_epoch_{}_N_{}".format(
            distribution, function_type, g, function_type, inputs_epoch, N)
        data = pickle.load(open(data_path, 'rb'))

    model = torch.load(Results_path+'{}/ae_prednet_4000.ckpt'.format(prediction_loss_type))
    model.load_state_dict(model.best_state)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')



    inputs, targets = data['inputs'], data['targets']

    # split = torch.load(Results_path + 'data_split_vfrac{}.p'.format(val_split))

    rec_outputs, hidden_outputs, pred_outputs = model(inputs)
    visualization(targets, pred_outputs)


    # RFORCE
    # RFORCE_data_path = "training_data/RFORCE/{}/g_{}/{}_epoch_{}_N_{}".format(
    #     function_type, g, function_type, inputs_epoch, N)
    # RFORCE_data = pickle.load(open(RFORCE_data_path, 'rb'))
    #
    # RFORCE_inputs, RFORCE_targets = RFORCE_data['inputs'], RFORCE_data['targets']
    #
    # _, RFORCE_hidden_outputs, RFORCE_pred_outputs = model(RFORCE_inputs)
    # # visualization(RFORCE_targets, RFORCE_pred_outputs)
    # hidden_outputs = torch.cat((hidden_outputs, RFORCE_hidden_outputs), 0)
    # targets = torch.cat((targets, RFORCE_targets), 0)
    # index = [0, 706, 776]

    # tsne
    # for i in range(9, 10):
    #     threshold = (i + 1) * 0.01
    #     tsne_results = tsne(X=inputs, targets=targets, dim=dim)
    #     torch.save(tsne_results, Results_path + '{}/tsne.p'.format(prediction_loss_type))
    #     # tsne_distribution(dim, Results_path+prediction_loss_type+'/', index)
    #     tsne_perf(dim, Results_path+prediction_loss_type+'/')
    #     tsne_binary(dim, Results_path+prediction_loss_type+'/', threshold=threshold)
    #     tsne_hist(dim, Results_path+prediction_loss_type+'/', threshold=threshold)


    # pca
    for i in range(19, 20):
        threshold = (i + 1) * 0.01
        pca_results = pca(X=hidden_outputs, dim=dim, targets=targets)
        torch.save(pca_results, '{}/{}/PCA_dim{}.p'.format(Results_path, prediction_loss_type, dim))
        # pca_distribution(dim, Results_path + prediction_loss_type + '/', index)
        # pca_perf(dim=dim, path=Results_path + prediction_loss_type + '/')
        # pca_binary(dim=dim, path=Results_path + prediction_loss_type + '/', threshold=threshold)
        pca_hist(dim, Results_path + prediction_loss_type + '/', threshold=threshold, num_bins = 20)
        pca_hist_stack(dim, path=Results_path + prediction_loss_type + '/', low_rank=None, num_bins=10)
        pca_scatter_stack(dim, path=Results_path + prediction_loss_type + '/', low_rank=None)


if __name__ == "__main__":
    main()
