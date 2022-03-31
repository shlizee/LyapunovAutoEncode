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
        ax0.legend()
        ax0.set_title(threshold)
    elif dim==3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_embedded[indices_bad, 0], x_embedded[indices_bad, 1], x_embedded[indices_bad, 2],
                    s=15, c='r', label="Bad performace")
        ax.scatter(x_embedded[indices_good, 0], x_embedded[indices_good, 1], x_embedded[indices_good, 2],
                    s=6, c='g', label="Good performace")
        plt.legend()
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
    ax2.set_xlim([0, 50])

    ax3 = plt.subplot(3, 2, 3)
    ax3.hist2d(good_points[:,0], good_points[:,1 ], bins=20)
    ax3.set_xlim([-80,80])
    ax3.set_ylim([-55, 55])
    ax3.set_title("Good performance")

    ax4 = plt.subplot(3, 2, 5)
    ax4.hist2d(bad_points[:,0], bad_points[:,1 ], bins=20)
    ax4.set_xlim([-80,80])
    ax4.set_ylim([-55, 55])
    ax4.set_title("Bad Performance")

    # plt.savefig('../lyapunov-hyperopt-master/Figures/tsne/combined/threshold_{}.png'.format(threshold), dpi=200)

    plt.show()

def pca(X, targets, dim=2):
    U,S,V = torch.pca_lowrank(X)
    low_rank = torch.matmul(X, V[:, :dim]).detach().numpy()
    fig = plt.figure()
    if (dim==2):
        plt.scatter(low_rank[:, 0], low_rank[:, 1], s=6)
    elif (dim==3):
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(low_rank[:, 0], low_rank[:, 1],
                   low_rank[:, 2], s=6)
    plt.legend()
    plt.title("PCA")
    plt.show()
    pca_results = {"low_rank": low_rank, "targets": targets}
    return pca_results

def pca_perf(dim, path, low_rank=None, targets=None):
    if not low_rank:
        pca_results = torch.load(path + 'PCA_dim{}.p'.format(dim))
        low_rank, targets = pca_results['low_rank'], pca_results['targets']
    fig = plt.figure()
    if (dim == 2):
        ax = fig.add_subplot(111)
        p = ax.scatter(low_rank[:, 0], low_rank[:, 1], s=6, c = targets,
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
        ax0 = plt.subplot(3, 2, 2)
        ax0.scatter(low_rank[indices_bad, 0], low_rank[indices_bad, 1],  s=15, c='r', label="Bad performace")
        ax0.scatter(low_rank[indices_good, 0], low_rank[indices_good, 1], c='g',s=6, label="Good performace")
        ax0.set_xlabel('PC 1')
        ax0.set_ylabel('PC 2')
        plt.legend()
        ax0.set_title(threshold)
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
        plt.legend()
        plt.title(threshold)
    # plt.savefig('../lyapunov-hyperopt-master/Figures/pca/binary/threshol_{}.png'.format(threshold), dpi=200)
    # plt.show()

def pca_hist(dim, path, low_rank=None, threshold=0.1, num_bins = 50 ):
    if not low_rank:
        pca_results = torch.load(path + 'PCA_dim{}.p'.format(dim))
        low_rank, targets = pca_results['low_rank'], pca_results['targets']

    bool_arr_bad = targets > threshold
    bool_arr_good = targets <= threshold
    indices_bad = np.where(bool_arr_bad)[0]
    indices_good = np.where(bool_arr_good)[0]

    ax1 = plt.subplot(3, 2, 4)

    # First Dimension
    good_points = low_rank[indices_good, :]
    counts, bins = np.histogram(good_points[:, 0], bins= num_bins)
    ax1.hist(bins[:-1], bins, weights=counts, color='g')

    bad_points = low_rank[indices_bad, :]
    counts, bins = np.histogram(bad_points[:, 0], bins= num_bins)
    ax1.hist(bins[:-1], bins, weights=counts, color='r')
    ax1.set_ylim([0, 50])
    ax1.set_title("PC 1")

    # plt.savefig('../lyapunov-hyperopt-master/Figures/pca/dim_0/threshol_{}.png'.format(threshold), dpi=200)
    # plt.show()
    # Second Dimension
    ax2 = plt.subplot(3, 2, 1)
    good_points = low_rank[indices_good, :]
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

def visualization(targets, predictions):
    plt.figure()
    plt.scatter(torch.ones_like(targets) , targets)
    plt.scatter(torch.ones_like(predictions) * 1.1, predictions.detach())
    plt.legend(["Targets", 'Predictions'])
    # plt.axis([0.95, 1.15, -.1, 1.])
    plt.show()

def main():
    g = 1.4
    N = 512
    inputs_epoch = 10
    val_split = 0.2
    dim = 2
    isRFORCE = False
    if isRFORCE:
        distribution = 'RFORCE'
    else:
        distribution = "FORCE"

    function_type = "random_4sine"
    prediction_loss_type = 'BCE'
    Results_path = 'Results/{}/N_{}_g_{}/epoch_{}/'.format(function_type, N, g, inputs_epoch)
    model = torch.load(Results_path+'{}/ae_prednet_4000.ckpt'.format(prediction_loss_type))
    model.load_state_dict(model.best_state)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data_path = "training_data/{}/{}/g_{}/{}_epoch_{}_N_{}".format(
        distribution, function_type, g, function_type, inputs_epoch, N)
    data = pickle.load(open(data_path, 'rb'))
    inputs, targets = data['inputs'], data['targets']

    split = torch.load(Results_path + 'data_split_vfrac{}.p'.format(val_split))

    rec_outputs, hidden_outputs, pred_outputs = model(inputs)
    visualization(targets, pred_outputs)

    # tsne
    for i in range(9, 10):
        threshold = (i + 1) * 0.01
        tsne_results = tsne(X=hidden_outputs, targets=targets, dim=dim)
        torch.save(tsne_results, Results_path + '{}/tsne.p'.format(prediction_loss_type))
        tsne_perf(dim, Results_path+prediction_loss_type+'/')
        tsne_binary(dim, Results_path+prediction_loss_type+'/', threshold=threshold)
        tsne_hist(dim, Results_path+prediction_loss_type+'/', threshold=threshold)

    # pca
    # for i in range(14, 15):
    #     threshold = (i + 1) * 0.01
    #     pca_results = pca(X=hidden_outputs, dim=dim, targets=targets)
    #     torch.save(pca_results, '{}/{}/PCA_dim{}.p'.format(Results_path, prediction_loss_type, dim))
    #     pca_perf(dim=dim, path=Results_path + prediction_loss_type + '/')
    #     pca_binary(dim=dim, path=Results_path + prediction_loss_type + '/', threshold=threshold)
    #     pca_hist(dim, Results_path + prediction_loss_type + '/', threshold=threshold)



if __name__ == "__main__":
    main()
