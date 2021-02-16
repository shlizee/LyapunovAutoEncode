import torch
from AEPredNet import AEPredNet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import colors
import pickle
import seaborn as sns
import pandas as pd

def tsne(X, targets, dim=2):
    tsne_model = TSNE(perplexity=10,n_components=dim, random_state=1)

    x_embedded = tsne_model.fit_transform(X.detach().numpy())

    # x_embedded = tsne(hidden_outputs.detach(), dim=tsne_dim)

    fig = plt.figure()
    if (dim==2):
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=6)
        plt.xlim([-60, 40])
        plt.ylim([-60, 80])
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
        p = ax.scatter(x_embedded[:, 0], x_embedded[:, 1], s=6, c = targets,
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

def visualization(targets, predictions):
    plt.figure()
    plt.scatter(torch.ones_like(targets) , targets)
    plt.scatter(torch.ones_like(predictions) * 1.1, predictions.detach())
    plt.legend(["Targets", 'Predictions'])
    # plt.axis([0.95, 1.15, -.1, 1.])
    plt.show()

def main():
    g = 1.5
    N = 512
    inputs_epoch = 7
    val_split = 0.1
    dim = 2
    prediction_loss_type = 'MSE'
    Results_path = 'Results/N_{}_g_{}/epoch_{}/'.format(N, g, inputs_epoch)
    model = torch.load(Results_path+'{}/ae_prednet_4000.ckpt'.format(prediction_loss_type))
    model.load_state_dict(model.best_state)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data_path = "training_data/g_{}/4sine_epoch_{}_N_{}".format(g, inputs_epoch, N)
    data = pickle.load(open(data_path, 'rb'))
    inputs, targets = data['inputs'], data['targets']

    split = torch.load(Results_path + 'data_split_vfrac{}.p'.format(val_split))

    rec_outputs, hidden_outputs, pred_outputs = model(inputs)
    visualization(targets, pred_outputs)

    # tsne
    tsne_results = tsne(X=hidden_outputs, targets=targets, dim=dim)
    torch.save(tsne_results, Results_path + '{}/tsne.p'.format(prediction_loss_type))
    tsne_perf(dim, Results_path+prediction_loss_type+'/')

    # pca
    pca_results = pca(X=hidden_outputs, dim=dim, targets=targets)
    torch.save(pca_results, '{}/{}/PCA_dim{}.p'.format(Results_path, prediction_loss_type, dim))
    pca_perf(dim=dim, path=Results_path + prediction_loss_type + '/')




if __name__ == "__main__":
    main()
