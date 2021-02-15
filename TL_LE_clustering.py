import torch
from AEPredNet import AEPredNet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import colors
import pickle
import seaborn as sns
import pandas as pd

def tsne(model, X, tsne_params={}):
    encoded = model(X)[1]
    tsne_model = TSNE(**tsne_params,n_components=2, random_state=1)
    X_embedded = tsne_model.fit_transform(encoded.detach().numpy())
    print(X_embedded.shape)
    return tsne_model


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = torch.load('Results/N_512_g_1.5/epoch_5/ae_prednet_4000.ckpt')
    model.load_state_dict(model.best_state)

    inputs_epoch = 13
    target_epoch = 14
    N = 512

    data_path = "training_data/g_1.5/4sine_epoch_{}_N_{}".format(inputs_epoch, N)
    data = pickle.load(open(data_path, 'rb'))

    x_data, targets = data['inputs'], data['targets']
    print(targets)
    # plt.figure()
    # plt.scatter(torch.ones_like(targets) , targets)
    # plt.scatter(torch.ones_like(outputs[-1]) * 1.1, outputs[-1].detach())
    # plt.axis([0.9, 1.1, -.1, 1.1])
    # plt.show()

    split = torch.load('Results/N_512_g_1.5/epoch_13/data_split_vfrac0.2.p')

    indices = [0, len(targets)]
    gs = [1.5]

    tsne_model = tsne(model, split['train_data'], tsne_params={'perplexity': 10})
    i_list = torch.arange(indices[-1])
    splits = [(i_list > torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1]) for
              i in range(len(indices) - 1)]
    outputs = model(x_data)
    Y = tsne_model.fit_transform(outputs[1].detach())

    plt.figure()
    plt.scatter(torch.ones_like(targets), targets)
    plt.scatter(torch.ones_like(outputs[-1]) * 1.1, outputs[-1].detach())
    plt.show()
    print(Y.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection='3d')
    for idx, g in enumerate(gs):
        y = Y[splits[idx]]
        ax.scatter(y[:, 0], y[:, 1], s=6, label=gs[idx])
        # ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=6, label=gs[idx])
    plt.legend()
    # plt.xlim([-60, 40])
    # plt.ylim([-60, 80])
    torch.save(Y, 'tsne.p')
    plt.savefig('../lyapunov-hyperopt-master/Figures/AEPredNet_tsne_size.png', dpi=200)
    plt.show()

def pca_plot():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = torch.load('ae_prednet_4000.ckpt')
    model.load_state_dict(model.best_state)

    inputs_epoch = 4
    target_epoch = 14
    N = 512
    data_path = "training_data/4sine_epoch_{}_N_512".format(inputs_epoch, N)
    data = pickle.load(open(data_path, 'rb'))
    x_data, targets = data['inputs'], data['targets']

    x_data, targets = x_data[300:550, :], targets[300:550].detach()
    # x_data = torch.load('Processed/lstm_allLEs.p')
    split = torch.load('data_split_vfrac0.2.p')
    # indices = [0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550 ,600]
    # gs = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    indices = [0, 250]
    gs = [1.4]

    splits = []

    i_list = torch.arange(indices[-1])
    splits = [(i_list > torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1]) for
              i in range(len(indices) - 1)]

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x_data)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1',
                                          'principal component 2'])
    finalDf = pd.concat([principalDf, targets], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    colors = ['r','g','b']
    for i, color in  enumerate(colors):
        indicesToKeep = targets < 0.3 * (i+1)
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()

    # plt.xlim([-60, 40])
    # plt.ylim([-60, 80])
    # torch.save(Y, 'pca.p')
    # plt.savefig('../lyapunov-hyperopt-master/Figures/AEPredNet_tsne_size.png', dpi=200)
    plt.show()
def param_plot():
    model = torch.load('ae_prednet_4000.ckpt')
    model.load_state_dict(model.best_state)
    x_data = torch.load('Processed/lstm_allLEs.p')
    params = torch.load('lstm_allParams.p')
    split = torch.load('data_split_vfrac0.2.p')
    indices = [0, 300, 600, 900, 1200]
    sizes = [64, 128, 256, 512]
    val_idx = split['val_idx']
    val_splits = []
    for i in range(len(sizes)):
        val_splits.append(
            ((val_idx > torch.ones_like(val_idx) * indices[i]) * (val_idx < torch.ones_like(val_idx) * indices[i + 1])))
        # print(torch.arange(1200).float()>torch.ones(1200)*indices[i])
        # print(val_idx[val_splits[i]].shape)
        plt.scatter(params[val_idx[val_splits[i]]], model(x_data[val_idx[val_splits[i]]])[2].detach(), label=sizes[i],
                    s=14)
    plt.legend()
    plt.xlabel('Init Param')
    plt.ylim([1.1, 2.6])
    plt.ylabel('Validation Loss \n (Predicted)')
    plt.title('AE Predictions')
    plt.savefig('../lyapunov-hyperopt-master/Figures/AEPredNet_paramPlot.png', bbox_inches="tight", dpi=200)

    plt.figure()
    targets = torch.load('Processed/lstm_allValLoss.p')
    for i in range(4):
        plt.scatter(params[val_idx[val_splits[i]]], targets[val_idx[val_splits[i]]], label=sizes[i], s=14)
    plt.legend(prop={'size': 12})
    plt.ylabel('Val Loss\n(Actual)')
    plt.xlabel('Init Param')
    plt.ylim([1.1, 2.6])
    plt.title(f'Ground Truth')
    plt.savefig('Actual_paramPlot.png', bbox_inches="tight", dpi=200)


def tsne_perf():

    inputs_epoch = 4
    target_epoch = 14
    N = 512
    data_path = "training_data/g_1.5/4sine_epoch_{}_N_{}".format(inputs_epoch, N)
    data = pickle.load(open(data_path, 'rb'))
    x_data, targets = data['inputs'], data['targets']
    # indices = [0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550 ,600]
    # gs = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    x_data, targets = x_data[0:396, :], targets[0:396]
    # x_data, targets = x_data[300:550, :], targets[300:550]
    # x_data, targets = x_data[550: 800, :], targets[550:800]
    indices = [0, 396]
    gs = [1.4]
    # indices = [0, 300, 550, 800]
    # gs = [1.4, 1.5, 1.6]


    # indices = [0, 300, 600, 900, 1200]
    # sizes = [64, 128, 256, 512]
    # targets = torch.load('Processed/lstm_allValLoss.p')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection='3d')

    splits = []
    i_list = torch.arange(max(indices))
    splits = [(i_list > torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1]) for
              i in range(len(indices) - 1)]
    Y = torch.load('tsne.p')
    for idx, g in enumerate(gs):
        y = Y[splits[idx]]
        # p = ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=6, c=targets[splits[idx]],
        #             norm=colors.LogNorm(vmin=targets.min(), vmax=targets.max()))
        p = ax.scatter(y[:, 0], y[:, 1], s=6, c=targets[splits[idx]],
                    norm=colors.LogNorm(vmin=targets.min(), vmax=targets.max()))
    fig.colorbar(p, label='Val Loss')
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    # plt.xlim([-60, 40])
    # plt.ylim([-60, 80])
    # plt.savefig('../lyapunov-hyperopt-master/Figures/AEPredNet_tsne_performance.png', dpi=200)
    plt.show()

def tsne_param():
    indices = [0, 300, 600, 900, 1200]
    sizes = [64, 128, 256, 512]
    # params = torch.load('lstm_allParams.p')
    splits = []
    i_list = torch.arange(1200)
    splits = [(i_list > torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1]) for
              i in range(len(indices) - 1)]
    Y = torch.load('tsne.p')
    for idx, size in enumerate(sizes):
        y = Y[splits[idx]]
        plt.scatter(y[:, 0], y[:, 1], s=6,
                    c=params[splits[idx]])  # , norm=colors.LogNorm(vmin=params.min(), vmax=params.max()))
    plt.colorbar(label='Init Param')
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    plt.savefig('../lyapunov-hyperopt-master/Figures/AEPredNet_tsne_params.png', dpi=200)


if __name__ == "__main__":
    main()
    # pca_plot()
    # tsne_perf()
# tsne_param()