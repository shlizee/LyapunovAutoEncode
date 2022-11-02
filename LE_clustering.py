import torch
from AEPredNet import AEPredNet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
# from config import *
from scipy.ndimage import rotate
import argparse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

latent_size = 32


def tsne(model, X, tsne_params={}, model_type='lstm'):
    encoded = model(X)[1]
    tsne_model = TSNE(**tsne_params)
    X_embedded = tsne_model.fit_transform(encoded.detach().numpy())
    print(X_embedded.shape)
    return tsne_model


def main(latent_size, model_type='lstm', dir=''):
    parser = argparse.ArgumentParser(description="Train Lyapunov Autoencoder")
    parser.add_argument("-model", "--model_type", type=str, default='rnn', required=False)
    parser.add_argument("-task", "--task_type", type=str, default='SMNIST', required=False)
    parser.add_argument("-latent", "--latent_size", type=int, default=32, required=False)
    parser.add_argument("-max_epoch", "--max_epoch", type=int, default=4000, required=False)
    parser.add_argument("-evals", "--evals", type=int, default=4000, required=False)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model_dir = f'{task_type}/AE_Models/{model_type}/Latent_{latent_size}'
    model = torch.load(f'{model_dir}/ae_prednet_{max_epoch}.ckpt').cpu()
    model.load_state_dict(model.best_state)
    data_dir = f'{task_type}/Processed'
    x_data = torch.load(f'{data_dir}/{model_type}_allLEs.p')
    targets = torch.load(f'{data_dir}/{model_type}_allValLoss.p')
    split = torch.load(f'{data_dir}/{model_type}_data_split_vfrac0.2.p')
    indices = [0, 300, 600, 900, 1200]
    sizes = [64, 128, 256, 512]
    thresh = torch.mean(targets)

    low_rank = pca(latent_size, dim=2, model_type=model_type, no_evals=evals, threshold=thresh)
    plt.figure(2)
    linewidth = 1
    plt.scatter(low_rank[~target_mask, 0], low_rank[~target_mask, 1], s=20, c='red', alpha=.5, label='Low')
    plt.scatter(low_rank[target_mask, 0], low_rank[target_mask, 1], s=20, c='lime', alpha=.5, label='High')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('Coloring by Accuracy')
    plt.savefig(f'{task_type}/Figures/Latent/{model_type}_pca_perf.png', bbox_inches='tight', dpi=200)


    # tsne_model = tsne(model, split['train_data'], tsne_params={'perplexity': 10})
    # splits = []
    # i_list = torch.arange(1200)
    # splits = [(i_list > torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1]) for
    #           i in range(len(indices) - 1)]
    # Y = tsne_model.fit_transform(model(x_data)[1].detach())
    # plt.figure()
    # for idx, size in enumerate(sizes):
    #     y = Y[splits[idx]]
    #     plt.scatter(y[:, 0], y[:, 1], s=6, label=size)
    # plt.legend()
    # torch.save(Y, f'{data_dir}/{model_type}_tsne.p')
    # plt.savefig(f'{task_type}/Figures/Latent/{model_type}_AEPredNet_tsne_size.png', dpi=200)


def param_plot(latent_size, model_type='lstm', dir='', no_evals=300, val_split=0.2):
    model = torch.load(f'{task_type}/AE_Models/{model_type}/Latent_{latent_size}/ae_prednet_4000.ckpt').cpu()
    model.load_state_dict(model.best_state)
    x_data = torch.load(f'{task_type}/Processed/{model_type}_allLEs.p')
    print(f'{model_type} Shape: {x_data.shape}')
    params = torch.load(f'{task_type}/Processed/{model_type}_allParams.p').flatten()
    split = torch.load(f'{task_type}/Processed/{model_type}_data_split_vfrac{val_split}.p')
    indices = [0, 1 * no_evals, 2 * no_evals, 3 * no_evals, 4 * no_evals]
    sizes = [64, 128, 256, 512]
    val_idx = split['val_idx']
    val_splits = []
    plt.figure()
    for i in range(len(sizes)):
        val_splits.append(
            ((val_idx > torch.ones_like(val_idx) * indices[i]) * (val_idx < torch.ones_like(val_idx) * indices[i + 1])))
        # print(torch.arange(1200).float()>torch.ones(1200)*indices[i])
        # print(val_idx[val_splits[i]].shape)
        plt.scatter(params[val_idx[val_splits[i]]], model(x_data[val_idx[val_splits[i]]])[2].detach(), label=sizes[i],
                    s=14)
    plt.legend(loc=2)
    plt.xlabel('Init Param')
    plt.ylim([1.1, 3.0])
    plt.ylabel('Validation Loss \n (Predicted)')
    plt.title('AE Predictions')
    plt.savefig(f'{task_type}/Figures/AEPredNet_{model_type}_paramPlot.png', bbox_inches="tight", dpi=200)

    plt.figure()
    targets = torch.load(f'{task_type}/Processed/{model_type}_allValLoss.p')
    for i in range(4):
        plt.scatter(params[val_idx[val_splits[i]]], targets[val_idx[val_splits[i]]], label=sizes[i], s=14)
    plt.legend(prop={'size': 12}, loc=2)
    plt.ylabel('Val Loss\n(Actual)')
    plt.xlabel('Init Param')
    plt.ylim([1.1, 3.0])
    plt.title(f'Ground Truth')
    plt.savefig(f'{task_type}/Figures/Actual_{model_type}_paramPlot.png', bbox_inches="tight", dpi=200)


def tsne_perf(no_evals=400, model_type='lstm'):
    indices = [0, 1 * no_evals, 2 * no_evals, 3 * no_evals, 4 * no_evals]
    sizes = [64, 128, 256, 512]
    targets = torch.load(f'Processed/{model_type}_allValLoss.p')
    splits = []
    i_list = torch.arange(len(sizes) * no_evals)
    splits = [(i_list > torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1]) for
              i in range(len(indices) - 1)]
    Y = torch.load('tsne.p')
    plt.figure()
    for idx, size in enumerate(sizes):
        y = Y[splits[idx]]
        plt.scatter(y[:, 0], y[:, 1], s=6, c=targets[splits[idx]],
                    norm=colors.LogNorm(vmin=targets.min() - 0.2, vmax=targets.max()), cmap=plt.get_cmap('hsv'))
    plt.colorbar(label='Val Loss')
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    plt.savefig('AEPredNet_tsne_performance.png', dpi=200)


def tsne_param(no_evals=400, model_type='lstm'):
    indices = [0, 1 * no_evals, 2 * no_evals, 3 * no_evals, 4 * no_evals]
    sizes = [64, 128, 256, 512]
    params = torch.load(f'Processed/{model_type}_allParams.p')
    splits = []
    i_list = torch.arange(4 * no_evals)
    splits = [(i_list > torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1]) for
              i in range(len(indices) - 1)]
    Y = torch.load('tsne.p')
    plt.figure()
    for idx, size in enumerate(sizes):
        y = Y[splits[idx]]
        plt.scatter(y[:, 0], y[:, 1], s=6,
                    c=params[splits[idx]])  # , norm=colors.LogNorm(vmin=params.min(), vmax=params.max()))
    plt.colorbar(label='Init Param')
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    plt.savefig('AEPredNet_tsne_params.png', dpi=200)


# noinspection PyShadowingNames
def perturbation(latent_size, dim=2, model_type='lstm', no_evals=300, v_frac=0.2, delta=1, task_type='', thresh=1):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # model_type = 'all'
    load_epoch = 4000
    model_name = f'{task_type}/AE_Models/{model_type}/Latent_{latent_size}/ae_prednet_{load_epoch}.ckpt'
    print(model_name)
    model = torch.load(model_name).cpu()
    model.load_state_dict(model.best_state)
    data_dir = f'{task_type}/Processed/'
    split_lstm = torch.load(f'{data_dir}/{model_type}_data_split_vfrac0.1_testfrac0.2.p')
    x_data_lstm = split_lstm['train_data'].detach()
    targets_lstm = split_lstm['train_targets']

    x_data = x_data_lstm
    targets = targets_lstm

    num_lstm = 840
    indices = [num_lstm]
    plt.figure(3)
    plt.scatter(range(0, indices[0]), targets[:indices[0]], s=20, c='b', alpha=.5, label='lstm')
    plt.legend()
    plt.show()

    target_mask = targets < thresh
    outputs = model(x_data)
    latent = outputs[1].detach()
    print(f"Number of samples smaller than threshold: {torch.sum(target_mask)}")
    print(f"Number of samples larger than threshold: {torch.sum(~target_mask)}")

    U, S, V = torch.pca_lowrank(latent)
    low_rank = torch.matmul(latent, V[:, :dim])

    x_data_lstm = split_lstm['val_data'].detach()
    targets_lstm = split_lstm['val_targets']
    x_data_gru = split_gru['val_data'].detach()
    targets_gru = split_gru['val_targets']
    x_data_rnn = split_rnn['val_data'].detach()
    targets_rnn = split_rnn['val_targets']
    x_data_asrnn = split_asrnn['val_data'].detach()
    targets_asrnn = split_asrnn['val_targets']
    x_data_cornn = split_cornn['val_data'].detach()
    targets_cornn = split_cornn['val_targets']

    x_data = torch.cat((x_data_lstm, x_data_gru, x_data_rnn, x_data_asrnn, x_data_cornn), dim=0)
    targets = torch.cat((targets_lstm, targets_gru, targets_rnn, targets_asrnn, targets_cornn))
    # thresh = 0.02
    thresh = torch.median(targets)

    target_mask = targets < thresh
    print(f'thresh: {thresh}')
    print(f"Number of samples smaller than threshold: {torch.sum(target_mask)}")
    print(f"Number of samples larger than threshold: {torch.sum(~target_mask)}")

    outputs = model(x_data)
    predictions = outputs[2].detach()

    latent = outputs[1].detach()
    if os.path.exists(f'{task_type}/Processed/{model_type}_PCA_dim{dim}.p'):
        trained_project = torch.load(f'{task_type}/Processed/{model_type}_PCA_dim{dim}.p')
        V = trained_project['V']
    else:
        U, S, V = torch.pca_lowrank(latent)
    low_rank = torch.matmul(latent, V[:, :dim])
    trained_project = {'V': V, 'low_rank': low_rank, 'latent': latent}
    torch.save(trained_project, f'{task_type}/Processed/{model_type}_PCA_dim{dim}.p')
    #
    fig = plt.figure(figsize=(4, 4))
    pc1_thresholds = torch.arange(-2, 2, 0.1)
    f1 = torch.zeros(len(pc1_thresholds))
    for i, pc1_threshold in enumerate(pc1_thresholds):
        f1[i] = f1_score(low_rank, target_mask, pc1_threshold, dim=0, inverse=False, verbose=False)
    print(f'f1 score with threshold ({pc1_thresholds}) is: {f1}')
    idx = torch.argmax(f1)
    optimal_threshold = pc1_thresholds[idx]
    print(f'optimal_threshold: {optimal_threshold}, optimal f1: {f1[idx]}')
    plt.scatter(low_rank[target_mask, 0], low_rank[target_mask, 1], s=20, c='lime', alpha=0.5)
    plt.scatter(low_rank[~target_mask, 0], low_rank[~target_mask, 1], s=20, c='red', alpha=0.5)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('val')
    # plt.xlim([-5.5, 5])
    # plt.ylim([-5, 5])
    plt.show()

    indices = [0, 1 * no_evals, 2 * no_evals, 3 * no_evals, 4 * no_evals]
    print(f"indices: {indices}")
    # sizes = [64, 128, 256, 512]
    i_list = torch.arange(4 * no_evals)
    splits = [(i_list > torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1]) for
              i in range(len(indices) - 1)]

    model_type = 'lstm'
    x_data = torch.load(f'Processed/trials/SMNIST/{model_type}/{model_type}_allLEs.p')
    targets = torch.load(f'Processed/trials/SMNIST/{model_type}/{model_type}_allValLoss.p')
    target_mask = targets < thresh
    epochs = [0, 1, 2, 3, 4, 5]
    indices = [0, 1 * no_evals, 2 * no_evals, 3 * no_evals, 4 * no_evals, 5 * no_evals, 6 * no_evals]
    i_list = torch.arange(len(epochs) * no_evals)
    splits = [(i_list >= torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1]) for
              i in range(len(indices) - 1)]

    device = torch.device('cpu')
    model = torch.load(f'Models/Latent_{latent_size}/ae_prednet_4000.ckpt', map_location=device)
    model.load_state_dict(model.best_state)
    new_model = AEPredNet(model.input_size, model.latent_size)
    new_model.load_state_dict(model.best_state)
    x_data = torch.load(f'Processed/{model_type}/{model_type}_allLEs.p')
    split = torch.load(f'Processed/{model_type}_data_split_vfrac{v_frac}.p')
    indices = [0, 1 * no_evals, 2 * no_evals, 3 * no_evals, 4 * no_evals]
    sizes = [64, 128, 256, 512]
    i_list = torch.arange(4 * no_evals)
    splits = [(i_list > torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1]) for
              i in range(len(indices) - 1)]
    le_out, latent, preds = model(x_data)
    # print(f'xdata shape: {x_data.shape}')
    # print(f'latent shape: {latent.shape}')
    W = model.prediction.weight
    b = model.prediction.bias
    latent = latent.detach()
    # print(f'W shape: {W.shape}')
    # print(f'preds shape: {preds.shape}')
    del_latent = torch.linalg.lstsq(W, - torch.ones_like(preds).unsqueeze(0) * delta)[0]
    # print(f'del latent shape: {del_latent.shape}')
    l_prime = latent - del_latent.T
    le_perturbed = model.decode(l_prime).detach()
    x = torch.arange(le_perturbed.shape[-1]).unsqueeze(0).repeat(le_perturbed.shape[0], 1)
    keep_num = 5
    plt.figure()
    # plt.scatter(x[:keep_num], x_data[:keep_num], label='original', alpha=1, s=5, c='black')
    plt.scatter(x[:keep_num], le_out[:keep_num].detach(), label='reconstructed', alpha=0.4, s=5, c='cyan')
    # plt.scatter(x[:keep_num], x_data[:keep_num] - le_out[:keep_num].detach(), label='rec diff', s=5, c='red')
    plt.scatter(x[:keep_num], le_perturbed[:keep_num], label='perturbed', alpha=0.3, s=5, c='hotpink')
    plt.scatter(x[:keep_num], le_perturbed[:keep_num] - le_out[:keep_num].detach(), label='difference', s=5,
                c='limegreen')
    plt.legend()
    plt.title(f'LE Latent Perturbations,' + r' $\Delta=$' + str(delta))
    plt.ylim([-12, 2])
    plt.savefig(f'Figures/perturb_latent{latent_size}_delta{delta}.png')

    plt.figure()
    plt.scatter(x[:keep_num], x_data[:keep_num], label='original', alpha=1, s=5, c='black')
    plt.scatter(x[:keep_num], le_out[:keep_num].detach(), label='reconstructed', alpha=0.4, s=5, c='cyan')
    plt.scatter(x[:keep_num], x_data[:keep_num] - le_out[:keep_num].detach(), label='rec diff', s=5, c='red')
    plt.legend()
    plt.title(f'LE Reconstruction')
    plt.ylim([-12, 2])
    plt.savefig(f'Figures/LE_reconstruction_latent{latent_size}.png')

    return l_prime, le_perturbed


def pca(latent_size, dim=2, model_type='lstm', no_evals=300, v_frac=0.2, thresh=1.75):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = torch.load(f'{task_type}/AE_Models/{model_type}/Latent_{latent_size}/ae_prednet_4000.ckpt').cpu()
    model.load_state_dict(model.best_state)
    new_model = AEPredNet(model.input_size, model.latent_size)
    new_model.load_state_dict(model.best_state)
    x_data = torch.load(f'{task_type}/Processed/{model_type}_allLEs.p')
    targets = torch.load(f'{task_type}/Processed/{model_type}_allValLoss.p')
    target_mask = targets < thresh
    # print(f'Target shape {targets.shape}')
    split = torch.load(f'{task_type}/Processed/{model_type}_data_split_vfrac{v_frac}.p')
    indices = [0, 1 * no_evals, 2 * no_evals, 3 * no_evals, 4 * no_evals]
    sizes = [64, 128, 256, 512]
    i_list = torch.arange(len(sizes) * no_evals)
    splits = [(i_list > torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1]) for
              i in range(len(indices) - 1)]
    _, latent, preds = model(x_data)

    U, S, V = torch.pca_lowrank(latent)
    low_rank = torch.matmul(latent, V[:, :dim])
    torch.save(low_rank, f'{task_type}/Processed/{model_type}_PCA_dim{dim}.p')

    return low_rank

    # for idx in [0, 1, 2, 5]:
    #     pca_and_hist(target_mask, splits, idx=idx, low_rank=low_rank, xy_dim=[0, 1], threshold=optimal_threshold,
    #                  inverse=False, verbose=True)


def pca_all(latent_size, dim=2, task_type='SMNIST', model_type='all', no_evals=1, v_frac=0.2, thresh=1.75,
            load_epoch=4000):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # model_type = 'all'
    print(f'Models/{model_type}/Latent_{latent_size}/ae_prednet_{load_epoch}.ckpt')
    model = torch.load(f'Models/{model_type}/Latent_{latent_size}/ae_prednet_{load_epoch}.ckpt').cpu()
    model.load_state_dict(model.best_state)
    dir = f'Processed/trials/{task_type}/{model_type}/'
    split_lstm = torch.load(f'Processed/trials/SMNIST/lstm/lstm_data_split_vfrac0.1_testfrac0.2.p')
    targets_lstm = split_lstm['train_targets']

    # x_data = torch.cat((x_data_lstm, x_data_gru, x_data_rnn, x_data_asrnn, x_data_cornn), dim=0)
    # targets = torch.cat((targets_lstm, targets_gru, targets_rnn, targets_asrnn, targets_cornn))
    targets = targets_lstm

    num_lstm = 840
    # num_gru = 840
    # num_rnn = 840
    # num_asrnn = 840
    # num_cornn = 840
    # indices = [num_lstm, num_lstm + num_gru, num_lstm + num_gru + num_rnn, num_lstm + num_gru + num_rnn + num_asrnn,
    #            num_lstm + num_gru + num_rnn + num_asrnn + num_cornn]
    indices = [num_lstm]

    plt.figure(1)
    plt.scatter(range(0, indices[0]), targets[:indices[0]], s=20, c='b', alpha=.5, label='lstm')
    # plt.scatter(range(indices[0], indices[1]), targets[indices[0]:indices[1]], s=20, c='y', alpha=.5, label='gru')
    # plt.scatter(range(indices[1], indices[2]), targets[indices[1]:indices[2]], s=20, c='g', alpha=.5, label='rnn')
    # plt.scatter(range(indices[2], indices[3]), targets[indices[2]:indices[3]], s=20, c='c', alpha=.5, label='asrnn')
    # plt.scatter(range(indices[3], indices[4]), targets[indices[3]:indices[4]], s=20, c='purple', alpha=.5, label='cornn')
    # plt.title("LSTM, GRU, RNN, ASRNN and CoRNN loss")
    plt.title('LSTM Losses')
    plt.legend()
    plt.show()
    thresh = torch.median(targets)
    # thresh = 0.005
    target_mask = targets <= thresh
    outputs = model(x_data)
    latent = outputs[1].detach()
    print(f"Threshold in training data: {thresh}")
    print(f"Number of samples smaller than threshold: {torch.sum(target_mask)}")
    print(f"Number of samples larger than threshold: {torch.sum(~target_mask)}")

    if os.path.exists(f'{dir}/PCA_dim{dim}.p'):
        print("Loading V")
        trained_project = torch.load(f'{dir}/PCA_dim{dim}.p')
        V = trained_project['V']
    else:
        print("Creating new V")
        U, S, V = torch.pca_lowrank(latent)

    low_rank = torch.matmul(latent, V[:, :dim])
    PC1_mean = torch.mean(low_rank[:, 0])
    PC1_median = torch.median(low_rank[:, 0])
    PC2_mean = torch.mean(low_rank[:, 1])
    PC2_median = torch.median(low_rank[:, 1])
    print(f"PC1 mean: {PC1_mean}, median: {PC1_median}")
    print(f"PC2 mean: {PC2_mean}, median: {PC2_median}")
    plt.figure(2)
    linewidth = 1
    plt.scatter(low_rank[~target_mask, 0], low_rank[~target_mask, 1], s=20, c='red', alpha=.5)
    plt.scatter(low_rank[target_mask, 0], low_rank[target_mask, 1], s=20, c='lime', alpha=.5)

    plt.plot([PC1_mean, PC1_mean], [torch.min(low_rank[:, 1]), torch.max(low_rank[:, 1])], label='PC1 mean',
             linewidth=linewidth, color='m')
    plt.plot([PC1_median, PC1_median], [torch.min(low_rank[:, 1]), torch.max(low_rank[:, 1])], label='PC1 median',
             linewidth=linewidth, color='k')
    plt.plot([torch.min(low_rank[:, 0]), torch.max(low_rank[:, 0])], [PC2_mean, PC2_mean], label='PC2 mean',
             linewidth=linewidth, color='m')
    plt.plot([torch.min(low_rank[:, 0]), torch.max(low_rank[:, 0])], [PC2_median, PC2_median], label='PC2 median',
             linewidth=linewidth, color='k')
    plt.title("LSTM, GRU, RNN, ASRNN and CoRNN training data")
    # plt.legend()
    plt.show()

    plt.figure(3)
    plt.scatter(low_rank[:indices[0], 0], low_rank[:indices[0], 1], s=20, c='b', alpha=.5, label='lstm')
    plt.scatter(low_rank[indices[0]:indices[1], 0], low_rank[indices[0]:indices[1], 1], s=20, c='y', alpha=.5,
                label='gru')
    plt.scatter(low_rank[indices[1]:indices[2], 0], low_rank[indices[1]:indices[2], 1], s=20, c='g', alpha=.5,
                label='rnn')
    plt.scatter(low_rank[indices[2]:indices[3], 0], low_rank[indices[2]:indices[3], 1], s=20, c='c', alpha=.5,
                label='asrnn')
    plt.scatter(low_rank[indices[3]:indices[4], 0], low_rank[indices[3]:indices[4], 1], s=20, c='purple', alpha=.5,
                label='cornn')
    plt.plot([PC1_mean, PC1_mean], [torch.min(low_rank[:, 1]), torch.max(low_rank[:, 1])], label='PC1 mean',
             linewidth=linewidth, color='m')
    plt.plot([PC1_median, PC1_median], [torch.min(low_rank[:, 1]), torch.max(low_rank[:, 1])], label='PC1 median',
             linewidth=linewidth, color='k')
    plt.plot([torch.min(low_rank[:, 0]), torch.max(low_rank[:, 0])], [PC2_mean, PC2_mean], label='PC2 mean',
             linewidth=linewidth, color='m')
    plt.plot([torch.min(low_rank[:, 0]), torch.max(low_rank[:, 0])], [PC2_median, PC2_median], label='PC2 median',
             linewidth=linewidth, color='k')
    plt.title("LSTM, GRU, RNN, ASRNN and CoRNN training data")
    # plt.legend()
    plt.show()
    metric = PC2_median
    metric_dim = 1
    LSTM_right = torch.sum(low_rank[0:indices[0], metric_dim] >= metric)
    GRU_right = torch.sum(low_rank[indices[0]:indices[1], metric_dim] >= metric)
    RNN_right = torch.sum(low_rank[indices[1]:indices[2], metric_dim] >= metric)
    ASRNN_right = torch.sum(low_rank[indices[2]:indices[3], metric_dim] >= metric)
    CoRNN_right = torch.sum(low_rank[indices[3]:indices[4], metric_dim] >= metric)

    LSTM_left = torch.sum(low_rank[0:indices[0], metric_dim] < metric)
    GRU_left = torch.sum(low_rank[indices[0]:indices[1], metric_dim] < metric)
    RNN_left = torch.sum(low_rank[indices[1]:indices[2], metric_dim] < metric)
    ASRNN_left = torch.sum(low_rank[indices[2]:indices[3], metric_dim] < metric)
    CoRNN_left = torch.sum(low_rank[indices[3]:indices[4], metric_dim] < metric)
    print(f'{LSTM_right / (LSTM_right + LSTM_left) * 100:.1f}, {GRU_right / (LSTM_right + LSTM_left) * 100:.1f},'
          f'{RNN_right / (LSTM_right + LSTM_left) * 100:.1f}, {ASRNN_right / (LSTM_right + LSTM_left) * 100:.1f}, '
          f'{CoRNN_right / (LSTM_right + LSTM_left) * 100:.1f}')
    print(f'{LSTM_left / (LSTM_right + LSTM_left) * 100:.1f}, {GRU_left / (LSTM_right + LSTM_left) * 100:.1f}, '
          f'{RNN_left / (LSTM_right + LSTM_left) * 100:.1f}, {ASRNN_left / (LSTM_right + LSTM_left) * 100:.1f}, '
          f'{CoRNN_left / (LSTM_right + LSTM_left) * 100:.1f}')
    x_data_lstm = split_lstm['val_data'].detach()
    targets_lstm = split_lstm['val_targets']
    x_data_gru = split_gru['val_data'].detach()
    targets_gru = split_gru['val_targets']
    x_data_rnn = split_rnn['val_data'].detach()
    targets_rnn = split_rnn['val_targets']
    x_data_asrnn = split_asrnn['val_data'].detach()
    targets_asrnn = split_asrnn['val_targets']
    x_data_cornn = split_cornn['val_data'].detach()
    targets_cornn = split_cornn['val_targets']
    x_data = torch.cat((x_data_lstm, x_data_gru, x_data_rnn, x_data_asrnn, x_data_cornn), dim=0)
    targets = torch.cat((targets_lstm, targets_gru, targets_rnn, targets_asrnn, targets_cornn))
    # thresh = 0.005
    thresh = torch.median(targets)

    target_mask = targets < thresh
    print(f'thresh: {thresh}')
    print(f"Val Number of samples smaller than threshold: {torch.sum(target_mask)}")
    print(f"Val Number of samples larger than threshold: {torch.sum(~target_mask)}")

    outputs = model(x_data)
    latent = outputs[1].detach()
    if os.path.exists(f'{dir}/PCA_dim{dim}.p'):
        trained_project = torch.load(f'{dir}/PCA_dim{dim}.p')
        V = trained_project['V']
        low_rank = torch.matmul(latent, V[:, :dim])
    else:
        U, S, V = torch.pca_lowrank(latent)
        low_rank = torch.matmul(latent, V[:, :dim])
        trained_project = {'V': V, 'low_rank': low_rank, 'latent': latent}
        torch.save(trained_project, f'{dir}/PCA_dim{dim}.p')

    fig = plt.figure(figsize=(4, 4))
    pc1_thresholds = torch.arange(-2, 0, )
    # ks = torch.arange(0.5, 2.0, 0.1)
    f1 = torch.zeros(len(pc1_thresholds))
    # f1 = torch.zeros(len(pc1_thresholds), len(ks))
    for i, pc1_threshold in enumerate(pc1_thresholds):
        f1[i] = f1_score(low_rank, target_mask, pc1_threshold, inverse=False, verbose=True)
        # for j, k in enumerate(ks):
        #     f1[i][j] = f1_score_2d(low_rank, target_mask, pc1_threshold, k=k, inverse=False, verbose=True)
    print(f'f1 score with threshold ({pc1_thresholds}) is: {f1}')
    idx = [(f1 == torch.max(f1)).nonzero()]
    optimal_threshold = pc1_thresholds[idx[0]]
    # optimal_k = ks[idx[1]]
    print(f'optimal_threshold: {optimal_threshold}, optimal f1: {f1[idx[0]]}')
    plt.scatter(low_rank[target_mask, 0], low_rank[target_mask, 1], s=20, c='lime', alpha=0.5)
    plt.scatter(low_rank[~target_mask, 0], low_rank[~target_mask, 1], s=20, c='red', alpha=0.5)
    # plt.plot(torch.arange(-25, 10, 0.1), torch.arange(-25, 10, 0.1) * optimal_k + optimal_threshold)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('val')
    # plt.xlim([-25, 10])
    # plt.ylim([-10, 30])
    plt.show()

    # indices = [0, 1 * no_evals, 2 * no_evals, 3 * no_evals, 4 * no_evals]
    # print(f"indices: {indices}")
    # i_list = torch.arange(4 * no_evals)
    # splits = [(i_list > torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1])
    #           for i in range(len(indices) - 1)]
    epochs = [0, 1, 2, 3, 4, 5]
    model_types = ['lstm', 'gru', 'rnn', 'asrnn']
    # splits = {'lstm': split_lstm, 'gru': split_gru, 'rnn': split_rnn, 'asrnn': split_asrnn}
    x_data_processed, targets_processed, indices = data_all(model_types, no_evals, epochs=epochs)

    targets_processed_mask = targets_processed < thresh
    indices = [0, 1 * no_evals * len(model_types), 2 * no_evals * len(model_types), 3 * no_evals * len(model_types),
               4 * no_evals * len(model_types), 5 * no_evals * len(model_types), 6 * no_evals * len(model_types)]
    i_list = torch.arange(len(epochs) * no_evals * len(model_types))
    splits = [(i_list >= torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1])
              for i in range(len(indices) - 1)]

    latent = model(x_data_processed)[1].detach()
    # U,S,V = torch.pca_lowrank(latent)
    low_rank = torch.matmul(latent, V[:, :dim])


def data_process(model_type, no_evals=200, epochs=[], task='SMNIST'):
    x_data = torch.load(f'{task}/Processed/{model_type}_allLEs.p')
    n_dim = x_data.shape[1]
    targets = torch.load(f'{task}/Processed/{model_type}_allValLoss.p')
    indices = [0]
    x_data_processed = torch.zeros((len(epochs), no_evals, n_dim))
    targets_processed = torch.zeros((len(epochs), no_evals))
    for i, epoch in enumerate(epochs):
        indices.append((epoch + 1) * no_evals)
        x_data_processed[i] = x_data[indices[i]:indices[i + 1]]
        targets_processed[i] = targets[indices[i]:indices[i + 1]]
    return indices, x_data_processed, targets_processed


def data_all(model_types, no_evals, epochs=[], task='SMNIST'):
    n_hid = 1024
    x_data_temp = torch.zeros((len(model_types), len(epochs), no_evals, n_hid))
    targets_temp = torch.zeros((len(model_types), len(epochs), no_evals))
    for i, model_type in enumerate(model_types):
        indices, x_data_temp[i], targets_temp[i] = data_process(model_type, no_evals=200, epochs=epochs, task=task)
    x_data_processed = x_data_temp.transpose(0, 1).reshape(len(epochs), len(model_types) * no_evals, -1).reshape(-1,
                                                                                                                 n_hid)
    targets_processed = targets_temp.transpose(0, 1).reshape(len(epochs), len(model_types) * no_evals).reshape(-1, )
    return x_data_processed, targets_processed, indices


def f1_score(low_rank, target_mask, threshold, dim=0, inverse=False, verbose=False):
    if inverse:
        fn = torch.sum(low_rank[target_mask, dim] < threshold)
        tp = torch.sum(low_rank[target_mask, dim] > threshold)
        tn = torch.sum(low_rank[~target_mask, dim] < threshold)
        fp = torch.sum(low_rank[~target_mask, dim] > threshold)
    else:
        fn = torch.sum(low_rank[target_mask, dim] > threshold)
        tp = torch.sum(low_rank[target_mask, dim] < threshold)
        tn = torch.sum(low_rank[~target_mask, dim] > threshold)
        fp = torch.sum(low_rank[~target_mask, dim] < threshold)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if verbose:
        print(
            f'threshold: {threshold}. tp:{tp}, tn: {tn}, fp: {fp}, fn: {fn}. precision: {precision}. recall: {recall}')
    return (2 * precision * recall) / (precision + recall)


def f1_score_2d(low_rank, target_mask, threshold, k=1, inverse=False, verbose=False):
    if inverse:
        print("inverse")
        tp = torch.sum((k * low_rank[target_mask, 0] - low_rank[target_mask, 1]) > -threshold)
        fp = torch.sum((k * low_rank[target_mask, 0] - low_rank[target_mask, 1]) < -threshold)
        tn = torch.sum((k * low_rank[~target_mask, 0] - low_rank[~target_mask, 1]) < -threshold)
        fn = torch.sum((k * low_rank[~target_mask, 0] - low_rank[~target_mask, 1]) > -threshold)
    else:
        tp = torch.sum((k * low_rank[target_mask, 0] - low_rank[target_mask, 1]) < -threshold)
        fp = torch.sum((k * low_rank[target_mask, 0] - low_rank[target_mask, 1]) > -threshold)
        tn = torch.sum((k * low_rank[~target_mask, 0] - low_rank[~target_mask, 1]) > -threshold)
        fn = torch.sum((k * low_rank[~target_mask, 0] - low_rank[~target_mask, 1]) < -threshold)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if verbose:
        print(
            f'threshold: {threshold}, k: {k}. tp:{tp}, tn: {tn}, fp: {fp}, fn: {fn}. precision: {precision}. recall: {recall}')
    return (2 * precision * recall) / (precision + recall)


def pca_and_hist(target_mask, splits, idx, low_rank, xy_dim=[0, 1], threshold=None, k=None, inverse=False,
                 verbose=False):
    fig = plt.figure(figsize=(4, 4))
    plt.subplot(211)
    x_dim, y_dim = xy_dim[0], xy_dim[1]
    y = low_rank[splits[idx]]

    plt.scatter(y[:, x_dim][target_mask[splits[idx]]], y[:, y_dim][target_mask[splits[idx]]], s=20, c='lime', alpha=0.5)
    plt.scatter(y[:, x_dim][~target_mask[splits[idx]]], y[:, y_dim][~target_mask[splits[idx]]], s=20, c='r', alpha=0.5)
    # plt.plot(torch.arange(-25, 10, 0.1), torch.arange(-25, 10, 0.1) * k + threshold)
    if threshold is not None:
        # f1 = f1_score_2d(y, target_mask[splits[idx]], threshold, k=k, inverse=False, verbose=True)
        f1 = f1_score(y, target_mask[splits[idx]], threshold, dim=0, inverse=inverse, verbose=verbose)
    print(f'f1 score with threshold ({threshold}) is: {f1}')

    plt.title(f'epoch {idx}')
    plt.xlim([-25, 10])
    plt.ylim([-10, 30])
    plt.axis('off')
    bins = torch.arange(-25, 10, 0.5).tolist()
    plt.subplot(212)
    plt.hist(y[:, x_dim], bins=bins, color='lime')
    plt.hist(y[:, x_dim][~target_mask[splits[idx]]], bins=bins, color='r')
    plt.ylim([0, 60])
    plt.show()

    torch.save(low_rank, f'{task_type}/Processed/{model_type}_PCA_dim{dim}.p')
    # Performance PCA Plot
    fig = plt.figure()

    ax = fig.add_subplot(111)
    for idx, size in enumerate(sizes):
        y = low_rank[splits[idx]]
        print(y[:, 0].shape)
        im = ax.scatter(y[:, 0][target_mask[splits[idx]]], y[:, 1][target_mask[splits[idx]]], s=6, c='g')
        im = ax.scatter(y[:, 0][~target_mask[splits[idx]]], y[:, 1][~target_mask[splits[idx]]], s=6, c='r')

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    if dim == 3:
        ax.set_zlabel('PCA 3')
    plt.savefig(f'{task_type}/Figures/{model_type}_AEPredNet_pca_perf_dim{dim}.png', dpi=200)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    if dim == 3:
        ax.set_zlabel('PCA 3')
    plt.savefig(f'{task_type}/Figures/AEPredNet_pca_size_dim{dim}.png', dpi=200)


def pca_size(latent_size, size=512, dim=2, model_type='lstm'):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = torch.load(f'{task_type}/AE_Models/{model_type}/Latent_{latent_size}/ae_prednet_4000.ckpt').cpu()
    model.load_state_dict(model.best_state)
    x_data = torch.load(f'{task_type}/Processed/lstm_allLEs.p')
    targets = torch.load(f'{task_type}/Processed/lstm_allValLoss.p')
    split = torch.load(f'{task_type}/Processed/data_split_vfrac0.2.p')
    indices = [0, 300, 600, 900, 1200]
    sizes = [size]
    size_list = torch.Tensor([64, 128, 256, 512])
    i_list = torch.arange(1200)
    splits = [(i_list > torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1]) for
              i in range(len(indices) - 1)]
    s_idx = splits[torch.where(size_list == size)[0]]
    vals = targets[s_idx]
    _, latent, prediction = model(x_data)
    print(f'Pred Model: {model.prediction}')
    latent = latent.detach()
    U, S, V = torch.pca_lowrank(latent)
    low_rank = torch.load(f'{task_type}/Processed/{model_type}_PCA_dim{dim}.p')

    # Performance PCA Plot
    fig = plt.figure()

    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    y = low_rank[s_idx]
    if dim == 3:
        im = ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=6, c=vals, norm=colors.LogNorm(vmin=vals.min(), vmax=vals.max()),
                        cmap=plt.get_cmap('gr'))
    else:
        im = ax.scatter(y[:, 0], y[:, 1], s=6, c=vals, norm=colors.LogNorm(vmin=vals.min(), vmax=vals.max()),
                        cmap=plt.get_cmap('gr'))

    plt.colorbar(im, label='Val Loss')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    if dim == 3:
        ax.set_zlabel('PCA 3')
        ax.set_zlim([1.6, 3.0])
    ax.set_xlim([-1, 4])
    ax.set_ylim([1.5, 4.5])
    ax.set_title(f'PCA for size {size}')
    plt.savefig(f'{task_type}/Figures/Latent/AEPredNet_pcaPerf_dim{dim}_size{size}.png', bbox_inches='tight', dpi=200)


def size_dist(model_type='lstm', dir='lstm/', no_evals=300, v_frac=0.2, thresh=1.75):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    params = torch.load(f'{task_type}/Processed/{model_type}_allParams.p').flatten()
    targets = torch.load(f'{task_type}/Processed/{model_type}_allValLoss.p')
    target_mask = targets < thresh
    split = torch.load(f'{task_type}/Processed/{model_type}_data_split_vfrac{v_frac}.p')
    target_mask = targets < thresh
    f = plt.figure()
    indices = [0, 1 * no_evals, 2 * no_evals, 3 * no_evals, 4 * no_evals]
    sizes = [64, 128, 256, 512]
    i_list = torch.arange(len(sizes) * no_evals)
    splits = [(i_list >= torch.ones_like(i_list) * indices[i]) * (i_list < torch.ones_like(i_list) * indices[i + 1]) for
              i in range(len(indices) - 1)]
    val_mask = torch.zeros(1200, dtype=torch.bool)
    val_mask.index_fill_(0, split['val_idx'].sort()[0], True)
    f = plt.figure()
    good_counts = []
    bad_counts = []
    plt.figure(figsize=(4, 2))
    for i, size in enumerate(sizes):
        s_idx = splits[torch.where(torch.Tensor(sizes) == size)[0]]
        good_counts.append(torch.sum(s_idx * val_mask * target_mask).item())
        bad_counts.append(torch.sum(s_idx * val_mask * ~target_mask).item())
    xs = [1, 2, 3, 4]
    plt.bar(xs, good_counts, color='lime', width=0.9)
    plt.bar(xs, bad_counts, bottom=good_counts, color='r', width=0.9)
    # plt.xlabel('Network Size')
    plt.xticks(xs, sizes)
    plt.savefig(f'{task_type}/Figures/{model_type}_size_dist.png', bbox_inches='tight', dpi=200)


def get_cmap(data, n_colors=5):
    base = np.min(data)
    dmax = np.max(data)
    gradient = np.logspace(base=base, start=np.log(0.5) / np.log(base), stop=np.log(dmax) / np.log(base), num=n_colors)
    cmap = plt.cm.get_cmap('gr')
    return cmap


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
