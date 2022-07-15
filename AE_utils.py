import torch
from AEPredNet import AEPredNet
import numpy as np
from math import floor
import os
from generate_trials import *
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import random

def interpolate_LEs(start_size, target_size, prefix = 'lstm', suffix = '', epoch=10, model_type='lstm'):
    factor_increase = target_size//start_size
    if not np.modf(np.log2(factor_increase))[0] == 0:
        raise ValueError(f'Target size ({target_size}) must be a power of 2 times the start size ({start_size}). The factor is {factor_increase:.2f}.')
    print(f'Interpolation Factor: {int(factor_increase)}')
    file_name = f'LE_stats/{model_type}_{start_size}_allLEs_e{epoch}.p'
    #
    # file_name = f'{prefix}_{start_size}{suffix}_LEs.p'
    print(f'loading from: {file_name}')
    le = torch.load(file_name)

    device = le.device
    m = le.shape[0]
    for factor in 2**((np.arange(int(np.log2(factor_increase))))+1):
        shift = torch.cat((le[:, 1:], torch.zeros((m, 1), dtype = le.dtype).to(device)), dim = 1).to(device)
        diffs = (le - shift)/2; diffs[:, -1] = diffs[:, -2]
        test = torch.zeros(m, 0).to(device)
        for col, diff in zip(le.T, diffs.T):
            test = torch.cat((test, col.unsqueeze(1)), dim = 1)
            test = torch.cat((test, (col-diff).unsqueeze(1)), dim = 1)
            le = test
    # print(f'saving to: {prefix}_{start_size}{suffix}_LEs_{target_size}.p')
    # torch.save(le, f'{prefix}_{start_size}{suffix}_LEs_{target_size}.p')
    return le


def combine_sizes(start_sizes, target_size, prefix = 'lstm', suffix = '', num_params = 1,
                  dir='lstm', model_type= 'lstm'):
    if not os.path.isdir('Processed/'):
        os.mkdir('Processed/')
    le_data = torch.zeros((0, target_size))
    val_data = torch.zeros((0, ))
    params = torch.zeros((0, ))
    epochs = range(6)
    # epochs = [10]
    for start_size in start_sizes:
        for epoch in epochs:
            le_temp = interpolate_LEs(start_size, target_size, prefix, suffix, epoch, model_type).cpu()
            # remove nan

            nan_temp = torch.isnan(le_temp)
            # not_nan_indices = [not index for index in nan_indices]
            # if nan_indices:
            nan_indices = np.where(torch.sum(nan_temp, dim=1))
            print(f"nan_indices: {nan_indices}")
            # else:
            #     nan_indices = []
            not_nan_indices = np.where(torch.sum(~nan_temp, dim=1) == 1024)
            # print(f'{epoch}: {nan_indices}')
            # print(f'{epoch}: {not_nan_indices}')
            # not_nan_indices = np.concatenate((not_nan_indices, random.choices(not_nan_indices[0], k=len(nan_indices[0]))), dim=0)
            augmented_indices = np.array(random.choices(not_nan_indices[0], k=len(nan_indices[0])))
            new_indices = np.concatenate((not_nan_indices[0], augmented_indices))

            le_temp = le_temp[new_indices]
            val_data_temp = torch.load(f'{prefix}_{start_size}{suffix}_valLoss.p').cpu()
            params_temp = torch.load(f'{prefix}_{start_size}{suffix}_params.p').unsqueeze(dim=0)
            # plt.figure()
            # plt.scatter(range(200), val_data_temp)
            # plt.show()
            #
            # plt.figure()
            # plt.scatter(range(200), params_temp[0])
            # plt.show()
            val_data_temp = val_data_temp[new_indices]
            params_temp = params_temp[:, new_indices]
            le_data = torch.cat((le_data, le_temp), dim=0)
            val_data = torch.cat((val_data, val_data_temp), dim=0)
            params = torch.cat((params, params_temp), dim = 1)

            # plt.figure()
            # plt.scatter(range(200), val_data_temp)
            # plt.show()
            #
            # plt.figure()
            # plt.scatter(range(200), params_temp[0])
            # plt.show()
            # print("ss")
    # b = torch.unsqueeze(torch.arange(0, 1200), 1)
    # x = torch.ones_like(le_data)
    # x = b * x
    # plt.figure()
    # plt.scatter(x[:10], le_data[:10])
    # plt.show()
    print(f"number of nan: {torch.sum(torch.isnan(le_data))}")
    if not os.path.exists(f'Processed/{dir}'):
        os.makedirs(f'Processed/{dir}')
        # os.mkdir(f'Processed/{dir}')
    torch.save(le_data, f'Processed/{prefix}_{suffix}allLEs.p')
    torch.save(val_data, f'Processed/{prefix}_{suffix}allValLoss.p')
    torch.save(params, f'Processed/{prefix}_{suffix}allParams.p')

def mini_batch_ae(features, batch_size):
    for start in range(0,len(features),batch_size):
        end = min(start+batch_size,len(features))
        yield features[start:end]

def train_val_split(data, targets, val_split = 0.1, test_split=0.2, dir='none',save = True, model_type = 'lstm', task_type='charRNN'):
    dir = f'Processed/{dir}'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    samples = data.shape[0]
    train_samples = torch.arange(floor(samples * (1- val_split - test_split)))
    val_samples = torch.arange(torch.max(train_samples)+1, floor(samples * (1 - test_split)))
    test_samples = torch.arange(torch.max(val_samples)+1, samples)

    shuffle_idx = torch.randperm(samples)
    train_idx, val_idx, test_idx = shuffle_idx[train_samples], shuffle_idx[val_samples], shuffle_idx[test_samples]
    train_data, val_data, test_data = data[train_idx], data[val_idx], data[test_idx]
    train_targets, val_targets, test_targets = targets[train_idx], targets[val_idx], targets[test_idx]

    split_dict = {'train_data': train_data, 'train_targets':train_targets, 'train_idx': train_idx,
                  'val_data': val_data,     'val_targets': val_targets,    'val_idx': val_idx,
                  'test_data': test_data,   'test_targets': test_targets,  'test_idx': test_idx}

    dataset_path = f'{dir}/{model_type}_data_split_vfrac{val_split}_testfrac{test_split}.p'
    torch.save(split_dict, f'{dataset_path}')
    return split_dict

def merge_data(dir = ''):
    #If using both LSTM and GRU data, you can combine them into a single dataset using this method
    lstm_data = torch.load('Processed/lstm_allLEs.p')
    lstm_targets = torch.load('Processed/lstm_allValLoss.p')
    gru_data = torch.load('Processed/gru_allLEs.p')
    gru_targets = torch.load('Processed/gru_allValLoss.p')
    merged_data = torch.cat((lstm_data, gru_data), dim = 0)
    merged_targets = torch.cat((lstm_targets, gru_targets), dim = 0)
    torch.save(merged_data, f'Processed/merged_allLEs.p')
    torch.save(merged_targets, f'Processed/merged_allValLoss.p')
    print(merged_targets.shape)
    torch.save(torch.cat((torch.zeros((lstm_data.shape[0]),), torch.ones((gru_data.shape[0],)))), 'network_labels.p')
    return merged_data, merged_targets


def main(args):
    parser = argparse.ArgumentParser(description="Train recurrent models")
    parser.add_argument("-model", "--model_type", type=str, default='lstm', required=False)
    parser.add_argument("-task", "--task_type", type=str, default='SMNIST', required=False)
    parser.add_argument("-evals", "--evals", type=int, default='20', required=False)
    args = parser.parse_args(args)
    model_type = args.model_type
    task_type = args.task_type
    no_evals = args.evals

    # testing code
    model_type = 'lstm'
    task_type  = 'SMNIST'
    no_evals   = 200

    dir = f'trials/{task_type}/{model_type}'
    # no_evals = 300
    sizes = [64]#, 128, 256, 512]
    # sizes = [512]
    print(dir)
    for size in sizes:
        extract_trials(size, dir, model_type=model_type,task_type=task_type)
    combine_sizes(sizes, 1024, prefix = f'{dir}/{model_type}', num_params = no_evals, dir=dir, model_type=model_type)
    data = torch.load(f'Processed/{dir}/{model_type}_allLEs.p')
    targets = torch.load(f'Processed/{dir}/{model_type}_allValLoss.p')

    # data, targets = merge_data()
    # model_type = 'merged'

    val_split = 0.1
    test_split = 0.2
    dataset_path = f'Processed/{dir}/{model_type}_data_split_vfrac{val_split}_testfrac{test_split}.p'
    print("")
    if os.path.exists(f'{dataset_path}'):
        split = torch.load(f'{dataset_path}')
    else:
        split = train_val_split(data, targets, val_split = val_split, test_split=test_split, dir=dir, task_type=task_type, model_type=model_type)
        print(f'New dataset created')
        print(split['train_data'].shape[0], split['val_data'].shape[0], split['test_data'].shape[0])
    return split

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
    # dataset = torch.load('Processed/trials/SMNIST/gru/gru_data_split_vfrac0.2.p')
    # print(dataset.keys())
    # main(sys.argv[1:])
# def main():
    # N = 512
    # g = 1.5
    # inputs_epoch = 3
    # target_epoch = 14
    # data_path = "training_data/g_{}/4sine_epoch_{}_N_{}".format(g, inputs_epoch, N)
    # data = pickle.load(open(data_path, 'rb'))
    # inputs, targets = data['inputs'], data['targets']
    # val_split = 0.1
    # split = train_val_split(data=inputs, targets=targets, val_split=val_split)
    # x_train, y_train, x_val, y_val = split['train_data'], split['train_targets'],\
                                     # split['val_data'], split['val_targets']

    # plt.figure()
    # plt.scatter(torch.ones_like(y_train), y_train, s=2)
    # plt.scatter(torch.ones_like(y_val) * 1.1, y_val.detach(), s=2)
    # plt.axis([0.95, 1.15, -.1, 1.])
    # plt.legend(["Train", "Validation"])
    # plt.show()

    # pca = PCA(2)
    # plt.figure()
    # x_pca = pca.fit_transform(x_train)
    # x_pca = pd.DataFrame(x_pca)
    # x_pca.columns = ['PC1', 'PC2']
    # plt.scatter(x_pca.values[:,0], x_pca.values[:,1], c=y_train)
    # plt.title('Scatter plot')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

# if __name__ == '__main__':
    # main()
    # x_data = torch.load('Processed/lstm_allLEs.p')
    # targets = torch.load('Processed/lstm_allValLoss.p')
    # if os.path.exists('data_split_vfrac0.2.p'):
    # 	split = torch.load('data_split_vfrac0.2.p')
    # else:
    # 	split = train_val_split(x_data, targets, 0.2)
    # 	print(f'New dataset created')
    # 	print(split['train_data'].shape[0])
