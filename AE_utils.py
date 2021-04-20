import torch
from AEPredNet import AEPredNet
import numpy as np
from math import floor
import os
<<<<<<< HEAD
from CharRNN_trials import *
=======
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
>>>>>>> 6a3b9cc7df7d2820c3d4d466dfa4a71729250b18

def interpolate_LEs(start_size, target_size, prefix = 'lstm', suffix = ''):
	factor_increase = target_size//start_size
	if not np.modf(np.log2(factor_increase))[0] == 0:
		raise ValueError(f'Target size ({target_size}) must be a power of 2 times the start size ({start_size}). The factor is {factor_increase:.2f}.')
	print(f'Interpolation Factor: {int(factor_increase)}')
	file_name = f'{prefix}_{start_size}{suffix}_LEs.p'
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
	torch.save(le, f'{prefix}_{start_size}{suffix}_LEs_{target_size}.p')
	return le
	
	
def combine_sizes(start_sizes, target_size, prefix = 'lstm', suffix = '', num_params = 1):
	le_data = torch.zeros((0, target_size))
	val_data = torch.zeros((0, ))
	params = torch.zeros((0, num_params))
	for start_size in start_sizes:
		le_data = torch.cat((le_data, interpolate_LEs(start_size, target_size, prefix, suffix).cpu()))
		val_data = torch.cat((val_data, torch.load(f'{prefix}_{start_size}{suffix}_valLoss.p').cpu()))
		params = torch.cat((params, torch.load(f'{prefix}_{start_size}{suffix}_params.p').unsqueeze(dim=0)), dim=0)
	torch.save(le_data, f'Processed/{prefix}_{suffix}allLEs.p')
	torch.save(val_data, f'Processed/{prefix}_{suffix}allValLoss.p')
	torch.save(params, f'Processed/{prefix}_{suffix}allParams.p')
		
def mini_batch_ae(features, batch_size):
	for start in range(0,len(features),batch_size):
		end = min(start+batch_size,len(features))
		yield features[start:end]
		
def train_val_split(data, targets, val_split = 0.2, save = True, prefix = 'lstm'):
	samples = data.shape[0]
<<<<<<< HEAD
	train_samples = torch.arange(floor(samples * (1- val_split)))
=======
	train_samples = torch.arange(floor(samples * (1 - val_split)))
>>>>>>> 6a3b9cc7df7d2820c3d4d466dfa4a71729250b18
	val_samples = torch.arange(torch.max(train_samples)+1, samples)
	shuffle_idx = torch.randperm(samples)
	train_idx, val_idx = shuffle_idx[train_samples], shuffle_idx[val_samples]
	# shuff_data, shuff_targ = data[shuffle_idx], targets[shuffle_idx]
	train_data, val_data = data[train_idx], data[val_idx]
	train_targets, val_targets = targets[train_idx], targets[val_idx]
	
	split_dict = {'train_data': train_data, 'train_targets':train_targets, 'val_data': val_data, 
					'val_targets': val_targets, 'train_idx': train_idx, 'val_idx': val_idx }
<<<<<<< HEAD
	torch.save(split_dict, f'Processed/{prefix}_data_split_vfrac{val_split}.p')
	return split_dict

def merge_data(dir = ''):
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
	
if __name__ == '__main__':
	model_type = 'lstm'
	dir = 'lstm'
	no_evals = 300
	sizes = [64, 128, 256, 512]
	for size  in sizes:
		extract_trials(size, dir)
	combine_sizes([64, 128, 256, 512], 1024, prefix = f'{dir}/{model_type}', num_params = no_evals)
	data = torch.load(f'Processed/{dir}/{model_type}_allLEs.p')
	targets = torch.load(f'Processed/{dir}/{model_type}_allValLoss.p')
	
	# data, targets = merge_data()
	# model_type = 'merged'
	if os.path.exists(f'Processed/{model_type}_data_split_vfrac0.2.p'):
		split = torch.load(f'Processed/{model_type}_data_split_vfrac0.2.p')
	else:
		split = train_val_split(data, targets, val_split = 0.2, prefix = model_type)
		print(f'New dataset created')
		print(split['train_data'].shape[0])
	
=======
	return split_dict

def main():
	N = 512
	g = 1.5
	inputs_epoch = 3
	target_epoch = 14
	data_path = "training_data/g_{}/4sine_epoch_{}_N_{}".format(g, inputs_epoch, N)
	data = pickle.load(open(data_path, 'rb'))
	inputs, targets = data['inputs'], data['targets']
	val_split = 0.1
	split = train_val_split(data=inputs, targets=targets, val_split=val_split)
	x_train, y_train, x_val, y_val = split['train_data'], split['train_targets'],\
									 split['val_data'], split['val_targets']

	plt.figure()
	plt.scatter(torch.ones_like(y_train), y_train, s=2)
	plt.scatter(torch.ones_like(y_val) * 1.1, y_val.detach(), s=2)
	plt.axis([0.95, 1.15, -.1, 1.])
	plt.legend(["Train", "Validation"])
	plt.show()

	pca = PCA(2)
	plt.figure()
	x_pca = pca.fit_transform(x_train)
	x_pca = pd.DataFrame(x_pca)
	x_pca.columns = ['PC1', 'PC2']
	plt.scatter(x_pca.values[:,0], x_pca.values[:,1], c=y_train)
	plt.title('Scatter plot')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

if __name__ == '__main__':
	main()
	# x_data = torch.load('Processed/lstm_allLEs.p')
	# targets = torch.load('Processed/lstm_allValLoss.p')
	# if os.path.exists('data_split_vfrac0.2.p'):
	# 	split = torch.load('data_split_vfrac0.2.p')
	# else:
	# 	split = train_val_split(x_data, targets, 0.2)
	# 	print(f'New dataset created')
	# 	print(split['train_data'].shape[0])
	#
>>>>>>> 6a3b9cc7df7d2820c3d4d466dfa4a71729250b18
	