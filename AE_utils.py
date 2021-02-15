import torch
from AEPredNet import AEPredNet
import numpy as np
from math import floor
import os

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
		shift = torch.cat((le[:, 1:], torch.zeros((m, 1), dtype = le.dtype)), dim = 1).to(device)
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
	# params = torch.zeros((0, num_params))
	for start_size in start_sizes:
		le_data = torch.cat((le_data, interpolate_LEs(start_size, target_size, prefix, suffix)))
		val_data = torch.cat((val_data, torch.load(f'{prefix}_{start_size}{suffix}_valLoss.p')))
		# params = torch.cat((params, torch.load(f'{prefix}_{start_size}{suffix}_params.p')))
	torch.save(le_data, f'Processed/{prefix}_{suffix}allLEs.p')
	torch.save(val_data, f'Processed/{prefix}_{suffix}allValLoss.p')
		
def mini_batch_ae(features, batch_size):
	for start in range(0,len(features),batch_size):
		end = min(start+batch_size,len(features))
		yield features[start:end]
		
def train_val_split(data, targets, val_split = 0.2, save = True):
	samples = data.shape[0]
<<<<<<< HEAD
	train_samples = torch.arange(floor(samples * (1- val_split)))
=======
	train_samples = torch.arange(floor(samples * (1 - val_split)))
>>>>>>> 40aeaf8ece1869e6a32deb9dc2e10862449e34b8
	val_samples = torch.arange(torch.max(train_samples)+1, samples)
	shuffle_idx = torch.randperm(samples)
	train_idx, val_idx = shuffle_idx[train_samples], shuffle_idx[val_samples]
	# shuff_data, shuff_targ = data[shuffle_idx], targets[shuffle_idx]
	train_data,val_data = data[train_idx], data[val_idx]
	train_targets, val_targets = targets[train_idx], targets[val_idx]
	
	split_dict = {'train_data': train_data, 'train_targets':train_targets, 'val_data': val_data, 
					'val_targets': val_targets, 'train_idx': train_idx, 'val_idx': val_idx }
	torch.save(split_dict, f'data_split_vfrac{val_split}.p')
	return split_dict
	
if __name__ == '__main__':
	x_data = torch.load('Processed/lstm_allLEs.p')
	targets = torch.load('Processed/lstm_allValLoss.p')
	if os.path.exists('data_split_vfrac0.2.p'):
		split = torch.load('data_split_vfrac0.2.p')
	else:
		split = train_val_split(x_data, targets, 0.2)
		print(f'New dataset created')
		print(split['train_data'].shape[0])
	
	