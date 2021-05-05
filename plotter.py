import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from config import *
from training import *
from torch.nn import functional as F
import numpy as np



def logit_plot(input_sample, losses = [], indices = [], trials_dir = 'lstm/', model_type = 'lstm', hidden_size = 512, batch_size = 32, dir = 'lstm/'):
	params = torch.load(f'Processed/{dir}{model_type}_allParams.p').flatten()
	targets = torch.load(f'Processed/{model_type}/{model_type}_allValLoss.p')
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	max_epoch = 15
	dcon = TestDataConfig('', test_file = 'book_data.p', train_frac = 0.8, val_frac = 0.2, test_frac = 0.0)
	tcon = TrainConfig('Models', batch_size, max_epoch, 'adam', learning_rate, {}, start_epoch = 0)
	mcon = ModelConfig(model_type, 1, hidden_size, dcon.input_size, output_size = dcon.input_size, dropout=dropout, 
						init_type = 'uni',  device = device, bias = False, id_init_param = 'b', encoding = 'one_hot')
	fcon = FullConfig(dcon, tcon, mcon)
	# print(mcon.encoder)
	keep_chars = 10
	full_outputs = np.zeros((6, keep_chars))
	f = plt.figure(figsize = (4, 3))
	ax = f.add_axes((0.02, 0.1, 0.73, 0.8))
	for num, i in enumerate(indices):
		p = params[i]
		# print(params[i])
		fcon.model.init_params = {'a': -p, 'b':p}
		mcon = ModelConfig(model_type, 1, hidden_size, dcon.input_size, output_size = dcon.input_size, dropout=dropout, 
						init_type = 'uni', init_params = {'b': params[i]},  device = device, bias = False, id_init_param = 'b', encoding = 'one_hot')
		ckpt = load_checkpoint(fcon, max_epoch)
		model = ckpt[0].to(fcon.device)
		# print(input_sample.unsqueeze(0).shape)
		input = input_sample.unsqueeze(0)
		model_out = model(input[:, :-1])[0][:, -1].squeeze()
		output = F.softmax(model_out, dim = 0).detach()
		if num == 0:
			max_vals, max_inds = torch.sort(output, descending = True)
			new_max_inds = max_inds[:keep_chars][torch.randperm(keep_chars)]
		# print(max_inds)
		char_labels = np.array(list(map.values()))
		# print(np.array(char_labels))
		# print(char_labels[max_inds.numpy()])
		c = output[new_max_inds][:keep_chars].unsqueeze(0).numpy()
		# print(c.shape)
		full_outputs[2*num] = c
	
	norm = colors.LogNorm(vmin=2e-3, vmax=1)
	cmap = plt.get_cmap('gray_r')
	pc = ax.pcolor(full_outputs, cmap = cmap,  norm=norm)
	ax.grid(False)
	ax.set_xticks(np.array(range(keep_chars))+0.5)
	ax.set_xticklabels(labels = char_labels[new_max_inds.numpy()][:keep_chars], Fontsize = 16)
	ax.set_yticks([])
	# ax.set_yticklabels([f'Error=\n{loss:0.3f}  ' for loss in losses])
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	for i in range(3):
		loss = losses[i]
		max_ind = torch.where(new_max_inds == max_inds[0])[0]
		# print(max_ind)
		ax.text(max_ind.item() + 0.5, 2*i + 0.5, f'{full_outputs[2*i, max_ind.item()]:.2f}', color = 'w', 
							horizontalalignment = 'center', rotation = 'vertical', verticalalignment = 'center')
		ax.text(keep_chars/2, 2*i + 1.2, f'Error={loss:0.2f}', horizontalalignment = 'center')
	cax = f.add_axes([0.77, 0.1, 0.05, 0.70])
	f.colorbar(pc, cax = cax, label = 'Probability')
	plt.savefig('CharRNNPredictions2', dpi = 200, hbbox_inches = 'tight')
	plt.show()
	
def LE_plotter(model_type = 'lstm', dir = 'lstm/', LE_idx = []):
	x_data = torch.load(f'Processed/{dir}{model_type}_allLEs.p')
	LE_plotted = x_data[LE_idx]
	plt.figure(figsize = (4,3))
	plt.gray()
	plt.style.use('grayscale')
	for LEs in LE_plotted:
		# print(LEs.shape)
		plt.scatter(list(range(1024)), LEs, s = 8, alpha = 0.2)
	plt.axis('off')
	plt.savefig('SampleLE_spectra', dpi = 200)
	plt.show()
		
		
		
		
if __name__ == "__main__":
	model_type = 'lstm'
	hidden_size = 512
	batch_size = 32
	indices = [948, 986, 903]
	losses = torch.load('Processed/lstm/lstm_allValLoss.p')
	# indices = [948, 903]
	# logit_plot(indices = indices, batch_size = batch_size)
	trials_dir = f'{model_type}'
	learning_rate = 0.002
	dropout = 0.1
	dcon = TestDataConfig('', test_file = 'book_data.p', train_frac = 0.8, val_frac = 0.2, test_frac = 0.0) 
	# dcon = torch.load('dcon.p')
	v = dcon.datasets['val_set']
	# print(dcon.datasets['int_to_char'])
	# print(dcon.datasets['char_to_int'])
	map =  dcon.datasets['int_to_char']
	# for i in range(30):
	sample = [map[c.item()] for c in v[0][0, 29, 10:55]]
	# print(f'Index {i}')
	print(''.join(sample))
	test_sample = v[0][0, 29, 10:55]
	logit_plot(test_sample, losses = losses[indices], indices = indices)
	# LE_plotter(LE_idx = [1,301, 451, 601, 901])
