import torch
from AEPredNet import AEPredNet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D 

latent_size = 32
def tsne(model, X, tsne_params = {}, model_type = 'lstm'):
	encoded = model(X)[1]
	tsne_model = TSNE(**tsne_params)
	X_embedded = tsne_model.fit_transform(encoded.detach().numpy())
	print(X_embedded.shape)
	return tsne_model
	
def main(latent_size, model_type = 'lstm', dir = ''):
	if torch.cuda.is_available():
		device= torch.device('cuda')
	else: 
		device= torch.device('cpu')
	model = torch.load(f'Models/Latent_{latent_size}/ae_prednet_4000.ckpt').cpu()
	model.load_state_dict(model.best_state)
	x_data = torch.load(f'Processed/{dir}{model_type}_allLEs.p')
	split = torch.load('data_split_vfrac0.2.p')
	indices = [0, 300, 600, 900, 1200]
	sizes = [64, 128, 256, 512]
	tsne_model = tsne(model, split['train_data'], tsne_params = {'perplexity' : 10})
	splits = []
	i_list = torch.arange(1200)
	splits = [(i_list>torch.ones_like(i_list)*indices[i])*(i_list<torch.ones_like(i_list)*indices[i+1]) for i in range(len(indices)-1)]
	Y = tsne_model.fit_transform(model(x_data)[1].detach())
	plt.figure()
	for idx, size in enumerate(sizes):
		y = Y[splits[idx]]
		plt.scatter(y[:,0], y[:,1], s = 6, label = size)
	plt.legend()
	torch.save(Y, 'tsne.p')
	plt.savefig('AEPredNet_tsne_size.png', dpi = 200)

def param_plot(latent_size, model_type = 'lstm', dir = '', no_evals = 300, val_split = 0.2):
	model = torch.load(f'Models/Latent_{latent_size}/ae_prednet_4000.ckpt').cpu()
	model.load_state_dict(model.best_state)
	x_data = torch.load(f'Processed/{dir}{model_type}_allLEs.p')
	print(f'{model_type} Shape: {x_data.shape}')
	params = torch.load(f'Processed/{dir}{model_type}_allParams.p').flatten()
	split = torch.load(f'Processed/{model_type}_data_split_vfrac{val_split}.p')
	indices = [0, 1*no_evals, 2*no_evals, 3*no_evals, 4*no_evals]
	sizes = [64, 128, 256, 512]
	val_idx = split['val_idx']
	val_splits = []
	plt.figure()
	for i in range(len(sizes)):
		val_splits.append(((val_idx>torch.ones_like(val_idx)*indices[i])*(val_idx<torch.ones_like(val_idx)*indices[i+1])))
		# print(torch.arange(1200).float()>torch.ones(1200)*indices[i])
		# print(val_idx[val_splits[i]].shape)
		plt.scatter(params[val_idx[val_splits[i]]], model(x_data[val_idx[val_splits[i]]])[2].detach(), label = sizes[i], s = 14)
	plt.legend(loc = 2)
	plt.xlabel('Init Param')
	plt.ylim([1.1, 3.0])
	plt.ylabel('Validation Loss \n (Predicted)')
	plt.title('AE Predictions')
	plt.savefig(f'Figures/AEPredNet_{model_type}_paramPlot.png', bbox_inches="tight",dpi=200)
	
	plt.figure()
	targets = torch.load(f'Processed/{dir}{model_type}_allValLoss.p')
	for i in range(4): 
		plt.scatter(params[val_idx[val_splits[i]]], targets[val_idx[val_splits[i]]], label = sizes[i], s = 14)
	plt.legend(prop = {'size':12}, loc = 2)
	plt.ylabel('Val Loss\n(Actual)')
	plt.xlabel('Init Param')
	plt.ylim([1.1, 3.0])
	plt.title(f'Ground Truth')
	plt.savefig(f'Figures/Actual_{model_type}_paramPlot.png', bbox_inches="tight",dpi=200)
	
def tsne_perf(no_evals = 400, model_type = 'lstm'):
	indices = [0, 1*no_evals, 2*no_evals, 3*no_evals, 4*no_evals]
	sizes = [64, 128, 256, 512]
	targets = torch.load(f'Processed/{model_type}_allValLoss.p')
	splits = []
	i_list = torch.arange(len(sizes) * no_evals)
	splits = [(i_list>torch.ones_like(i_list)*indices[i])*(i_list<torch.ones_like(i_list)*indices[i+1]) for i in range(len(indices)-1)]
	Y = torch.load('tsne.p')
	plt.figure()
	for idx, size in enumerate(sizes):
		y = Y[splits[idx]]
		plt.scatter(y[:,0], y[:,1], s = 6, c = targets[splits[idx]], norm=colors.LogNorm(vmin=targets.min()-0.2, vmax=targets.max()), cmap = plt.get_cmap('hsv'))
	plt.colorbar(label = 'Val Loss')
	plt.xlabel('TSNE 1')
	plt.ylabel('TSNE 2')
	plt.savefig('AEPredNet_tsne_performance.png', dpi = 200)
	
def tsne_param(no_evals = 400, model_type = 'lstm'):
	indices = [0, 1*no_evals, 2*no_evals, 3*no_evals, 4*no_evals]
	sizes = [64, 128, 256, 512]
	params = torch.load(f'Processed/{model_type}_allParams.p')
	splits = []
	i_list = torch.arange(4*no_evals)
	splits = [(i_list>torch.ones_like(i_list)*indices[i])*(i_list<torch.ones_like(i_list)*indices[i+1]) for i in range(len(indices)-1)]
	Y = torch.load('tsne.p')
	plt.figure()
	for idx, size in enumerate(sizes):
		y = Y[splits[idx]]
		plt.scatter(y[:,0], y[:,1], s = 6, c = params[splits[idx]])#, norm=colors.LogNorm(vmin=params.min(), vmax=params.max()))
	plt.colorbar(label = 'Init Param')
	plt.xlabel('TSNE 1')
	plt.ylabel('TSNE 2')
	plt.savefig('AEPredNet_tsne_params.png', dpi = 200)

def pca(latent_size, dim=2, model_type = 'lstm', no_evals = 300, v_frac= 0.2, suffix = '', thresh = 1.75):
	if torch.cuda.is_available():
		device= torch.device('cuda')
	else: 
		device= torch.device('cpu')
	model = torch.load(f'Models/Latent_{latent_size}/ae_prednet_4000.ckpt').cpu()
	model.load_state_dict(model.best_state)
	x_data = torch.load(f'Processed/{model_type}/{model_type}_allLEs.p')
	targets = torch.load(f'Processed/{model_type}/{model_type}_allValLoss.p')
	target_mask = targets < thresh
	# print(f'Target shape {targets.shape}')
	split = torch.load(f'Processed/{model_type}_data_split_vfrac{v_frac}.p')
	indices = [0, 1*no_evals, 2*no_evals, 3*no_evals, 4*no_evals]
	sizes = [64, 128, 256, 512]
	i_list = torch.arange(4*no_evals)
	splits = [(i_list>torch.ones_like(i_list)*indices[i])*(i_list<torch.ones_like(i_list)*indices[i+1]) for i in range(len(indices)-1)]
	latent = model(x_data)[1].detach()
	U,S,V = torch.pca_lowrank(latent)
	low_rank = torch.matmul(latent, V[:, :dim])
	torch.save(low_rank, f'PCA_dim{dim}.p')
	#Performance PCA Plot
	fig = plt.figure()
	
	ax = fig.add_subplot(111)
	for idx, size in enumerate(sizes):
		y = low_rank[splits[idx]]
		print(y[:,0].shape)
		im = ax.scatter(y[:,0][target_mask[splits[idx]]], y[:,1][target_mask[splits[idx]]], s = 6, c = 'g')
		im = ax.scatter(y[:,0][~target_mask[splits[idx]]], y[:,1][~target_mask[splits[idx]]], s = 6, c = 'r')
		
	
	# if dim == 3:
		# ax = fig.add_subplot(111, projection='3d')
	# else:
		# ax = fig.add_subplot(111)
	# for idx, size in enumerate(sizes):
		# y = low_rank[splits[idx]]
		# if dim == 3:
			# im = ax.scatter(y[:,0], y[:,1], y[:,2], s = 6, c = targets[splits[idx]], norm=colors.LogNorm(vmin=targets.min(), vmax=targets.max()+1.2), cmap = plt.get_cmap('brg_r'))
		# else:
			# im = ax.scatter(y[:,0], y[:,1], s = 6, c = targets[splits[idx]], norm=colors.LogNorm(vmin=targets.min(), vmax=targets.max()+1.2), cmap = plt.get_cmap('brg_r'))
	# ax.add_colorbar(label = 'Val Loss')
	# plt.colorbar(im, label = 'Val Loss', )
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	if dim == 3:
		ax.set_zlabel('PCA 3')
	plt.savefig(f'Figures/Latent/{suffix}AEPredNet_pca_perf_dim{dim}.png', dpi = 200)
	#Size PCA Plot
	fig = plt.figure()
	if dim == 3:
		ax = fig.add_subplot(111, projection='3d')
	else:
		ax = fig.add_subplot(111)
	for idx, size in enumerate(sizes):
		y = low_rank[splits[idx]]
		if dim == 3:
			im = ax.scatter(y[:,0], y[:,1], y[:,2], s = 6, label = size)
		else:
			im = ax.scatter(y[:,0], y[:,1], s = 6, label = size)
	plt.legend()
	# plt.colorbar(label = 'Val Loss')
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	if dim == 3:
		ax.set_zlabel('PCA 3')
	plt.savefig(f'Figures/Latent/{suffix}AEPredNet_pca_size_dim{dim}.png', dpi = 200)
	
def pca_size(latent_size, size = 512, dim=2, model_type = 'lstm'):
	if torch.cuda.is_available():
		device= torch.device('cuda')
	else: 
		device= torch.device('cpu')
	model = torch.load(f'Models/Latent_{latent_size}/ae_prednet_4000.ckpt').cpu()
	model.load_state_dict(model.best_state)
	x_data = torch.load('Processed/lstm_allLEs.p')
	targets = torch.load('Processed/lstm_allValLoss.p')
	split = torch.load('data_split_vfrac0.2.p')
	indices = [0, 300, 600, 900, 1200]
	sizes = [size]
	size_list = torch.Tensor([64, 128,256, 512])
	i_list = torch.arange(1200)
	splits = [(i_list>torch.ones_like(i_list)*indices[i])*(i_list<torch.ones_like(i_list)*indices[i+1]) for i in range(len(indices)-1)]
	s_idx = splits[torch.where(size_list == size)[0]]
	vals = targets[s_idx]
	latent = model(x_data)[1].detach()
	U,S,V = torch.pca_lowrank(latent)
	low_rank = torch.load(f'PCA_dim{dim}.p')

	#Performance PCA Plot
	fig = plt.figure()

	if dim == 3:
		ax = fig.add_subplot(111, projection='3d')
	else:
		ax = fig.add_subplot(111)

	y = low_rank[s_idx]
	if dim == 3:
		# im = ax.scatter(y[:,0], y[:,1], y[:,2], s = 6, c = vals, norm=colors.LogNorm(vmin=vals.min(), vmax=vals.max()), cmap = plt.get_cmap('gr'))
		im = ax.scatter(y[:,0], y[:,1], y[:,2], s = 6, c = vals, norm=colors.LogNorm(vmin=vals.min(), vmax=vals.max()), cmap = plt.get_cmap('gr'))
	else:
		im = ax.scatter(y[:,0], y[:,1], s = 6, c = vals, norm=colors.LogNorm(vmin=vals.min(), vmax=vals.max()), cmap = plt.get_cmap('gr'))

	plt.colorbar(im, label = 'Val Loss')
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	if dim == 3:
		ax.set_zlabel('PCA 3')
		ax.set_zlim([1.6, 3.0])
	ax.set_xlim([-1, 4])
	ax.set_ylim([1.5, 4.5])
	ax.set_title(f'PCA for size {size}')
	plt.savefig(f'Figures/Latent/AEPredNet_pcaPerf_dim{dim}_size{size}.png', bbox_inches = 'tight', dpi = 200)

def param_dist(model_type = 'lstm', dir = 'lstm/', no_evals = 300, v_frac= 0.2, suffix = '', thresh = 1.75):
	if torch.cuda.is_available():
		device= torch.device('cuda')
	else: 
		device= torch.device('cpu')
	params = torch.load(f'Processed/{dir}{model_type}_allParams.p').flatten()
	targets = torch.load(f'Processed/{model_type}/{model_type}_allValLoss.p')
	split = torch.load(f'Processed/{model_type}_data_split_vfrac{v_frac}.p')
	target_mask = targets < thresh
	f = plt.figure()
	bins = [0.04, 0.10, 0.16, 0.22, 0.28, 0.34, 0.40]
	i = 0
	# print(torch.sum((params<i+1)*(params >i)*target_mask).item())
	good_counts = [torch.sum(((params<bins[i+1])*(params >bins[i])*target_mask)[split['val_idx']]).item() for i in range(len(bins)-1)]
	bad_counts = [torch.sum(((params<bins[i+1])*(params >bins[i])*~target_mask)[split['val_idx']]).item() for i in range(len(bins)-1)]
	bar_width = 0.05
	plt.bar(torch.Tensor(bins[1:])-0.03, good_counts, width = bar_width, color = 'g')
	plt.bar(torch.Tensor(bins[1:])-0.03, bad_counts, bottom = good_counts, width = bar_width, color = 'r')
	plt.xticks(torch.tensor(bins[1:])-0.03, labels = [f'{bins[i]:.2f} -'.lstrip('0') + f'{bins[i+1]:.2f}'.lstrip('0') for i in range(len(bins)-1)], rotation = 0)
	plt.xlabel('Initialization Parameter')
	plt.savefig(f'Figures/param_dist.png', bbox_inches = 'tight', dpi = 200)


def size_dist(model_type = 'lstm', dir = 'lstm/', no_evals = 300, v_frac= 0.2, suffix = '', thresh = 1.75):
	if torch.cuda.is_available():
		device= torch.device('cuda')
	else: 
		device= torch.device('cpu')
	params = torch.load(f'Processed/{dir}{model_type}_allParams.p').flatten()
	targets = torch.load(f'Processed/{model_type}/{model_type}_allValLoss.p')
	target_mask = targets < thresh
	split = torch.load(f'Processed/{model_type}_data_split_vfrac{v_frac}.p')
	target_mask = targets < thresh
	f = plt.figure()
	indices = [0, 1*no_evals, 2*no_evals, 3*no_evals, 4*no_evals]
	sizes = [64, 128, 256, 512]
	i_list = torch.arange(len(sizes)*no_evals)
	splits = [(i_list>=torch.ones_like(i_list)*indices[i])*(i_list<torch.ones_like(i_list)*indices[i+1]) for i in range(len(indices)-1)]
	val_mask = torch.zeros(1200, dtype = torch.bool)
	val_mask.index_fill_(0, split['val_idx'].sort()[0], True)
	f = plt.figure()
	good_counts = []
	bad_counts = []
	for i, size in	enumerate(sizes):
		s_idx = splits[torch.where(torch.Tensor(sizes) == size)[0]]
		good_counts.append(torch.sum(s_idx*val_mask*target_mask).item())
		bad_counts.append(torch.sum(s_idx*val_mask*~target_mask).item())
	xs = [1, 2, 3, 4]
	plt.bar(xs, good_counts, color = 'g', width = 0.9)
	plt.bar(xs, bad_counts, bottom = good_counts, color = 'r', width = 0.9)
	plt.xlabel('Network Size')
	plt.xticks(xs, sizes)
	plt.savefig(f'Figures/size_dist.png', bbox_inches = 'tight', dpi = 200)


def get_cmap(data, n_colors = 5):
	base = np.min(data)
	dmax = np.max(data)
	gradient = np.logspace(base = base, start = np.log(0.5)/np.log(base), stop = np.log(dmax)/np.log(base), num = n_colors)
	cmap = plt.cm.get_cmap('gr')
	return cmap
	

if __name__ == "__main__":
	latent_size = 32
	model_type = 'lstm'
	# main(latent_size)
	# tsne_param(model_type = 'merged')
	# param_plot(latent_size, 'gru', no_evals = 100, val_split = 0.9)
	# param_plot(latent_size, 'lstm', no_evals = 300, val_split = 0.2, dir = 'lstm/')
	# pca(latent_size = latent_size, dim = 3, model_type = model_type, no_evals = 300, v_frac = 0.2, suffix = 'LSTM_')
	thresh = 1.75
	pca(latent_size = latent_size, dim = 2, model_type = model_type, no_evals = 300, v_frac = 0.2, suffix = 'LSTM_', thresh = thresh)
	param_dist(thresh = thresh)
	size_dist(thresh = thresh)
	# tsne_perf(model_type = 'merged')
	# pca(3, model_type = 'gru', no_evals = 100, v_frac = 0.9, suffix = 'GRU_')
	# pca(3)
	# for size in [64, 128, 256, 512]:
		# pca_size(size, dim = 2)
		# pca_size(size, dim = 3)
	