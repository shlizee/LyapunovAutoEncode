import torch
from AEPredNet import AEPredNet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D 

def tsne(model, X, tsne_params = {}):
	encoded = model(X)[1]
	tsne_model = TSNE(**tsne_params)
	X_embedded = tsne_model.fit_transform(encoded.detach().numpy())
	print(X_embedded.shape)
	return tsne_model
	
def main():
	if torch.cuda.is_available():
		device= torch.device('cuda')
	else: 
		device= torch.device('cpu')
	model = torch.load('ae_prednet_4000.ckpt').cpu()
	model.load_state_dict(model.best_state)
	x_data = torch.load('Processed/lstm_allLEs.p')
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

def param_plot():
	model = torch.load('ae_prednet_4000.ckpt').cpu()
	model.load_state_dict(model.best_state)
	x_data = torch.load('Processed/lstm_allLEs.p')
	params = torch.load('lstm_allParams.p')
	split = torch.load('data_split_vfrac0.2.p')
	indices = [0, 300, 600, 900, 1200]
	sizes = [64, 128, 256, 512]
	val_idx = split['val_idx']
	val_splits = []
	plt.figure()
	for i in range(len(sizes)):
		val_splits.append(((val_idx>torch.ones_like(val_idx)*indices[i])*(val_idx<torch.ones_like(val_idx)*indices[i+1])))
		# print(torch.arange(1200).float()>torch.ones(1200)*indices[i])
		# print(val_idx[val_splits[i]].shape)
		plt.scatter(params[val_idx[val_splits[i]]], model(x_data[val_idx[val_splits[i]]])[2].detach(), label = sizes[i], s = 14)
	plt.legend()
	plt.xlabel('Init Param')
	plt.ylim([1.1, 2.6])
	plt.ylabel('Validation Loss \n (Predicted)')
	plt.title('AE Predictions')
	plt.savefig('AEPredNet_paramPlot.png', bbox_inches="tight",dpi=200)
	
	plt.figure()
	targets = torch.load('Processed/lstm_allValLoss.p')
	for i in range(4): 
		plt.scatter(params[val_idx[val_splits[i]]], targets[val_idx[val_splits[i]]], label = sizes[i], s = 14)
	plt.legend(prop = {'size':12})
	plt.ylabel('Val Loss\n(Actual)')
	plt.xlabel('Init Param')
	plt.ylim([1.1, 2.6])
	plt.title(f'Ground Truth')
	plt.savefig('Actual_paramPlot.png', bbox_inches="tight",dpi=200)
	
def tsne_perf():
	indices = [0, 300, 600, 900, 1200]
	sizes = [64, 128, 256, 512]
	targets = torch.load('Processed/lstm_allValLoss.p')
	splits = []
	i_list = torch.arange(1200)
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
	
def tsne_param():
	indices = [0, 300, 600, 900, 1200]
	sizes = [64, 128, 256, 512]
	params = torch.load('lstm_allParams.p')
	splits = []
	i_list = torch.arange(1200)
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

def pca(dim=2):
	if torch.cuda.is_available():
		device= torch.device('cuda')
	else: 
		device= torch.device('cpu')
	model = torch.load('Models/ae_prednet_4000.ckpt').cpu()
	model.load_state_dict(model.best_state)
	x_data = torch.load('Processed/lstm_allLEs.p')
	targets = torch.load('Processed/lstm_allValLoss.p')
	split = torch.load('data_split_vfrac0.2.p')
	indices = [0, 300, 600, 900, 1200]
	sizes = [64, 128, 256, 512]
	i_list = torch.arange(1200)
	splits = [(i_list>torch.ones_like(i_list)*indices[i])*(i_list<torch.ones_like(i_list)*indices[i+1]) for i in range(len(indices)-1)]
	latent = model(x_data)[1].detach()
	U,S,V = torch.pca_lowrank(latent)
	low_rank = torch.matmul(latent, V[:, :dim])
	torch.save(low_rank, f'PCA_dim{dim}.p')
	#Performance PCA Plot
	fig = plt.figure()

	if dim == 3:
		ax = fig.add_subplot(111, projection='3d')
	else:
		ax = fig.add_subplot(111)
	for idx, size in enumerate(sizes):
		y = low_rank[splits[idx]]
		if dim == 3:
			im = ax.scatter(y[:,0], y[:,1], y[:,2], s = 6, c = targets[splits[idx]], norm=colors.LogNorm(vmin=targets.min()-0.2, vmax=targets.max()), cmap = plt.get_cmap('hsv'))
		else:
			im = ax.scatter(y[:,0], y[:,1], s = 6, c = targets[splits[idx]], norm=colors.LogNorm(vmin=targets.min()-0.2, vmax=targets.max()), cmap = plt.get_cmap('hsv'))
	# ax.add_colorbar(label = 'Val Loss')
	plt.colorbar(im, label = 'Val Loss')
	ax.set_xlabel('PCA 1')
	ax.set_ylabel('PCA 2')
	if dim == 3:
		ax.set_zlabel('PCA 3')
	plt.savefig(f'Figures/Latent/AEPredNet_pca_perf_dim{dim}.png', dpi = 200)
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
	plt.savefig(f'Figures/Latent/AEPredNet_pca_size_dim{dim}.png', dpi = 200)
	
def pca_size(size = 512, dim=2):
	if torch.cuda.is_available():
		device= torch.device('cuda')
	else: 
		device= torch.device('cpu')
	model = torch.load('Models/ae_prednet_4000.ckpt').cpu()
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
		im = ax.scatter(y[:,0], y[:,1], y[:,2], s = 6, c = vals, norm=colors.LogNorm(vmin=vals.min(), vmax=vals.max()), cmap = plt.get_cmap('hsv'))
	else:
		im = ax.scatter(y[:,0], y[:,1], s = 6, c = vals, norm=colors.LogNorm(vmin=vals.min(), vmax=vals.max()), cmap = plt.get_cmap('hsv'))

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

	
	

if __name__ == "__main__":
	# main()
	# param_plot()
	# tsne_perf()
	# tsne_param()
	pca(2)
	pca(3)
	for size in [64, 128, 256, 512]:
		pca_size(size, dim = 2)
		pca_size(size, dim = 3)
	