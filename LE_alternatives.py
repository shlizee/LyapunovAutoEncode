import torch
from AEPredNet import AEPredNet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.linear_model import LinearRegression
import numpy as np

def latent_plot():
	if torch.cuda.is_available():
		device= torch.device('cuda')
	else: 
		device= torch.device('cpu')
	model = torch.load('Models/ae_prednet_4000.ckpt').cpu()
	model.load_state_dict(model.best_state)
	x_data = torch.load('Processed/lstm_allLEs.p')
	
	indices = [0, 300, 600, 900, 1200]
	sizes = [64, 128, 256, 512]
	splits = []
	i_list = torch.arange(1200)
	splits = [(i_list>torch.ones_like(i_list)*indices[i])*(i_list<torch.ones_like(i_list)*indices[i+1]) for i in range(len(indices)-1)]
	
	latents=  model(x_data)[1]
	print(latents.shape)
	
	plt.plot(torch.arange(32), latents)

def LE_regression(model_type = 'lstm'):
	split = torch.load(f'Processed/{model_type}_data_split_vfrac0.2.p')
	x_train, y_train, x_val, y_val = split['train_data'], split['train_targets'], split['val_data'], split['val_targets']
	max_train = torch.max(x_train, dim = 1)[0]
	max_val = torch.max(x_val, dim = 1)[0]
	mean_train = torch.mean(x_train, dim = 1)
	mean_val = torch.mean(x_val, dim = 1)
	pred_loss = torch.nn.MSELoss()
	
	full_reg = LinearRegression().fit(x_train, y_train)
	print(full_reg.coef_)
	print(f'Torch Loss: {pred_loss(torch.Tensor(full_reg.predict(x_val)), y_val)}')
	print(f'Full LE Prediction Error: {1/len(y_val)*np.sum((full_reg.predict(x_val.cpu().numpy())-y_val.cpu().numpy())**2)}')
	# plt.plot(full_reg.predict(x_val)-y_val.numpy())
	# plt.show()
	
	max_reg = LinearRegression().fit(max_train.view(-1, 1), y_train)
	print(max_reg.coef_)
	print(f'Max LE Prediction Error: {1/len(y_val)*np.sum((max_reg.predict(max_val.view(-1,1))-y_val.cpu().numpy())**2)}')
	
	mean_reg = LinearRegression().fit(mean_train.view(-1, 1), y_train)
	print(mean_reg.coef_)
	print(f'Mean LE Prediction Error: {1/len(y_val)*np.sum((mean_reg.predict(mean_val.view(-1,1))-y_val.cpu().numpy())**2)}')
	
	regs = {'full': full_reg, 'max': max_reg, 'mean': mean_reg}
	torch.save(regs, 'Processed/lstm_LinModels.p')
	
	
	
	
	
	
if __name__ == "__main__":
	model_type = 'lstm'
	LE_regression(model_type)
	no_evals = 300
	models = torch.load(f'Processed/{model_type}_LinModels.p')
	full_reg, max_reg, mean_reg = (models['full'], models['max'], models['mean'])
	
	split = torch.load(f'Processed/{model_type}_data_split_vfrac0.2.p')
	x_train, y_train, x_val, y_val = split['train_data'], split['train_targets'], split['val_data'], split['val_targets']
	max_val = torch.max(x_val, dim = 1)[0]
	mean_val = torch.mean(x_val, dim = 1)
	
	full_pred = full_reg.predict(x_val)
	max_pred = full_reg.predict(x_val)
	mean_pred = full_reg.predict(x_val)
	
	x_data = torch.load(f'Processed/{model_type}/{model_type}_allLEs.p')
	params = torch.load(f'Processed/{model_type}/{model_type}_allParams.p').flatten()
	indices = [0, 1*no_evals, 2*no_evals, 3*no_evals, 4*no_evals]
	sizes = [64, 128, 256, 512]
	val_idx = split['val_idx']
	val_splits = []
	
	for type in ['full', 'max', 'mean']:
		model = models[type]
		plt.figure()
		for i in range(len(sizes)):
			val_splits.append(((val_idx>torch.ones_like(val_idx)*indices[i])*(val_idx<torch.ones_like(val_idx)*indices[i+1])))
			# print(torch.arange(1200).float()>torch.ones(1200)*indices[i])
			# print(val_idx[val_splits[i]].shape)
			# print(params[val_idx[val_splits[i]]].shape)
			if type == 'full':
				x = x_data
			else:
				x = x_data.view(-1, 1)
				
			plt.scatter(params[val_idx[val_splits[i]]], model.predict(x[val_idx[val_splits[i]]]), label = sizes[i], s = 14)
		
		plt.legend(loc = 2)
		plt.xlabel('Init Param')
		plt.ylim([1.1, 3.0])
		plt.ylabel('Validation Loss \n (Predicted)')
		plt.title(f'{type} LE Predictions')
		plt.savefig(f'Figures/LE{type}_{model_type}_paramPlot.png', bbox_inches="tight",dpi=200)
		
	plt.figure()
	targets = torch.load(f'Processed/{model_type}/{model_type}_allValLoss.p')
	for i in range(4): 
		plt.scatter(params[val_idx[val_splits[i]]], targets[val_idx[val_splits[i]]], label = sizes[i], s = 14)
	plt.legend(prop = {'size':12}, loc = 2)
	plt.ylabel('Val Loss\n(Actual)')
	plt.xlabel('Init Param')
	plt.ylim([1.1, 3.0])
	plt.title(f'Ground Truth')
	plt.savefig(f'Figures/Actual_{model_type}_paramPlot.png', bbox_inches="tight",dpi=200)