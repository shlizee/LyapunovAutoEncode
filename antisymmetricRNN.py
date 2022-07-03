# from config import ModelConfig
import torchvision
import matplotlib.pyplot as plt
import time
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class AntisymmetricRNN(nn.Module):
	def __init__(self, input_size, hidden_size=32, eps=0.01, gamma=0.01, init_W_std=1,
				 is_cuda=True, num_layers=1, bias = False, batch_first = True):
		super(AntisymmetricRNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.eps = eps
		self.gamma = gamma
		self.gamma_I = torch.eye(self.hidden_size, self.hidden_size).cuda() * self.gamma
		self.init_W_std = init_W_std
		self.is_cuda = is_cuda
		self.num_layers = num_layers
		self.bias = bias
		self.batch_first = batch_first
		self.get_init(self.init_W_std)

	def get_init(self, init_W_std):
		normal_sampler_ih_l0 = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([1 / self.input_size]))
		weight_ih_l0 = nn.Parameter(normal_sampler_ih_l0.sample((self.hidden_size, self.input_size))[..., 0])
		bias_ih_l0 = nn.Parameter(torch.zeros(self.hidden_size))
		self.weight_ih_l0 = nn.Linear(self.input_size, self.hidden_size)
		self.weight_ih_l0.weight = weight_ih_l0
		if self.bias:
			self.weight_ih_l0.bias = bias_ih_l0


		# init W
		normal_sampler_W = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([init_W_std/ self.hidden_size]))
		self.weight_W_l0 = nn.Parameter(normal_sampler_W.sample((self.hidden_size, self.hidden_size))[..., 0])
		self.all_weights = [[self.weight_ih_l0.weight, self.weight_W_l0]]

	def forward(self, x, h=None):
		# x.shape = (batch_size, timesteps, input_dim)
		if h is None:
			h = torch.zeros(x.shape[0], self.hidden_size)
		hs = torch.zeros(x.shape[0], x.shape[1]+1, self.hidden_size)
		hs[:, 0, :] = h
		if self.is_cuda:
			hs = hs.cuda()
			h = h.cuda()
		T = x.shape[1]

		# if not self.use_gating:
		for i, t in enumerate(range(T)):
			# (W - WT - gammaI)h
			WmWT_h = torch.matmul(h, (self.weight_W_l0 - self.weight_W_l0.transpose(1, 0) - self.gamma_I))

			# WmWT_h = torch.matmul(h, (self.W - self.W.transpose(1, 0) - self.gamma_I))
			# WmWT_h = torch.matmul(h, self.W)

			# Vhx + bh
			Vh_x = self.weight_ih_l0(x[:, t, :])

			# (W - WT - gammaI)h + Vhx + bh
			linear_transform = WmWT_h + Vh_x

			# tanh((W - WT - gammaI)h + Vhx + bh)
			f = torch.tanh(linear_transform)

			# eq. 12
			h = h + self.eps * f
			hs[:, i+1, :] = h
		return hs[:, 1:, :], torch.unsqueeze(h, dim=0)

def main():
	train = torchvision.datasets.MNIST('./data/MNIST', train=True, download=False,
									   transform=torchvision.transforms.Compose([
										   torchvision.transforms.ToTensor()
									   ]))

	test = torchvision.datasets.MNIST('./data/MNIST', train=False, download=False,
									  transform=torchvision.transforms.Compose([
										  torchvision.transforms.ToTensor()
									  ]))
	x_train = train.train_data
	y_train = train.train_labels
	x_test = test.test_data
	y_test = test.test_labels

	x_train_f = x_train / 255
	x_test_f = x_test / 255
	train_loader = DataLoader(TensorDataset(x_train_f, y_train), shuffle=True, batch_size=512)
	test_loader = DataLoader(TensorDataset(x_test_f, y_test), shuffle=True, batch_size=512)
	model = AntisymmetricRNN(28, 10, hidden_size=32, gamma=0.01, eps=0.01).cuda()
	opt = torch.optim.Adagrad(model.parameters(), lr=0.1)
	loss = nn.CrossEntropyLoss()

	for e in range(1000):
		time_start = time.time()
		ce_train, ce_test, acc_train, acc_test = 0, 0, 0, 0
		for batch_x, batch_y in train_loader:
			batch_x = batch_x.cuda()
			batch_y = batch_y.cuda()
			opt.zero_grad()
			output = model(batch_x)
			l = loss(output, batch_y.long())
			l.backward()
			opt.step()
			ce_train += l.item() * batch_x.shape[0]
			_, preds = torch.max(output, 1)
			acc_train += (preds == batch_y).sum().item()
		ce_train /= len(x_train_f)
		acc_train = acc_train * 100 / len(x_train_f)
		with torch.no_grad():
			for batch_x, batch_y in test_loader:
				batch_x = batch_x.cuda()
				batch_y = batch_y.cuda()
				output = model(batch_x)
				l = loss(output, batch_y.long())
				ce_test += l.item() * batch_x.shape[0]
				_, preds = torch.max(output, 1)
				# print(preds)
				acc_test += (preds == batch_y).sum().item()
		ce_test /= len(x_test_f)
		acc_test = acc_test * 100 / len(x_test_f)
		time_end = time.time()
		print("Iter: ", e, "train loss: ", ce_train, "train acc: ", acc_train,
			  "test loss: ", ce_test, "test acc: ", acc_test, "iter time: ", time_end - time_start)
if __name__ == '__main__':
    main()
