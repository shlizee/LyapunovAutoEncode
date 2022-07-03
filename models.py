import torch
from torch import nn
from config import ModelConfig
		
class RNNModel(nn.ModuleList):
	def __init__(self, model_con, att_name = ''):
		super(RNNModel, self).__init__()
		
		self.con = model_con
		self.L = self.con.rnn_atts['num_layers']*self.con.rnn_atts['hidden_size']
		
		#self.encoder = lambda xt: nn.functional.one_hot(xt.long(), self.con.rnn_atts['input_size']).to(self.con.device)
		self.encoder = self.con.encoder
		self.dropout = nn.Dropout(p = self.con.dropout)
		self.fc = nn.Linear(in_features = self.con.rnn_atts['hidden_size'], out_features = self.con.output_size).cuda()
		
		self.rnn_layer = model_con.get_RNN()
		self.gate_size = model_con.get_gate_size()
		if self.con.model_type == 'asrnn':
			self.rnn_layer.get_init(self.con.init_params['b'])
			# normal_sampler_V = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([1 / input_size]))
			# Vh_init_weight = nn.Parameter(normal_sampler_V.sample((hidden_size, input_size))[..., 0])
			# Vh_init_bias = nn.Parameter(torch.zeros(hidden_size))
			# self.Vh = nn.Linear(input_size, hidden_size)
			# self.Vh.weight = Vh_init_weight
			# if bias:
			# 	self.Vh.bias = Vh_init_bias
			#
			# # init W
			# normal_sampler_W = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([init_W_std / hidden_size]))
			# self.W = nn.Parameter(normal_sampler_W.sample((hidden_size, hidden_size))[..., 0])
		else:
			for layer_p in self.rnn_layer._all_weights:
				for p in layer_p:
					if 'weight' in p:
						# self.con.get_init(self.rnn_layer.__getattr__(p))
						w = getattr(self.rnn_layer, p)
						if w.shape[-1] == self.con.rnn_atts['input_size']:
							self.con.get_init(w, input = True)
						else:
							self.con.get_init(getattr(self.rnn_layer, p))

	def forward(self, xt, h = None):
		if xt.shape[-1] == self.con.rnn_atts['input_size']:
			encoded = xt
		else:
			# print(f'Encoded: {self.encoder(xt)}')
			encoded = self.encoder(xt).to(self.con.device).squeeze()
		if self.con.model_type in ['lstm', 'rnn', 'gru']:
			self.rnn_layer.flatten_parameters()
		rnn_out, rnn_hn = self.rnn_layer(encoded.float(), h)
		d_out = self.dropout(rnn_out)
		output = self.fc(d_out)
		return output, rnn_hn
		
	def init_hidden(self, batch_size):
		h = torch.zeros(self.con.rnn_atts['num_layers'], batch_size, self.con.rnn_atts['hidden_size']).to(self.con.device)
		c = torch.zeros(self.con.rnn_atts['num_layers'], batch_size, self.con.rnn_atts['hidden_size']).to(self.con.device)
		
		if self.con.model_type == 'lstm':
			return (h,c)
		else:
			del c
			return h
