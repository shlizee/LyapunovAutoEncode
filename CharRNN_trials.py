import torch
from models import RNNModel
from config import *
from training import *
import time
import pickle as pkl

class CharRNNTrials(object):
	def __init__(self, fcon, min_pval = 0.04, max_pval = 0.40, hidden_size = 512, evals = 300, keep_amt = 0.4, model_type = 'lstm'):
		self.params = torch.round((torch.rand(evals)*(max_pval - min_pval) + min_pval)*10**3)/(10**3)
		self.keep_amt = keep_amt
		# print(self.params)
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		self.train_losses = []
		self.val_losses= torch.zeros((evals,))
		self.train_trials(fcon)
		
		
	def train_trials(self, fcon, model_type = 'lstm', hidden_size = 512):
		for idx, p in enumerate(self.params):
			start_time = time.time()
			print(f'Training network {idx + 1} out of {len(self.params)}', end = ', ')
			# mcon = ModelConfig(model_type, num_layers= 1, hidden_size = hidden_size, input_size = self.fcon.data.input_size, output_size = self.fcon.data.input_size,
								# dropout = 0.1, init_type = 'uni', device = self.device, init_params = {'a':-p, 'b':p}, bias = False, id_init_param = 'b')
			fcon.model.init_params = {'a': -p, 'b':p}
			trial_data = self.load_trial_data(fcon.data, keep_amt = self.keep_amt)
			model = RNNModel(fcon.model).to(self.device)
			optimizer = fcon.train.get_optimizer(model.parameters())
			train_loss, val_loss = train_model(fcon, model, optimizer, trial_data = trial_data, verbose = False, save_interval = 3)
			self.train_losses.append(train_loss)
			self.val_losses[idx] = val_loss
			end_time = time.time()
			time_diff = end_time - start_time
			remaining_time = (len(self.params)- (idx+1)) * time_diff
			print(f'Training time {time_diff:.2f}s, Expected remaining time: {remaining_time:.2f}s')
		
		
	def load_trial_data(self, config, keep_amt = 0.4):
		train_data, train_targets = config.datasets['train_set']
		train_len = train_data.shape[0]
		val_data, val_targets = config.datasets['val_set']
		val_len = val_data.shape[0] 
		train_idx = torch.randperm(train_len)
		val_idx = torch.randperm(val_len)
		trial_data_train = train_data[train_idx][:int(keep_amt*train_len)].to(self.device)
		trial_targets_train = train_targets[train_idx][:int(keep_amt*train_len)].to(self.device)
		trial_data_val = val_data[val_idx][:int(keep_amt*val_len)].to(self.device)
		trial_targets_val = val_targets[val_idx][:int(keep_amt*val_len)].to(self.device)
		return {'train_set': (trial_data_train, trial_targets_train), 'val_set': (trial_data_val, trial_targets_val)}
	
	def LE_spectra(self, fcon, lcon,  le_data, keep_amt = 0.4, seq_length = 500, warmup = 500, epoch = 15):
		self.all_LEs = torch.zeros(len(self.params), fcon.model.rnn_atts['hidden_size']).to(self.device)
		hidden_size = fcon.model.rnn_atts['hidden_size']
		for idx, p in enumerate(self.params):
			if (idx%25) == 0:
				print(f'Network {idx +1} of {len(self.params)}:')
			fcon.model.init_params = {'a': -p, 'b':p}
			ckpt = load_checkpoint(fcon, epoch)
			model = ckpt[0].to(self.device)
			LE_stats, _ = lcon.calc_lyap(le_data, model, fcon)
			self.all_LEs[idx] = LE_stats[0]
			torch.save(LE_stats, 'LE_stats/{}_LE_stats_e{}.p'.format(fcon.name(), epoch))
		torch.save(self.all_LEs, f'LE_stats/{fcon.model.model_type}_{hidden_size}_allLEs.p')
		
	
def main(size = 64, model_type = 'lstm'):
	test = True
	batch_size = 32
	le_batch_size = 10
	max_epoch = 15
	keep_amt = 0.2
	evals = 50
	dcon = TestDataConfig('', test_file = 'book_data.p', train_frac = 0.8, val_frac = 0.2, test_frac = 0.0)   
	# torch.save(dcon, 'dcon.p')
	# dcon = torch.load('dcon.p')
	# print('dcon saved')
	# hidden_size = size
	trials_dir = f'{model_type}'
	if test:
		trials_dir = f'test_{trials_dir}'
	learning_rate = 0.002
	dropout = 0.1
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	mcon = ModelConfig(model_type, 1, 64, dcon.input_size, output_size = dcon.input_size, dropout=dropout, 
						init_type = 'uni', device = device, bias = False, id_init_param = 'b', encoding = 'one_hot')
	tcon = TrainConfig('Models', batch_size, max_epoch, 'adam', learning_rate, {}, start_epoch = 0)
	for hidden_size in [64, 128, 256, 512]:
		torch.save([], f'{trials_dir}/test.p')
		print(f'Hidden Size: {hidden_size}')
		mcon.rnn_atts['hidden_size'] = hidden_size
		fcon = FullConfig(dcon, tcon, mcon)
		trials = CharRNNTrials(fcon, hidden_size = hidden_size, evals= evals, keep_amt = keep_amt)
		torch.save(trials, f'{trials_dir}/CharRNNTrials_keep{keep_amt}_size{hidden_size}.p')
	fcon = FullConfig(dcon, tcon, mcon)
	lcon = LyapConfig(batch_size = le_batch_size, seq_length = 100, ON_step = 1, warmup = 500, one_hot= True)
	print('Retrieving LE data')
	le_data = lcon.get_input(fcon)
	for hidden_size in [64, 128, 256, 512]:
		print(f'Hidden Size: {hidden_size}')
		mcon.rnn_atts['hidden_size'] = hidden_size
		fcon = FullConfig(dcon, tcon, mcon)
		trials = torch.load(f'{trials_dir}/CharRNNTrials_keep{keep_amt}_size{hidden_size}.p')
		trials.LE_spectra(fcon, lcon, le_data, keep_amt = keep_amt)
		torch.save(trials, f'{trials_dir}/CharRNNTrials_keep{keep_amt}_size{hidden_size}.p')
		
def extract_trials(size, dir = '', model_type = 'lstm', keep = 0.4):
	trials = torch.load(f'{dir}/CharRNNTrials_keep{keep}_size{size}.p')
	le_data = trials.all_LEs
	valLoss = trials.val_losses
	params = trials.params
	torch.save(le_data, f'{dir}/{model_type}_{size}_LEs.p')
	torch.save(valLoss, f'{dir}/{model_type}_{size}_valLoss.p')
	torch.save(params, f'{dir}/{model_type}_{size}_params.p')
	
if __name__ == '__main__':
	main(model_type = 'lstm')
	# hidden_size = 64
	# keep_amt = 0.4
	# model_type = 'lstm'
	# trials = torch.load(open(f'CharRNNTrials_keep{keep_amt}_size{hidden_size}.p', 'rb'))
	# print(trials)
	# torch.save(trials, f'CharRNNTrials_keep{keep_amt}_size{hidden_size}.p')
	# trials.all_LEs = torch.load(f'LE_stats/{model_type}_{hidden_size}_allLEs.p')