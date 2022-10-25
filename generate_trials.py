import torch
import os
import argparse
from models import RNNModel
from config import *
from training import *
import time
import pickle as pkl


class SMNISTTrails(object):
    def __init__(self, fcon, min_pval=0.1, max_pval=30, hidden_size=512, evals=300, keep_amt=0.4, model_type='lstm'):
        self.params = torch.round((torch.rand(int(evals)) * (max_pval - min_pval) + min_pval) * 10 ** 3) / (10 ** 3)
        self.keep_amt = keep_amt
        # print(self.params)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.train_losses = []
        self.val_losses = torch.zeros((int(evals),))
        self.train_trials(fcon)

    def train_trials(self, fcon, model_type='lstm', hidden_size=512):
        for idx, p in enumerate(self.params):
            start_time = time.time()
            print(f'Training network {idx + 1} out of {len(self.params)}.\n'
                  f'init param: [{-p}, {p}]\fn')
            fcon.model.init_params = {'a': -p, 'b': p}
            model = RNNModel(fcon.model).to(self.device)
            optimizer = fcon.train.get_optimizer(model.parameters())
            train_loss, val_loss = train_model_SMNIST(fcon, model, optimizer, start_epoch=0, save_interval=5)
            self.train_losses.append(train_loss)
            self.val_losses[idx] = val_loss
            end_time = time.time()
            time_diff = end_time - start_time
            remaining_time = (len(self.params) - (idx + 1)) * time_diff
            print(f'Training time {time_diff:.2f}s, Expected remaining time: {remaining_time:.2f}s')

    def LE_spectra(self, fcon, lcon, le_data, keep_amt=0.4, seq_length=500, warmup=500, epoch=1):
        self.all_LEs = torch.zeros(len(self.params), fcon.model.rnn_atts['hidden_size']).to(self.device)
        hidden_size = fcon.model.rnn_atts['hidden_size']
        for idx, p in enumerate(self.params):
            if (idx % 25) == 0:
                print(f'Network {idx + 1} of {len(self.params)}:')
            fcon.model.init_params = {'a': -p, 'b': p}
            epoch_ = epoch
            print(f'{fcon.train.model_dir}/{fcon.model.model_type}/{fcon.name()}_e{epoch}.ckpt')
            while (not os.path.exists(f'{fcon.train.model_dir}/{fcon.model.model_type}/{fcon.name()}_e{epoch}.ckpt')):
                epoch_ -= 1
            ckpt = load_checkpoint(fcon, epoch_)
            model = ckpt[0].to(self.device)


            LE_stats, _ = lcon.calc_lyap(le_data, model, fcon)
            self.all_LEs[idx] = LE_stats[0]
            # print(f"LE_stats: {LE_stats}")
            if not os.path.exists('LE_stats'):
                os.mkdir('LE_stats')
            # torch.save(LE_stats, 'LE_stats/{}_LE_stats_e{}.p'.format(fcon.name(), epoch))
        torch.save(self.all_LEs, f'LE_stats/{fcon.model.model_type}_{hidden_size}_allLEs_e{epoch}.p')

    def val_loss_epoch(self, fcon, epoch):
        val_losses = torch.zeros((len(self.params)))
        for idx, p in enumerate(self.params):
            start_time = time.time()
            print(f'Testing network {idx + 1} out of {len(self.params)}.\n'
                  f'init param: [{-p}, {p}]\fn')
            fcon.model.init_params = {'a': -p, 'b': p}
            ckpt = load_checkpoint(fcon, epoch)
            model = ckpt[0].to(self.device)
            val_loss = test_model_SMNIST(fcon, model, epoch)
            val_losses[idx] = val_loss
            end_time = time.time()
            time_diff = end_time - start_time
            remaining_time = (len(self.params) - (idx + 1)) * time_diff
            print(f'Testing time {time_diff:.2f}s, Expected remaining time: {remaining_time:.2f}s')
            # if idx > 3:
            #     break
        return val_losses
# calculate LEs
# h = torch.zeros(model.num_layers, images.size(0), model.hidden_size).to(model.device)
# c = torch.zeros(model.num_layers, images.size(0), model.hidden_size).to(model.device)
# params = (images, (h, c))
# if i == 0:
#     LEs, rvals = calc_LEs_an(*params, model=model)


class CharRNNTrials(object):
    def __init__(self, fcon, min_pval=0.1, max_pval=3, hidden_size=512, evals=300, keep_amt=0.4, model_type='lstm'):
        self.params = torch.round((torch.rand(int(evals)) * (max_pval - min_pval) + min_pval) * 10 ** 3) / (10 ** 3)
        self.keep_amt = keep_amt
        # print(self.params)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.train_losses = []
        self.val_losses = torch.zeros((int(evals),))
        self.train_trials(fcon)

    def train_trials(self, fcon, model_type='lstm', hidden_size=512):
        for idx, p in enumerate(self.params):
            start_time = time.time()
            print(f'Training network {idx + 1} out of {len(self.params)}\n')
            fcon.model.init_params = {'a': -p, 'b': p}
            trial_data = self.load_trial_data(fcon.data, keep_amt=self.keep_amt)
            model = RNNModel(fcon.model).to(self.device)
            optimizer = fcon.train.get_optimizer(model.parameters())
            # print(f'Trial Data shape: {trial_data}')
            train_loss, val_loss = train_model(fcon, model, optimizer, trial_data=trial_data, verbose=False,
                                               save_interval=5)
            self.train_losses.append(train_loss)
            self.val_losses[idx] = val_loss
            end_time = time.time()
            time_diff = end_time - start_time
            remaining_time = (len(self.params) - (idx + 1)) * time_diff
            print(f'Training time {time_diff:.2f}s, Expected remaining time: {remaining_time:.2f}s')

    def load_trial_data(self, config, keep_amt=0.4):
        train_data, train_targets = config.datasets['train_set']
        train_len = train_data.shape[0]
        val_data, val_targets = config.datasets['val_set']
        val_len = val_data.shape[0]
        train_idx = torch.randperm(train_len)
        val_idx = torch.randperm(val_len)
        trial_data_train = train_data[train_idx][:int(keep_amt * train_len)].to(self.device)
        trial_targets_train = train_targets[train_idx][:int(keep_amt * train_len)].to(self.device)
        trial_data_val = val_data[val_idx][:int(keep_amt * val_len)].to(self.device)
        trial_targets_val = val_targets[val_idx][:int(keep_amt * val_len)].to(self.device)
        return {'train_set': (trial_data_train, trial_targets_train), 'val_set': (trial_data_val, trial_targets_val)}

    def LE_spectra(self, fcon, lcon, le_data, keep_amt=0.4, seq_length=500, warmup=500, epoch=1, task=''):
        self.all_LEs = torch.zeros(len(self.params), fcon.model.rnn_atts['hidden_size']).to(self.device)
        hidden_size = fcon.model.rnn_atts['hidden_size']
        for idx, p in enumerate(self.params):
            if (idx % 25) == 0:
                print(f'Network {idx + 1} of {len(self.params)}:')
            fcon.model.init_params = {'a': -p, 'b': p}
            ckpt = load_checkpoint(fcon, epoch)
            model = ckpt[0].to(self.device)
            i = torch.randint(low=0, high=le_data.shape[0], size=(1,)).item()

            LE_stats, _ = lcon.calc_lyap(le_data[i], model, fcon)

            self.all_LEs[idx] = LE_stats[0]
            torch.save(LE_stats, f'{task}/LE_stats/{fcon.name()}_LE_stats_e{epoch}.p')
        torch.save(self.all_LEs, f'{task}/LE_stats/{fcon.model.model_type}_{hidden_size}_allLEs_e{epoch}.p')


def main(args):
    parser = argparse.ArgumentParser(description="Train recurrent models")
    parser.add_argument("-model", "--model_type", type=str, default='rnn', required=False)
    parser.add_argument("-evals", "--evals", type=float, default=2, required=False)
    parser.add_argument("-task", "--task", type=str, default='CharRNN', required=False)
    args = parser.parse_args(args)
    model_type = args.model_type
    evals = args.evals
    task = args.task

    # Non-argument hyperparameters
    size = 64  # Ignore this value (needed for initial creation of Config object but meaningless)
    batch_size = 32  # Training batch size
    max_epoch = 10  # Max epochs for which models are trained
    keep_amt = 0.2  # Proportion of training set kept for each trial, to maintain independence of models

    # LE Hyperparameters
    le_batch_size = 15  # Batch size used for LE calculation
    le_seq_length = 100  # Length of LE sequence (in theory, should be infinity, but usually converges after finite number)
    ON_step = 1  # Number of steps between orthonormalization. Greater than 1 means calculations are faster but less precise for more negative LEs
    warmup = 500  # Number of steps to evolve the system before calculating LEs. Helps to remove transients by relaxing onto attractor



    # Define Hyperparameters
    learning_rate = 0.002
    dropout = 0.1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # model_type = 'lstm'
    # max_epoch = 10
    # evals = 200
    # task = 'SMNIST'

    trials_dir = f'{task}/trials/{model_type}'
    if not os.path.isdir(trials_dir):
        # os.makedirs(trials_dir)
        os.mkdir(trials_dir)
    print(f"trials_dir: {trials_dir}")
    if task == 'CharRNN':
        print('Training CharRNN')
        # Define data configuration and trial directory
        dcon = TestDataConfig('CharRNN/data/', test_file='book_data.p', train_frac=0.8, val_frac=0.2, test_frac=0.0)
        # Create Model and Training Configuration Objects
        mcon = ModelConfig(model_type, 1, 64, dcon.input_size, output_size=dcon.input_size, dropout=dropout,
                           init_type='uni', device=device, bias=False, id_init_param='b', encoding='one_hot')
        tcon = TrainConfig(f'Models/{task}', batch_size, max_epoch, 'adam', learning_rate, {}, start_epoch=0)

        # Train Models
        for hidden_size in [64, 128, 256, 512]:
        # for hidden_size in [256]:
            print(f'Hidden Size: {hidden_size}')
            mcon.rnn_atts['hidden_size'] = hidden_size
            fcon = FullConfig(dcon, tcon, mcon)
            # print(fcon.name())
            #trials = CharRNNTrials(fcon, hidden_size=hidden_size, evals=evals, keep_amt=keep_amt)
            #torch.save(trials, f'{trials_dir}/CharRNN_Trials_keep{keep_amt}_size{hidden_size}.p')
            #
        # Calculate Lyapunov Exponents
        lcon = LyapConfig(batch_size=le_batch_size, seq_length=le_seq_length, ON_step=1, warmup=500, one_hot=True)
        print('Retrieving LE data')
        le_data = lcon.get_input(fcon)
        # for hidden_size in [64, 128, 256, 512]:
        for hidden_size in [512]:
            print(f'Hidden Size: {hidden_size}')
            mcon.rnn_atts['hidden_size'] = hidden_size
            fcon = FullConfig(dcon, tcon, mcon)
            trials = torch.load(f'{trials_dir}/CharRNN_Trials_keep{keep_amt}_size{hidden_size}.p')
            trials.LE_spectra(fcon, lcon, le_data, keep_amt=keep_amt, epoch=max_epoch, task=task)
            torch.save(trials, f'{trials_dir}/CharRNN_Trials_keep{keep_amt}_size{hidden_size}.p')
            # print(trials)
    elif task == 'SMNIST':
        print("Training SMNIST")
        dcon = TestDataConfig('data/MNIST/', test_file='MNIST_train_test.p', train_frac=0.8, val_frac=0.2, test_frac=0.0)
        mcon = ModelConfig(model_type, num_layers=1, hidden_size=64, input_size=28, output_size=10,
                           dropout=dropout, init_type='uni', device=device, bias=False, id_init_param='b',
                           encoding='one_hot')
        tcon = TrainConfig(model_dir=f'Models/{task}', batch_size=128, max_epoch=max_epoch, optimizer='adam',
                           learning_rate=0.01, optim_params={}, start_epoch=0)

        #Train Models
        # for hidden_size in [64, 128, 256, 512]:
        # for hidden_size in [64]:
        #     print(f'Hidden Size: {hidden_size}')
        #     mcon.rnn_atts['hidden_size'] = hidden_size
        #     fcon = FullConfig(dcon, tcon, mcon)
            # print(fcon.name())
            # trials = SMNISTTrails(fcon, hidden_size=hidden_size, evals=evals, keep_amt=keep_amt)
            # #
            # torch.save(trials, f'{trials_dir}/SMNIST_Trials_keep{keep_amt}_size{hidden_size}.p')

        # Calculate Lyapunov Exponents
        fcon = FullConfig(dcon, tcon, mcon)
        lcon = LyapConfig(batch_size=le_batch_size, seq_length=le_seq_length, ON_step=1, warmup=500, one_hot=True)
        print('Retrieving LE data')
        le_data = lcon.get_mnist_input(fcon)
        # for hidden_size in [64, 128, 256, 512]:
        for hidden_size in [64]:
            print(f'Hidden Size: {hidden_size}')
            mcon.rnn_atts['hidden_size'] = hidden_size
            fcon = FullConfig(dcon, tcon, mcon)
            trials = torch.load(f'{trials_dir}/SMNIST_Trials_keep{keep_amt}_size{hidden_size}.p')
            print(trials.val_losses.shape)
            for epoch in range(max_epoch):
                trials.LE_spectra(fcon, lcon, le_data, keep_amt=keep_amt, epoch=epoch)
            trials.LE_spectra(fcon, lcon, le_data, keep_amt=keep_amt, epoch=max_epoch)
            torch.save(trials, f'{trials_dir}/SMNIST_Trials_keep{keep_amt}_size{hidden_size}.p')

        # calculate the validation loss
        # for hidden_size in [64]:
        #     print(f'Hidden Size: {hidden_size}')
        #     mcon.rnn_atts['hidden_size'] = hidden_size
        #     fcon = FullConfig(dcon, tcon, mcon)
        #     trials = torch.load(f'{trials_dir}/SMNIST_Trials_keep{keep_amt}_size{hidden_size}.p')
        #     print(trials.val_losses.shape)
        #     trials.val_loss_epochs = {}
        #     for epoch in range(max_epoch + 1):
        #         val_loss_epoch = trials.val_loss_epoch(fcon, epoch=epoch)
        #         trials.val_loss_epochs[epoch] = val_loss_epoch
        #     torch.save(trials, f'{trials_dir}/SMNIST_Trials_keep{keep_amt}_size{hidden_size}.p')
            # print(trials)


def extract_trials(size, dir='', model_type='lstm', task_type='CharRNN', keep=0.2):
    # trials = torch.load(f'{dir}/CharRNNTrials_keep{keep}_size{size}.p')
    trials = torch.load(f'{dir}/{task_type}_Trials_keep{keep}_size{size}.p')
    le_data = trials.all_LEs
    valLoss = trials.val_losses
    params = trials.params
    # val_loss_epochs = trials.val_loss_epochs
    # for epoch in val_loss_epochs.keys():
    #     print(val_loss_epochs[epoch].shape)
    # print(f'Directory: {dir}')
    save_dir = f'{dir}/Data'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(le_data, f'{save_dir}/{model_type}_{size}_LEs.p')
    torch.save(valLoss, f'{save_dir}/{model_type}_{size}_valLoss.p')
    torch.save(params, f'{save_dir}/{model_type}_{size}_params.p')

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
    # model_type = 'asrnn'
    # extract_trials(64, f'trials/SMNIST/{model_type}/', model_type=model_type, task_type='SMNIST')
