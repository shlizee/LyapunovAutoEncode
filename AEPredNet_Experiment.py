import torch
from torch import nn


class AEPredNet(nn.Module):
    '''
    A simple vanilla autoencoder with three intermediate layers. The latent space can be used to predict the validation loss.
    Encoder: 3 fully connected layers which reduce the dimensionality of the data by a factor of two. Each layer has 'tanh' activation.
    Latent Space:

    '''

    def __init__(self, input_size=512, latent_size=128, lr=1e-3, decay_rate=0.999, dtype=torch.float, p_drop=0.1,
                 device=torch.device('cpu'), act='tanh', act_param=0.01, prediction_loss_type='L1'):
        super(AEPredNet, self).__init__()
        self.input_size = input_size
        self.lr = lr
        self.learning_rate_decay = decay_rate
        self.dtype = dtype
        self.device = device
        self.global_step = 0
        self.train_loss = torch.zeros((0,))
        self.val_loss = torch.zeros((0,))
        self.vl1 = torch.zeros((0,))
        self.vl2 = torch.zeros((0,))
        self.alphas = torch.zeros((0, 2))
        self.prediction_loss_type = prediction_loss_type
        self.latent_size = latent_size

        self.drop = nn.Dropout(p=p_drop)
        if act == 'tanh':
            self.act = nn.Tanh()
        if act == 'relu':
            self.act = nn.ReLU()
        if act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=act_param)

        # Encoder
        self.fc_e1 = nn.Linear(in_features=input_size, out_features=input_size // 4)
        # self.fc_e2 = nn.Linear(in_features=input_size // 2, out_features=input_size // 4)
        self.fc_e3 = nn.Linear(in_features=input_size // 4, out_features=latent_size)

        # Prediction
        self.prediction = nn.Linear(in_features=latent_size, out_features=1)
        # self.pred_activation = nn.ReLU()
        self.pred_activation = nn.Sigmoid()
        # Decoder
        self.fc_d1 = nn.Linear(in_features=latent_size, out_features=input_size // 4)
        # self.fc_d2 = nn.Linear(in_features=input_size // 4, out_features=input_size // 2)
        self.fc_d3 = nn.Linear(in_features=input_size // 4, out_features=input_size)

        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.rec_loss = nn.L1Loss()
        if self.prediction_loss_type == 'MSE':
            self.pred_loss = nn.MSELoss()
        elif self.prediction_loss_type == 'L1':
            self.pred_loss = nn.L1Loss()
        elif self.prediction_loss_type == 'BCE':
            self.pred_loss = nn.BCELoss()

        self.best_val = 1e7
        self.best_state = self.state_dict()

    def forward(self, input, predict=True):
        # input = input.to(self.device)
        enc = self.drop(self.act(self.fc_e1(input)))
        # enc = self.drop(self.act(self.fc_e2(enc)))
        enc = self.fc_e3(enc)

        latent = enc

        dec = self.drop(self.act(self.fc_d1(enc)))
        # dec = self.drop(self.act(self.fc_d2(dec)))
        dec = self.drop(self.fc_d3(dec))

        out = dec

        if predict:
            # print("prediction target:", self.prediction(latent).squeeze())
            prediction = self.prediction(latent).squeeze()
            prediction = self.pred_activation(prediction)
            return out, latent, prediction

        else:
            return out, latent, torch.zeros_like(out)

    def train_step_ae(self, input, targets=None, alpha=1, predict=True):
        self.opt.zero_grad()
        self.train()
        # print(input.shape)
        outputs = self(input)
        out = outputs[0]
        loss1 = self.rec_loss(input, out)
        if predict:
            pred = outputs[2]
            # loss2 = self.pred_loss(pred, targets)
            targets = targets.type(torch.FloatTensor).to(out.device)
            loss2 = self.pred_loss(pred, targets)
        else:
            loss2 = torch.zeros_like(loss1)
        loss = loss1 + alpha * loss2
        loss.backward()
        self.opt.step()
        return loss, outputs

    def val_step_ae(self, input, targets=None, alpha=1, predict=True):
        self.opt.zero_grad()
        self.eval()
        with torch.no_grad():
            outputs = self(input)
            out = outputs[0]
            loss1 = self.rec_loss(input, out)
            if predict:
                pred = outputs[-1]
                targets = targets.type(torch.FloatTensor).to(out.device)
                loss2 = self.pred_loss(pred, targets)
            else:
                loss2 = torch.zeros_like(loss1)
            loss = loss1 + alpha * loss2
            return loss, loss1, loss2, outputs
