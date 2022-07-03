from torch import nn
import torch
from torch.autograd import Variable
import math

class coRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dt=0.042, gamma=2.7, epsilon=4.7, num_layers = 1,
                 bias=False, batch_first=True, is_cuda=True):
        super(coRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.is_cuda = is_cuda
        self.get_init()


    def forward(self, x, h):
        # x.shape = [batch_size, time_steps, input_dim]
        ## initialize hidden states
        if h is None:
            hz = torch.zeros(x.size(0), self.hidden_size)
            hy = torch.zeros(x.size(0), self.hidden_size)
        else:
            hz = h[0]
            hy = h[1]

        hzs = torch.zeros([x.size(0), x.size(1), self.hidden_size])
        hys = torch.zeros([x.size(0), x.size(1), self.hidden_size])
        if self.is_cuda:
            hz = hz.to('cuda')
            hy = hy.to('cuda')
            hzs = hzs.to('cuda')
            hys = hys.to('cuda')

        for t in range(x.size(1)):
            hz = hz + self.dt * (torch.tanh(self.wy(hy) + self.wz(hz) + self.V(x[:, t, :])) - self.gamma * hy - self.epsilon * hz)
            hzs[:, t, :] = hz
            hy = hy + self.dt * hz
            hys[:, t, :] = hy
        return hys, (hy, hz)

    def get_init(self, params=None):
        if params is None:
            params = {'a': -1, 'b': 1}
        self.wy = nn.Linear(self.hidden_size, self.hidden_size)
        self.wz = nn.Linear(self.hidden_size, self.hidden_size)
        self.V = nn.Linear(self.input_size, self.hidden_size)
        # nn.init.uniform_
        if self.is_cuda:
            self.wy = self.wy.cuda()
            self.wz = self.wz.cuda()
            self.V = self.V.cuda()
        nn.init.uniform_(self.wy.weight, **params)
        nn.init.uniform_(self.wz.weight, **params)
        nn.init.uniform_(self.V.weight, **params)
        self.all_weights = [[self.V.weight, torch.cat((self.wy.weight, self.wz.weight), dim=0)]]

def param_split(model_params, bias):
    #   model_params should be tuple of the form (W_i, W_h, b_i, b_h)
    hidden_size = int(model_params[0][0].shape[0])
    layers = len(model_params)
    W = []
    U = []
    b_i = []
    b_h = []
    if bias:
        param_list = (W, U, b_i, b_h)
    else:
        param_list = (W, U)
    grouped = []
    for idx, param in enumerate(param_list):
        for layer in range(layers):
            #             if len(param.shape) == 1:
            #                 param = param.squeeze(dim=1)
            param.append(model_params[layer][idx].detach())
        grouped.append(param)
    return grouped


def main():
    batch_size = 1
    seq_len = 1
    n_inp = 2
    n_hid = 3
    n_out = 1
    num_layers = 1
    model = coRNN(n_inp=n_inp, n_hid=n_hid, n_out=n_out, dt=0.1, gamma=0.01, epsilon=0.01)
    hy = Variable(torch.ones(num_layers, batch_size, n_hid))
    hz = Variable(torch.ones(num_layers, batch_size, n_hid))
    u = Variable(torch.ones(batch_size, seq_len, n_inp))
    # model.cornn_jac(hy, hz, u)
    W, U = param_split(model.all_weights, bias=False)
    print(model.cell.wy.weight, model.cell.wz.weight)
    print(W, U)
if __name__ == '__main__':
    main()
