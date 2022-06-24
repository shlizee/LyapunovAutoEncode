import torch
from torch.autograd import Variable
from torch.nn import RNN, GRU, LSTM
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import gc
import math


def calc_Jac(*params, model):
    cells = len(params) > 2
    if cells:
        inputs = params[0]
        states = (params[1], params[2])
        h0, c0 = states
    else:
        inputs, states = params  # inputs should be a single time step with batch_size entries
        h0 = states
    num_layers, batch_size, hidden_size = h0.shape
    _, seq_len, input_size = inputs.shape
    L = num_layers * hidden_size

    # Feed forward into network
    if cells:
        model_out, states_out = model(inputs, states)
        hn, cn = states_out
    else:
        model_out, hn = model(inputs, states)

    # Flatten output from different layers of RNN (needed for J calculation)
    hn_flat = torch.reshape(torch.transpose(hn, 0, 1), (batch_size, 1, L))

    J = torch.zeros(batch_size, L, L)  # placeholder
    for i in range(L):
        hn_flat[:, :, i].backward(torch.ones_like(hn_flat[:, :, i]), retain_graph=True)
        der = h0.grad
        der = torch.reshape(torch.transpose(der, 0, 1), (batch_size, L))
        J[:, i, :] = der
        h0.grad.zero_()
    return J


def oneStep(*params, model):
    # Params is a tuple including h, x, and c (if LSTM)
    l = len(params)
    if l < 2:
        print('Params must be a tuple containing at least (x_t, h_t)')
        return None
    elif l > 2:
        states = (params[1], params[2])
        return model(params[0], states)
    else:
        return model(*params)


def oneStepVarQR(J, Q):
    Z = torch.matmul(torch.transpose(J.float(), 1, 2), Q)  # Linear extrapolation of the network in many directions
    q, r = torch.qr(Z, some=True)  # QR decomposition of new directions
    s = torch.diag_embed(torch.sign(torch.diagonal(r, dim1=1, dim2=2)))  # extract sign of each leading r value
    return torch.matmul(q, s), torch.diagonal(torch.matmul(s, r), dim1=1,
                                              dim2=2)  # return positive r values and corresponding vectors


def calc_LEs_an(h, wo, learner, k_LE=100000, rec_layer= 'rnn', kappa=10, diff=10, warmup=10, T_ons=1, dt = 0.1):
    device = torch.device('cuda')

    # x_in = Variable(x_in, requires_grad=False).to(device)
    # print("x_in:", x_in.shape)

    h = Variable(h, requires_grad=False).to(device)
    # print("h0:", h0.shape)

    wo = Variable(wo, requires_grad=False).to(device)

    num_layers, batch_size, hidden_size, feed_seq = h.shape
    # _, _, input_size = x_in.shape

    L = num_layers * hidden_size    # the max num of LEs

    k_LE = max(min(L, k_LE), 1)
    Q = torch.reshape(torch.eye(L), (1, L, L)).repeat(batch_size, 1, 1).to(device) # Q = [batch_size, L, L]
    Q = Q[:, :, :k_LE]  # Choose how many exponents to track

    rvals = torch.ones(batch_size, feed_seq, k_LE).to(device)  # storage

    # qvect = torch.zeros(batch_size, feed_seq, L, k_LE) #storage
    t = 0

    Q, _ = torch.qr(Q, some=True) # compute the QR decomposition
    #     print(Q.shape)

    T_pred = math.log2(kappa / diff)
    T_ons = max(1, math.floor(T_pred))
    print('Pred = {}, QR Interval: {}'.format(T_pred, T_ons))

    t_QR = t
    # for xt in x_in.transpose(0,1):
    h = torch.squeeze(torch.squeeze(h,0).transpose(1,2))
    for ht in tqdm(h):
        # print("t: ",t)
        if (t - t_QR) >= T_ons or t == 0 or t == feed_seq:
            QR = True
        else:
            QR = False
        if rec_layer == 'rnn':
            wf = torch.tensor(learner.wf).to(device)
            M = torch.tensor(learner.M).to(device)
            dt = learner.dt
            wf = torch.unsqueeze(wf, 0)
            M = torch.unsqueeze(M, 0)
            J = tl_jac(M, dt, wf, wo, ht)

        if QR:
            Q, r = oneStepVarQR(J, Q)
            t_QR = t
        else:
            Q = torch.matmul(torch.transpose(J, 1, 2), Q)
            r = torch.ones((batch_size, hidden_size))

        rvals[:, t, :] = r
        t = t + 1
    LEs = torch.sum(torch.log2(rvals.detach()), dim=1) / feed_seq
    #     print(torch.log2(rvals.detach()).shape)
    # print("LEs: ", LEs)
    # print("rvals: ", rvals)
    return LEs, rvals  # , qvect


def plot_evolution(rvals, k_LE, model_name='', sample_id=0, title=False, plot_size=(10, 7)):
    plt.figure(figsize=plot_size)
    feed_seq = rvals.shape[1]
    if type(k_LE == int):
        for i in range(k_LE):
            f = plt.plot(
                torch.div(torch.cumsum(torch.log2(rvals[sample_id, :, i]), dim=0), torch.arange(1., feed_seq + 1)))
    else:
        f = plt.plot(torch.div(torch.cumsum(torch.log2(rvals[sample_id, :, i]), dim=0), torch.arange(1., feed_seq + 1)))
    f = plt.xlabel('Iteration #')
    f = plt.ylabel('Lyapunov Exponent')
    if title:
        plt.title('LE Spectrum Evolution for ' + model_name + ', Sample #' + str(sample_id))
    return f


def LE_stats(LE, save_file=False, file_name='LE.p'):
    mean, std = (torch.mean(LE, dim=1), torch.std(LE, dim=1))
    if save_file:
        pkl.dump((mean, std), open(file_name, "wb"))
    return mean, std

def plot_spectrum(LE, model_name, k_LE=100000, plot_size=(10, 7), legend=[]):
    k_LE = max(min(LE.shape[1], k_LE), 1)
    LE_mean, LE_std = LE_stats(LE)
    f = plt.figure(figsize=plot_size)
    x = range(1, k_LE + 1)
    plt.title('Mean LE Spectrum for ' + model_name)
    f = plt.errorbar(x, LE_mean[:k_LE].to(torch.device('cpu')), yerr=LE_std[:k_LE].to(torch.device('cpu')), marker='.',
                     linestyle=' ', markersize=7, elinewidth=2)
    plt.xlabel('Exponent #')

def num_Jac(xt, *states, model, eps=0.01):
    if len(states) > 1:
        h, c = states
    else:
        h = states[0]
    layers, batch_size, hidden_size = h.shape
    L = layers * hidden_size
    h_flat = h.transpose(0, 1).reshape(batch_size, L, 1)
    delta = eps * torch.eye(L).repeat(batch_size, 1, 1)
    hf = h_flat.repeat(1, 1, L) + delta
    hb = h_flat.repeat(1, 1, L) - delta
    del delta
    if len(states) > 1:
        fstates = (hf, c)
        bstates = (hb, c)
    else:
        fstates = hf,
        bstates = hb,
    fwd = model.evolve_hidden(xt, *fstates)
    bwd = model.evolve_hidden(xt, *bstates)
    Jac = (fwd - bwd) / (2 * eps)
    del fwd, bwd, hf, hb, fstates, bstates
    gc.collect()
    return Jac

def tl_jac(M, dt, wf, wo, rt):
    device = get_device(rt)
    _, _, N = M.shape
    M_ = M.clone().detach().to(device)
    wf_ = wf.clone().detach().to(device)
    wo_ = wo.clone().detach().to(device)
    wo_ = torch.unsqueeze(torch.unsqueeze(wo_, 0), 0)

    J = (torch.eye(N, device=device) - torch.diag(torch.squeeze(rt)**2))  @ (M_ * dt + wf_ @ wo_)
    return J

def rnn_jac(wf, M, h, x, bias):
    # if bias:
    #     W, U, b_i, b_h = param_split(params_array, bias)
    # else:
    #     W, U = param_split(params_array, bias)
    bias = False
    device = get_device(h)
    W = wf.clone().detach().to(device)
    U = M.clone().detach().to(device)
    # W = torch.tensor(wf).to(device)
    # U = torch.tensor(M).to(device)

    num_layers, batch_size, hidden_size = h.shape
    input_shape = x.shape[-1]
    h_in = h.transpose(1, 2).detach()
    x_in = [x.squeeze(dim=1).t()]  # input_shape, batch_size)]
    if bias:
        b = [b1 + b2 for (b1, b2) in zip(b_i, b_h)]
    else:
        b = [torch.zeros(W_i.shape[0], ).to(device) for W_i in W]
    J = torch.zeros(batch_size, num_layers * hidden_size, num_layers * hidden_size).to(device)
    y = []
    h_out = []

    for layer in range(num_layers):
        if layer > 0:
            x_l = h_out[layer - 1]
            x_in.append(x_l)

        # print("W[layer]: ", W[layer].shape)
        # print("x_in[layer]: ", x_in[layer].shape)
        y.append((W[layer] @ x_in[layer] + U[layer] @ h_in[layer] + b[layer].repeat(batch_size, 1).t()).t())
        # print(y)
        h_out.append(torch.tanh(y[layer]).t())
        # print("y[layer]: ", y[layer].shape)
        # print("U[layer]: ", U[layer].shape)
        J_h = sech(y[layer]) ** 2 @ U[layer]
        J[:, layer * hidden_size:(layer + 1) * hidden_size, layer * hidden_size:(layer + 1) * hidden_size] = J_h

        if layer > 0:
            J_xt = sech(y[layer]) ** 2 @ W[layer]
            for l in range(layer, 0, -1):
                J[:, layer * hidden_size:(layer + 1) * hidden_size, (l - 1) * hidden_size:l * hidden_size] = J_xt @ J[:,
                                                                                                                    (
                                                                                                                                layer - 1) * hidden_size:(
                                                                                                                                                             layer) * hidden_size,
                                                                                                                    (
                                                                                                                                l - 1) * hidden_size:l * hidden_size]
    return J


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


## Define Math Functions
def get_device(X):
    # print("X: ", X)
    if X.is_cuda:
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def sig(X):
    device = get_device(X)
    return 1 / (1 + torch.exp(-X))


def sigmoid(X):
    device = get_device(X)
    return torch.diag_embed(1 / (1 + torch.exp(-X)))


def sigmoid_p(X):
    device = get_device(X)
    ones = torch.ones_like(X)
    return torch.diag_embed(sig(X) * (ones - sig(X)))


def sech(X):
    device = get_device(X)
    return torch.diag_embed(1 / (torch.cosh(X)))


def tanh(X):
    device = get_device(X)
    return torch.diag_embed(torch.tanh(X))


def LEs(epochs, feed_seq, is_test=True, tl_learner = None):
    # print(dir(tl_learner))

    if is_test:
        # h = tl_learner.testing_outputs[epochs]['hidden_states'][:, :feed_seq]
        h = tl_learner.testing_stats[epochs]['hidden_states'][:, :feed_seq]
        x_in = tl_learner.testing_stats[epochs]['inputs'][:, :feed_seq]
        wo = tl_learner.wo_recording[:,epochs]

        h = torch.unsqueeze(torch.unsqueeze(torch.tensor(h), 0), 0)  # h0 = [num hidden layer, batch size, hidden size, feed_seq]
        # x_in = torch.unsqueeze(torch.transpose(torch.tensor(x_in), 0, 1), 0)  # x_in = [batch_size, feed_seq, input_size]
        wo = torch.tensor(wo)
        # print(h0.size)
        # print(x_in.shape)
        LEs, rvals = calc_LEs_an(h, wo, learner=tl_learner)
        # print(LEs.shape)

        LE_mean, LE_std = LE_stats(LEs)
        LEs = torch.squeeze(LEs, 0).cpu().detach().numpy()
        stats = {'LEs': LEs, 'LE_mean': LE_mean, 'LE_std': LE_std, 'rvals': rvals,
                 'val_loss': tl_learner.testing_stats[epochs]['val_loss']}
        return stats



device = torch.device('cuda')

def main():
    trial = 6
    g = 2.0
    epochs = 15
    # batch_size = 10
    feed_seq = 200
    function_type = '4sine'
    N = 200
    trained_model = True
    test = False
    repeat = 5
    LEs_recording = np.zeros([repeat, N])
    plt.figure()
    for i in range(repeat):
        if trained_model:
            tl_learner = pickle.load(open('../Models/Target_Learning/{}_learner_N_{}_g_{}_trial_{}_trained_e_{}.p'.format
                                          (function_type, N, g, trial, epochs),'rb'))
        else:
            tl_learner = pickle.load(open('../Models/Target_Learning/{}_learner_N_{}_g_{}_trial_{}_untrained_e_{}.p'.format
                                          (function_type, N, g, trial, epochs),'rb'))
        print(dir(tl_learner))

        starting_point = tl_learner.dataloader.signal_length * (i)
        if test:
            h0 = tl_learner.h_testing_recording[:,0]
            x_in = tl_learner.input_testing_recording[:,:feed_seq]
        else:
            h0 = tl_learner.h_training_recording[:, starting_point]
            x_in = tl_learner.input_training_recording[:, starting_point: starting_point + feed_seq]
        print(x_in)
        h0 = torch.unsqueeze(torch.unsqueeze(torch.tensor(h0), 0), 0)  # h0 = [num hidden layer, batch size, hidden size]
        x_in = torch.unsqueeze(torch.transpose(torch.tensor(x_in), 0, 1), 0)  # x_in = [batch_size, feed_seq, input_size]
        print(h0.size)
        print(x_in.shape)
        LEs, rvals = calc_LEs_an(x_in, h0, learner=tl_learner)
        print(LEs.shape)

        LE_mean, LE_std = LE_stats(LEs)
        stats = {'LEs': LEs, 'LE_mean': LE_mean, 'LE_std': LE_std, 'rvals': rvals}
        if trained_model:
            pickle.dump(stats, open('../LE_stats/{}_LE_stats_N_{}_trial_{}_trained_e_{}.p'.format(function_type, N, trial, epochs), 'wb'))

        LEs = torch.squeeze(LEs, 0).cpu().detach().numpy()
        LEs_recording[i, :] = LEs[:]
        x_axis = np.linspace(0, len(LEs), num = len(LEs), endpoint=False)
        # plt.figure()
        plt.scatter(x_axis, LEs)
        plt.xlim(-5, 10)
        plt.ylim(-0 , 1)

        # plt.xlim(-5, 205)
        # plt.ylim(-5, 1)
        if not trained_model:
            plt.title("Before training")
        elif not test:
            plt.title("Training")
        else:
            plt.title("Testing")
    plt.legend(['e1','e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'])
    plt.show()
    print(LEs_recording)


# if __name__ == "__main__":
#     main()