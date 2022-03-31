import pickle
from CharRNN_ex.AS_RNN import *

device = torch.device('cuda')
def main():
    trial = 1
    epochs = 15
    batch_size = 10
    h = torch.zeros(1, batch_size, 500).to(device)
    function_type = '4sine'
    N = 200
    train = True
    if train:
        tl_learner = pickle.load(open('../Models/Target_Learning/{}_learner_N_{}_trial_{}_trained_e_{}.p'.format
                                      (function_type, N, trial, epochs),'rb'))
    else:
        tl_learner = pickle.load(open('../Models/Target_Learning/{}_learner_N_{}_trial_{}_untrained_e_{}.p'.format
                                      (function_type, N, trial, epochs),'rb'))


    print(dir(tl_learner))
# LEs, rvals = lyap.calc_LEs_an(le_input[i], h, model = model, k_LE = 10000, rec_layer = fcon.model.model_type,
#                               warmup = self.warmup, T_ons = self.T_ONS)
# LE_mean, LE_std = lyap.LE_stats(LEs)
# model.lyapunov = False

# return (LE_mean, LE_std), rvals
if __name__ == "__main__":
    main()
