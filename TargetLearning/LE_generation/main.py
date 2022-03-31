from TargetLearning.dataloader import targetLearningDataloader
import numpy.random as npr
import matplotlib.pyplot as plt
import os
from TargetLearning.learner import Learner
import pickle
import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
import scipy.linalg as la
from plotting import plotting
import tl_lyapunov as ly
import time

def plotting_target(tl_learner):
    val_loss = tl_learner.testing_stats[training_epochs - 1]['val_loss']
    # Low Error
    # if val_loss < 0.05:

    # High Error
    if val_loss > 0.5:
        # Mid Error
        # if val_loss> 0.15 and val_loss < 0.25:
        plt.figure()
        start_point = 1200 * 7
        end_point = 1200 * 9
        xrange = range(tl_learner.testing_stats[training_epochs - 1]['inputs'].size)

        plt.plot(xrange[start_point:end_point], tl_learner.testing_stats[14]['inputs'][0, start_point:end_point],
                 'g', linewidth=3)

        plt.plot(xrange[start_point:end_point], tl_learner.dataloader.train_dataset[start_point:end_point],
                 'r', linewidth=3)

        plt.title("val loss: {}, seed: {}".format(val_loss, seed))
        plt.show()
        print("this")


def main():
    function_type = 'random_4sine'
    training_epochs = 15
    testing_epochs = 15
    dt = 0.1
    N = 512
    feed_seq = 200
    train = True
    isRFORCE = False
    if isRFORCE:
        distribution = 'RFORCE'
    else:
        distribution = "FORCE"
    g_s = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    g_s = [1.4]
    for trial in range(10, 11):
        trials = {}
        for i, g in enumerate(g_s):
            a = time.perf_counter()
            for seed in range(10, 13):
                print('g = {}, trial = {}; seed={}'.format(g, trial, seed))
                npr.seed(seed=seed)

                tl_dataloader = targetLearningDataloader(function_type=function_type, training_epochs=training_epochs,
                                                         testing_epochs=testing_epochs, dt=dt)
                tl_learner = Learner(dataloader=tl_dataloader, N=N, g=g, train=train, isRFORCE=isRFORCE)

                tl_learner.learn()

                LEs_stats = {}
                for i in range(0, training_epochs):

                    LEs_stats[i] = ly.LEs(epochs=i, feed_seq=feed_seq, is_test=True, tl_learner = tl_learner)

                trials[seed] = {"seed": seed, "LEs_stats": LEs_stats, "wo": tl_learner.wo_recording[:, -1]}
                saved_path = f'./../../dataset/trials/{distribution}/{function_type}/N_{N}/g_{g}'
                if not os.path.exists(saved_path):
                    os.makedirs(saved_path)
                pickle.dump(trials, open(f'{saved_path}/{function_type}_learner_N_{N}_g_{g:0.1f}_trial_{trial}.p', 'wb'))

            b = time.perf_counter()
            print("elapse time: ", b - a)
if __name__ == "__main__":
    main()