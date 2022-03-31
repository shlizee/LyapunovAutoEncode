from dataloader import targetLearningDataloader
import numpy.random as npr
import matplotlib.pyplot as plt
from learner import Learner
import pickle
import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
import scipy.linalg as la
from plotting import plotting
import tl_lyapunov as ly
import time

def main():
    trials = {}
    for idx in range(0, 1):
        a = time.perf_counter()
        g = 1.5 + 0.1 * idx
        print('g = ',g)
        for seed in range(0, 50):
            print('seed=' ,seed)
            npr.seed(seed=seed)
            function_type = '4sine'
            training_epochs = 15
            testing_epochs = 10
            dt = 0.1
            N = 512

            feed_seq = 200
            train = True
            tl_dataloader = targetLearningDataloader(function_type=function_type, training_epochs=training_epochs,
                                                     testing_epochs=testing_epochs, dt=dt)

            tl_learner = Learner(dataloader=tl_dataloader, N=N, g=g, train=train)

            tl_learner.learn()
            LEs_stats = {}
            for i in range(0, training_epochs):

                LEs_stats[i] = ly.LEs(epochs=i, feed_seq=feed_seq, is_test=True, tl_learner = tl_learner)

            trials[seed] = {"seed": seed, "LEs_stats": LEs_stats}

        pickle.dump(trials, open('../trials/{}_learner_N_{}_g_{:0.1f}.p'.format
                         (function_type, N, g), 'wb'))

        b = time.perf_counter()
        print("elapse time: ", b - a)
if __name__ == "__main__":
    main()