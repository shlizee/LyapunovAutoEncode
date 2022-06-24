from TargetLearning.dataloader import targetLearningDataloader
import TargetLearning.RFORCE_Distribution as R
from TargetLearning.util import *
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.stats as stats
import scipy.linalg as la
import pickle

# Args:
# @ dataloader (targetLearningDataloader): consists of the training and testing targets
# @ N (int): size of the network
# @ g (float) : scaling factor
# @ learn_every (int): updating frequency (default to 2)
# @ train (boolean) : whether or not train the model (default True)
# @ isRFORCE (boolean): whether it is RFORCE. If not, then FORCE (default False)
# @ verbose (boolean): whether output testing after each training epoch
# Attributes:
# @ dataloader, N, g, learn_every, train, isRFORCE: same as Args
# @ dt (float): time interval
# @ p (float): < 1.0, the sparsity of the matrix M
# @ M (2d np.array): [N, N], the recurrent matrix
# @ training_stats, testing_stats (dict)
# @ z (float): readout(output) of the network
# @ P (np.array): [N, N] the reverse matrix
# @ x (np.array): [N, 1] initial state of neurons
# @ wo_recording (np.array): [N, training_epochs] record wo after training for each epoch
# @ wf (np.array): [N, 1] the feedback vector
# @ ti (int):
# @ training_outputs (np.array): [training_epochs * epoch length, 1], the output of network during training
# @ testing_outputs (np.array): [testing_epochs * epoch length, training_epoch], each col is the outputs
#                               of the network during testing
class Learner:
    def __init__(self, dataloader, N, g, learn_every=2, train=True, isRFORCE=False, verbose=False):
        self.dataloader = dataloader
        self.N = N
        self.g = g
        self.learn_every = learn_every
        self.isRFORCE = isRFORCE
        self.train = train
        self.verbose = verbose
        self.dt = self.dataloader.dt
        self.p = 0.1

        if self.isRFORCE:
            self.M, self.per, self.radius, self.theta = R.R_FORCEDistribution(N=self.N, g=self.g)
        else:
            # specify probability distribution
            rvs = stats.norm(loc=0, scale=1).rvs
            # create sparse random matrix with specific probability distribution/random numbers.
            scale = 1 / np.sqrt(self.p * self.N)
            S = sparse.random(self.N, self.N, density=self.p, data_rvs=rvs) * self.g * scale
            self.M = S.toarray()

        self.z = 0.5 * np.random.normal(0, 1, [1, 1])
        alpha = 1
        self.P = np.identity(self.N) * (1 / alpha)
        self.x = 0.5 * np.random.normal(0, 1, [self.N, 1])
        self.wo_recording = np.zeros([self.N, self.dataloader.training_epochs])
        self.wf = np.random.uniform(-1, 1, [self.N, 1])
        self.ti = 0
        self.testing_stats = {}
        self.training_outputs = np.zeros([len(self.dataloader.train_dataset), ])
        self.testing_outputs = np.zeros([len(self.dataloader.test_dataset), self.dataloader.training_epochs])

    # start learning(training). After training for each epoch, testing is conducted.
    # @ verbose (boolean): if True, testing result after each epoch will be display, otherwise only the last one
    #                      will be displayed.
    def learn(self):
        wo = np.zeros([1, self.N])
        for epoch in range(0, self.dataloader.training_epochs):
            wo, error_avg = self.training(wo)
            self.wo_recording[:, epoch] = wo
            if epoch == (self.dataloader.training_epochs - 1):
                self.testing(epoch, wo, display=True)
            elif self.verbose:
                self.testing(epoch, wo, display=True)
            else:
                self.testing(epoch, wo, display=False)

    def training(self, wo):
        starting_point = self.ti
        r = np.tanh(self.x)
        for _ in range(0, self.dataloader.signal_length):
            self.x = (1 - self.dt) * self.x + np.dot(self.M, r * self.dt) + self.wf * (self.z * self.dt)
            r = np.tanh(self.x)
            self.z = np.dot(wo, r)
            self.training_outputs[self.ti] = self.z

            # update wo
            if self.train:
                if self.ti % self.learn_every == 0:
                    Pr = np.dot(self.P, r)
                    k = np.transpose(Pr) / (1 + np.dot(np.transpose(r), Pr))
                    self.P = self.P - np.dot(Pr, k)
                    e = self.z - self.dataloader.train_dataset[self.ti]
                    wo = wo - e * k
            self.ti += 1
        error_total = np.sum(np.absolute(self.dataloader.train_dataset[starting_point: self.ti]
                                         - self.training_outputs[starting_point: self.ti]))
        error_avg = error_total / self.dataloader.signal_length

        return wo, error_avg

    def testing(self, epoch, wo, display=False):
        r = np.tanh(self.x)
        x = self.x
        z = self.z
        testing_length = self.dataloader.testing_epochs * self.dataloader.signal_length
        self.testing_stats[epoch] = {}
        # save the first 1000 hidden states and inputs
        self.testing_stats[epoch]['hidden_states']= np.zeros([x.shape[0], 1000])
        self.testing_stats[epoch]['inputs'] = np.zeros([1, 1000])
        for idx in range(testing_length):
            x = (1 - self.dt) * x + np.dot(self.M, r * self.dt) + self.wf * (z * self.dt)
            r = np.tanh(x)
            z = np.dot(wo, r)
            self.testing_outputs[idx, epoch] = z
            if idx < 1000:
                self.testing_stats[epoch]['hidden_states'][:, idx] = np.squeeze(x)
                self.testing_stats[epoch]['inputs'][:, idx] = np.squeeze(z)
        error_avg = np.sum(
            np.absolute(self.dataloader.test_dataset - self.testing_outputs[:, epoch])) / testing_length
        self.testing_stats[epoch]['val_loss'] = error_avg
        if display:
            print("Testing error: ", error_avg)



def main():
    seed = 12
    npr.seed(seed=seed)
    function_type = 'random_4sine'
    training_epochs = 15
    testing_epochs = 15
    tl_dataloader = targetLearningDataloader(function_type=function_type, training_epochs=training_epochs,
                                             testing_epochs=testing_epochs)
    train = True

    N = 512
    g = 1.4
    tl_learner = Learner(dataloader=tl_dataloader, N=N, g=g, train=train)
    tl_learner.learn()
    if N == 512 and g == 1.4 and seed == 12:
        # val loss = 0.2301757
        pickle.dump(tl_learner, open('trials/FORCE/random_4sine/paperExample/midExample.pickle', 'wb'))
    if N == 512 and g == 1.4 and seed == 11:
        # val loss = 0.5162395
        pickle.dump(tl_learner, open('trials/FORCE/random_4sine/paperExample/badExample.pickle', 'wb'))
    if N == 512 and g == 1.4 and seed == 10:
        # val loss = 0.0315793
        pickle.dump(tl_learner, open('trials/FORCE/random_4sine/paperExample/goodExample.pickle', 'wb'))


def generateFigure():
    tl_learner = pickle.load(open('trials/FORCE/random_4sine/paperExample/goodExample.pickle', 'rb'))
    fig, ax = plt.subplots()
    plt.plot(tl_learner.dataloader.test_dataset[1200 * 6: 1200 * 9], color='gray', linestyle='dashed', linewidth=3)
    plt.plot(tl_learner.testing_outputs[1200 * 6: 1200 * 9:, -1], color='black', linewidth=3)
    ax.axis('off')
    plt.ylim([-1.1, 1.1])
    plt.show()

if __name__ == "__main__":
    # main()
    generateFigure()