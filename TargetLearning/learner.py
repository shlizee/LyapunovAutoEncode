from dataloader import targetLearningDataloader
import RFORCE_Distribution as R
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.stats as stats
import scipy.linalg as la


class Learner():

    def __init__(self, dataloader, N, g, learn_every = 2, train=True):
        self.dataloader = dataloader
        self.N = N
        self.g = g
        self.dt = self.dataloader.dt
        self.learn_every = learn_every
        self.train = train
        self.p = 0.1

        # specify probability distribution
        rvs = stats.norm(loc=0, scale=1).rvs
        # create sparse random matrix with specific probability distribution/random numbers.
        scale = 1 / np.sqrt(self.p * self.N)
        S = sparse.random(self.N, self.N, density=self.p, data_rvs=rvs) * self.g * scale
        self.M = S.toarray()
        # self.M, self.per, self.radius, self.theta = R.R_FORCEDistribution(N=self.N, g=self.g)

        self.training_stats = {}
        self.testing_stats = {}

        self.z = 0.5 * np.random.normal(0, 1, [1, 1])

    def learn(self):
        alpha = 1
        self.P = np.identity(self.N) * (1 / alpha)
        self.x = 0.5 * np.random.normal(0, 1, [self.N, 1])
        r = np.tanh(self.x)

        self.wo_recording = np.zeros([self.N, self.dataloader.training_epochs + 1])
        wo = np.zeros([1, self.N])
        self.wo_recording[:, 0] = wo
        self.wf = np.random.uniform(-1, 1, [self.N, 1])

        ti = 0
        for epoch in range(0, self.dataloader.training_epochs):
            wo, error_avg, ti = self.training(epoch, wo, ti)
            self.wo_recording[:, epoch + 1] = wo
            testing_loss = self.testing(epoch, wo)
            print(testing_loss)
    def training(self, epoch, wo, ti):
        r = np.tanh(self.x)
        h_training_recording = np.zeros([self.N, self.dataloader.signal_length + 1])
        h_training_recording[:, 0] = r.flatten()
        input_training_recording = np.zeros([1, self.dataloader.signal_length + 1])
        input_training_recording[:, 0] = self.z
        for t in range(0, self.dataloader.signal_length):
            self.x = (1 - self.dt) * self.x + np.dot(self.M, r * self.dt) + self.wf * (self.z * self.dt)
            r = np.tanh(self.x)
            self.z = np.dot(wo, r)
            h_training_recording[:, t + 1] = r.flatten()
            input_training_recording[:, t + 1] = self.z
            if self.train:
                if (ti % self.learn_every == 0):
                    Pr = np.dot(self.P, r)
                    k = np.transpose(Pr) / (1 + np.dot(np.transpose(r), Pr))
                    self.P = self.P - np.dot(Pr, k)
                    e = self.z - self.dataloader.train_dataset[ti]
                    wo = wo - e * k
            ti += 1
        error_avg = np.sum(np.absolute(self.dataloader.train_dataset[ti - self.dataloader.signal_length: ti]
                                       - input_training_recording[:, 1:])) / self.dataloader.signal_length
        # print("Training error: ", error_avg)
        self.training_stats[epoch] = {'hidden_states': h_training_recording, 'inputs': input_training_recording}
        return wo, error_avg, ti

    def testing(self, epoch, wo):
        r = np.tanh(self.x)
        x = self.x
        z = self.z
        ti = 0

        input_testing_recording = np.zeros([1, len(self.dataloader.test_simtime) + 1])
        input_testing_recording[:, ti] = self.z
        h_testing_recording = np.zeros([self.N, len(self.dataloader.test_simtime) + 1])
        h_testing_recording[:, ti] = r.flatten()

        for _ in range(0, self.dataloader.testing_epochs):
            for t in range(0, self.dataloader.signal_length):
                x = (1 - self.dt) * x + np.dot(self.M, r * self.dt) + self.wf * (z * self.dt)
                r = np.tanh(x)
                z = np.dot(wo, r)

                h_testing_recording[:, ti + 1] = r.flatten()
                input_testing_recording[:, ti + 1] = z
                ti += 1
        error_avg = np.sum(np.absolute(self.dataloader.test_dataset - input_testing_recording[:, 1:])) / len(self.dataloader.test_simtime)
        # print("Testing error: ", error_avg)
        self.testing_stats[epoch] = {'hidden_states': h_testing_recording, 'inputs': input_testing_recording, 'val_loss': error_avg}
        return error_avg
def main():
    npr.seed(seed=0)
    function_type = '4sine'
    epochs = 15
    tl_dataloader = targetLearningDataloader(function_type=function_type)
    train = True

    N = 256
    g = 1.5
    tl_learner = Learner(dataloader=tl_dataloader, N=N, g=g, train=train)
    # eig_values, _ = la.eig(tf_learner.M)
    # print(eig_values.shape)
    # eig_values_real = np.real(eig_values)
    # eig_values_imag = np.imag(eig_values)
    # plt.figure()
    # plt.scatter(eig_values_real, eig_values_imag)
    # theta = np.linspace(0, 2 * np.pi, num=200)
    # plt.scatter(np.cos(theta) * g, np.sin(theta) * g)
    # plt.show()
    tl_learner.learn()

    # plt.figure()
    # plt.scatter(tl_dataloader.train_simtime, tl_dataloader.train_dataset)
    # plt.scatter(tl_dataloader.train_simtime, zt)
    # plt.title("Training")
    # plt.show()
    #
    # plt.figure()
    # plt.scatter(tl_dataloader.test_simtime, tl_dataloader.test_dataset)
    # plt.scatter(tl_dataloader.test_simtime, zpt)
    # plt.title("Testing")
    # plt.show()


if __name__ == "__main__":
    main()
