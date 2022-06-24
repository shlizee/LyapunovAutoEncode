import numpy as np
import numpy.random as npr
from numpy import linalg as LA
import matplotlib.pyplot as plt


class targetLearningDataloader():
    def __init__(self, function_type, idling_epoch = 1, training_epochs = 15,
                 testing_epochs = 10, dt = 0.1):
        self.function_type = function_type
        self.idling_epoch = idling_epoch
        self.training_epochs = training_epochs
        self.testing_epochs = testing_epochs
        self.dt = dt
        self.signal_time = 120
        self.signal_length = int(self.signal_time / self.dt)
        if function_type == "4sine":
            # generationg target functions
            amp = 1
            freq = 1 / 60

            idling_time = self.signal_time * self.idling_epoch
            training_time = self.signal_time * self.training_epochs
            testing_time = self.signal_time * self.testing_epochs

            simtime0 = np.arange(0, idling_time, self.dt)
            simtime1 = np.arange(0, training_time, self.dt)
            simtime2 = np.arange(0, testing_time, self.dt)

            ft = (amp / 1.0) * np.sin(1.0 * np.pi * freq * simtime1) + \
                 (amp / 2.0) * np.sin(2.0 * np.pi * freq * simtime1) + \
                 (amp / 6.0) * np.sin(3.0 * np.pi * freq * simtime1) + \
                 (amp / 3.0) * np.sin(4.0 * np.pi * freq * simtime1)
            scale = np.max(ft)

            ft = ft / (1 * scale)
            ft2 = (amp / 1.0) * np.sin(1.0 * np.pi * freq * simtime2) + \
                  (amp / 2.0) * np.sin(2.0 * np.pi * freq * simtime2) + \
                  (amp / 6.0) * np.sin(3.0 * np.pi * freq * simtime2) + \
                  (amp / 3.0) * np.sin(4.0 * np.pi * freq * simtime2)
            ft2 = ft2 / (1 * scale)
        self.idle_simtime = simtime0
        self.train_simtime = simtime1
        self.test_simtime = simtime2
        self.train_dataset = ft
        self.test_dataset = ft2


def main():
    function_type = '4sine'
    epochs = 12
    tl_dataloader = targetLearningDataloader(function_type=function_type)
    # print(tl_dataloader.train_dataset)
    plt.figure()
    plt.scatter(tl_dataloader.train_simtime, tl_dataloader.train_dataset)
    plt.title("Training")
    plt.show()

    plt.figure()
    plt.scatter(tl_dataloader.test_simtime, tl_dataloader.test_dataset)
    plt.title("Testing")
    plt.show()

if __name__ == '__main__':
    main()

# plt.figure()
# [D,V] = LA.eig(M)
# plt.scatter(np.real(D),np.imag(D))
# plt.show()
#



