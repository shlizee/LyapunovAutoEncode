import numpy as np
import numpy.random as npr
from numpy import linalg as LA
import matplotlib.pyplot as plt
from TargetLearning.util import *

# Args:
# @ function_type (string): (random) 4sine or 2sine
# @ idling_epoch, training_epochs, testing_epochs (int)
# @ dt (float): time interval
# @ signal_time (float): time length of each epoch
# @ display (boolean): whether or not display the target (both training and testing)
# Attributes:
# @ function_type
# @ idling_epochs, training_epochs, testing_epochs
# @ dt, signal_time
# @ signal_length (int): length of one epoch of signal
# @ idle_simtime, train_simtime, test_simtime (np.array): [0: signal_length * num_epochs] separated by dt
# @ train_dataset, test_dataset (np.array): training targets and testing targets
class targetLearningDataloader:
    def __init__(self, function_type, idling_epochs=1, training_epochs=15,
                 testing_epochs=10, dt=0.1, signal_time=120, displayResult=False):
        self.function_type = function_type
        self.idling_epochs = idling_epochs
        self.training_epochs = training_epochs
        self.testing_epochs = testing_epochs
        self.dt = dt
        self.signal_time = signal_time
        self.signal_length = int(self.signal_time / self.dt)
        self.displayResult = displayResult
        self.idle_simtime, self.train_simtime, self.test_simtime, self.train_dataset, self.test_dataset = self.setup()

    # args:
    # output:
    # @ simtime0, simtime1, simtime2 (np.array): [0: signal_length * num_epochs] separated by dt
    # @ ft, ft2: (np.array): training targets and testing targets
    def setup(self):

        # set up time length for idling, training and testing
        idling_time = self.signal_time * self.idling_epochs
        training_time = self.signal_time * self.training_epochs
        testing_time = self.signal_time * self.testing_epochs

        simtime0 = np.arange(0, idling_time, self.dt)
        simtime1 = np.arange(0, training_time, self.dt)
        simtime2 = np.arange(0, testing_time, self.dt)

        # set up target function
        amp = 1
        freq = 1 / 60
        if "4sine" in self.function_type:
            # generate a 4-sine target functions
            if "random" not in self.function_type:
                # fixed scales and freq
                scale_vec = [1.0, 2.0, 6.0, 3.0]
                freq_vec = [1.0, 2.0, 3.0, 4.0]
            else:
                # random scales and freq
                scale_vec = np.random.randint(0, 10, 4) + 1
                freq_vec = np.random.randint(0, 5, 4) + 1
            ft = (amp / scale_vec[0]) * np.sin(freq_vec[0] * np.pi * freq * simtime1) + \
                 (amp / scale_vec[1]) * np.sin(freq_vec[1] * np.pi * freq * simtime1) + \
                 (amp / scale_vec[2]) * np.sin(freq_vec[2] * np.pi * freq * simtime1) + \
                 (amp / scale_vec[3]) * np.sin(freq_vec[3] * np.pi * freq * simtime1)
            ratio = (np.max(ft) - np.min(ft)) / 2
            transition = (np.max(ft) + np.min(ft)) / 2
            ft = ft / (1 * ratio) - transition

            ft2 = (amp / scale_vec[0]) * np.sin(freq_vec[0] * np.pi * freq * simtime2) + \
                  (amp / scale_vec[1]) * np.sin(freq_vec[1] * np.pi * freq * simtime2) + \
                  (amp / scale_vec[2]) * np.sin(freq_vec[2] * np.pi * freq * simtime2) + \
                  (amp / scale_vec[3]) * np.sin(freq_vec[3] * np.pi * freq * simtime2)
            ft2 = ft2 / (1 * ratio) - transition

        elif self.function_type == "2sine":
            # generationg 2-sine target functions

            ft = (amp / 1.0) * np.sin(1.0 * np.pi * freq * simtime1) + \
                 (amp / 2.0) * np.sin(2.0 * np.pi * freq * simtime1)
            scale = np.max(ft)

            ft = ft / (1 * scale)
            ft2 = (amp / 1.0) * np.sin(1.0 * np.pi * freq * simtime2) + \
                  (amp / 2.0) * np.sin(2.0 * np.pi * freq * simtime2)
            ft2 = ft2 / (1 * scale)
        if self.displayResult:
            display(ft, simtime1, 'Training')
            display(ft2, simtime2, 'Testing')
        return simtime0, simtime1, simtime2, ft, ft2



def main():
    function_type = 'random_4sine'
    tl_dataloader = targetLearningDataloader(function_type=function_type, displayResult=True)


if __name__ == '__main__':
    main()
