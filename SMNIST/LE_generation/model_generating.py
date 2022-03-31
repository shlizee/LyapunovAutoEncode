'''
This code generates model for later LE calculation
'''
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataloader import MNIST_dataloader
from model import LSTM, GRU
from tl_lyapunov import calc_LEs_an
import pickle
import copy
import random

def main(trial_num):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = "lstm"

    # Hyper-parameters
    sequence_length = 28
    input_size = 28
    num_layers = 1
    hidden_size = 128
    num_classes = 10
    batch_size = 100
    num_epochs = 20
    learning_rate = 0.01
    num_trials = 100
    a_range = [1.0, 3.0]
    # a_s = [1.5, 2.0, 2.2, 2.5, 2.7, 3.0]

    # just for testing
    # num_trials = 1
    # num_epochs = 20
    # a_s = [1.0]

    # for a in a_s:
    trials = {}
    for num_trial in range(num_trials):
        a = random.random() * (a_range[1] - a_range[0]) + a_range[0]
        print('trial Num: ', trial_num,  "a: ", a, "num_trial: ", num_trial)
        trial = {}
        trial['a'] = a
        # define model
        if model_type == 'lstm':
            model = LSTM(input_size, hidden_size, num_layers, num_classes, a, device).to(device)
        elif model_type == 'gru':
            model = GRU(input_size, hidden_size, num_layers, num_classes, a, device).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_dataloader = MNIST_dataloader(batch_size, train=True)
        test_dataloader = MNIST_dataloader(batch_size, train=False)
        # Train the model
        total_step = len(train_dataloader.dataloader)

        total = 0
        total_loss = 0
        for epoch in range(num_epochs):
            model.train()
            for i, (images, labels) in enumerate(train_dataloader.dataloader):
                images = images.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs, hts = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss * labels.size(0)
                total += labels.size(0)
                # print(LEs, rvals)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if (i + 1) % 300 == 0:
                #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                #           .format(epoch + 1, num_epochs, i + 1, total_step, total_loss / total))

            # for i, (name, param) in enumerate(model.named_parameters()):
            #     if i == 3:
            #         print(name, param)
            # Test the model
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                total_loss = 0
                for i, (images, labels) in enumerate(test_dataloader.dataloader):
                    images = images.reshape(-1, sequence_length, input_size).to(device)
                    labels = labels.to(device)
                    outputs, _ = model(images)

                    # h = torch.zeros(model.num_layers, images.size(0), model.hidden_size).to(model.device)
                    # c = torch.zeros(model.num_layers, images.size(0), model.hidden_size).to(model.device)
                    # params = (images, (h, c))
                    # if i == 0:
                    #     LEs, rvals = calc_LEs_an(*params, model=model)

                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    total_loss += loss * labels.size(0)
                if epoch == (num_epochs - 1):
                    print('Epoch [{}/{}] Loss: {}, Test Accuracy: {} %'.format(epoch + 1, num_epochs, total_loss / total, 100 * correct / total))
            saved_model = copy.deepcopy(model)
            trial[epoch] = {"model": saved_model, "accuracy": 100 * correct / total, "loss": total_loss / total}
            del saved_model
        trials[num_trial] = trial
        pickle.dump(trials, open('trials/{}/models/{}_{}_trials_{}.pickle'.format(model_type, model_type, hidden_size, trial_num), 'wb'))

if __name__ == "__main__":
    for trial_num in range(0,1):
        main(trial_num)



