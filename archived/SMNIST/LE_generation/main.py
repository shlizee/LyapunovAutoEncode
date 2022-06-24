import os.path
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataloader import MNIST_dataloader
from model import LSTM, GRU
from tl_lyapunov import calc_LEs_an
import pickle
import numpy as np
def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = "lstm"
    # Hyper-parameters
    sequence_length = 28
    input_size = 28

    num_layers = 1
    num_classes = 10
    batch_size = 100
    num_epochs = 10
    learning_rate = 0.01
    num_trials = 100
    a_s = [2]
    trials = {}

    # just for testing
    num_trials = 1
    num_epochs = 5
    a_s = np.random.uniform(0.1, 2, [2])
    for a in a_s:
        for num_trial in range(num_trials):
            print("a: ", a, "num_trial: ", num_trial)
            hidden_size = 8
            trial = {}
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

            for epoch in range(num_epochs):
                model.train()
                for i, (images, labels) in enumerate(train_dataloader.dataloader):
                    images = images.reshape(-1, sequence_length, input_size).to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs, hts = model(images)
                    loss = criterion(outputs, labels)
                    # print(LEs, rvals)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (i + 1) % 300 == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


                # Test the model
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for i, (images, labels) in enumerate(test_dataloader.dataloader):
                        images = images.reshape(-1, sequence_length, input_size).to(device)
                        labels = labels.to(device)
                        outputs, _ = model(images)

                        # calculate LEs
                        # h = torch.zeros(model.num_layers, images.size(0), model.hidden_size).to(model.device)
                        # c = torch.zeros(model.num_layers, images.size(0), model.hidden_size).to(model.device)
                        # params = (images, (h, c))
                        # if i == 0:
                        #     LEs, rvals = calc_LEs_an(*params, model=model)

                        loss = criterion(outputs, labels)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    if epoch == (num_epochs - 1):
                        print('Epoch [{}/{}] Loss: {}, Test Accuracy: {} %'.format(epoch + 1, num_epochs, loss, 100 * correct / total))
                trial[epoch] = {'model': model, 'accuracy': 100 * correct / total,
                                'loss': loss}
            trials[num_trial] = trial
        saved_path = f'../../../dataset/trials/{model_type}/models/'
        pickle.dump(trials, open(f'{saved_path}/lstm_{hidden_size}_trials_0.pickle', 'wb'))


if __name__ == "__main__":
    main()



