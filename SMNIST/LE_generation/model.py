import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Recurrent neural network (many-to-one)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, a, device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.a = a
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                # nn.init.xavier_normal(param)
                nn.init.uniform_(param, -self.a, self.a)


        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, (ht, ct) = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out, (ht, ct)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, a=0.1, device='cpu'):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.a = a
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                # nn.init.xavier_normal(param)
                nn.init.uniform_(param, -self.a, self.a)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
    def forward(self, x, h=None):
        if h is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        else:
            h0 = h
        # Forward propagate GRU
        out, ht = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out, ht

def main():
    input_size = 10
    hidden_size = 20
    num_layers = 1
    num_classes = 10
    a = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 5
    time_seq = 3
    inputs = torch.randn(batch_size, time_seq, input_size)


    inputs = inputs.to(device)
    model = GRU(input_size, hidden_size, num_layers, num_classes, a, device)
    model.to(device)
    # for name, param in model.gru.named_parameters():
    #     print(name)

    outputs, ht = model(inputs)
    print(outputs.shape, ht.shape)
# if __name__ == "__main__":
    # main()

