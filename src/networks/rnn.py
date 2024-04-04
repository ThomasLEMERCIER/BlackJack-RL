import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out