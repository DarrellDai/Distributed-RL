import numpy as np
import torch.nn as nn
import torch


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, atten_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.atten_size = atten_size
        self.lstm_layer = nn.LSTM(input_size=self.input_size + self.hidden_size, hidden_size=self.hidden_size,
                                  num_layers=1,
                                  batch_first=True)
        self.atten_layer = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                                         nn.ReLU(),
                                         nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                                         nn.ReLU(),
                                         nn.Linear(self.hidden_size * 2, 1))

    '''
    Recursively take all previous states as attention
    '''

    # Input must be tensor
    def forward(self, x, hidden_state, cell_state, out):
        device = x.device
        hidden_state = hidden_state.reshape((hidden_state.shape[1], hidden_state.shape[0]) + hidden_state.shape[2:])
        cell_state = cell_state.reshape((cell_state.shape[1], cell_state.shape[0]) + cell_state.shape[2:])
        for T in range(x.shape[1]):
            # Calculate weighted outputs as attention to make new input by concatenating with the input
            if out.shape[1] > 1:
                score = torch.zeros((out.shape[0], 1, out.shape[1]), requires_grad=False).float()
                for t in range(out.shape[1]):
                    score[:, :, t] = self.atten_layer(torch.concat((out[:, t, :], hidden_state[0]), 1))
                weight = nn.Softmax(2)(score).to(device)
                atten = torch.bmm(weight, out)
                x_new = torch.concat((x[:, T:T + 1, :], atten), 2)
            else:
                x_new = torch.concat((x[:, T:T + 1, :], out), 2)
            self.lstm_layer.flatten_parameters()
            lstm_out = self.lstm_layer(x_new, (hidden_state, cell_state))
            hidden_state = lstm_out[1][0]
            cell_state = lstm_out[1][1]
            single_out = lstm_out[0]
            out = torch.concat((out, single_out), 1)
            start_outs = max(0, out.shape[1] - self.atten_size)
            out = out[:, start_outs:, :]
        hidden_state = hidden_state.reshape((hidden_state.shape[1], hidden_state.shape[0]) + hidden_state.shape[2:])
        cell_state = cell_state.reshape((cell_state.shape[1], cell_state.shape[0]) + cell_state.shape[2:])
        return out, (hidden_state, cell_state)

    def init_hidden_states_and_outputs(self, bsize):
        h = torch.zeros(bsize, 1, self.hidden_size).float()
        c = torch.zeros(bsize, 1, self.hidden_size).float()
        o = torch.zeros(bsize, 1, self.hidden_size).float()
        return h, c, o
