import numpy as np
import torch.nn as nn
import torch


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, atten_size):
        super(LSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.device=x.get_device()
        hidden_state = hidden_state.reshape((hidden_state.shape[1], hidden_state.shape[0]) + hidden_state.shape[2:])
        cell_state = cell_state.reshape((cell_state.shape[1], cell_state.shape[0]) + cell_state.shape[2:])
        x = x.reshape(x.shape[0], -1, self.input_size).float()
        for T in range(x.shape[1]):
            # Calculate weighted outputs as attention to make new input by concatenating with the input
            if out.shape[1] > 1:
                score = torch.zeros((out.shape[0], 1, out.shape[1]), requires_grad=False).float()
                for t in range(out.shape[1]):
                    score[:, :, t] = self.atten_layer(torch.concat((out[:, t, :], hidden_state[0]), 1))
                weight = nn.Softmax(1)(score).to(self.device)
                atten = torch.bmm(weight, out)
                x_new = torch.concat((x[:, T:T + 1, :], atten), 2)
            else:
                x_new = torch.concat((x[:, 0:1, :], out), 2)
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

    # '''
    # Treat time_step number of states as encoder, and these state won't be calculated with attention
    # '''
    # def forward(self, x, bsize, time_step, hidden_state, cell_state):
    #     # time_step: number of time_step before an action, so time_step number of obs are needed as input
    #     outs=torch.zeros(bsize,time_step-1,self.hidden_size)
    #     x = x.view(bsize,time_step,self.input_size)
    #
    #     for t in range(time_step-1):
    #         lstm_out = self.lstm_layer(x[:,t:t+1,:], (hidden_state, cell_state))
    #         outs[:,t,:]=lstm_out[0]
    #         hidden_state=lstm_out[1][0]
    #         cell_state=lstm_out[1][1]
    #     score = np.zeros(bsize,time_step-1)
    #     for t in range(time_step-1):
    #         score[:,t] = self.atten_layer(torch.concat(hidden_state, outs[:,t,:]),2)
    #     weight = nn.Softmax(1)(score)
    #     atten = torch.bmm(weight,outs)
    #     x_new=torch.concat((x[:,time_step-1:time_step,:],atten),2)
    #     lstm_out = self.lstm_layer(x_new, (hidden_state, cell_state))
    #     hidden_state = lstm_out[1][0]
    #     cell_state = lstm_out[1][1]
    #     out=lstm_out[0]
    #     return out, (hidden_state, cell_state)
    def init_hidden_states_and_outputs(self, bsize):
        h = torch.zeros(bsize, 1, self.hidden_size).float().to(self.device)
        c = torch.zeros(bsize, 1, self.hidden_size).float().to(self.device)
        o = torch.zeros(bsize, 1, self.hidden_size).float().to(self.device)
        return h, c, o
