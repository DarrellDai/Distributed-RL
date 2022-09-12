import numpy as np
import torch.nn as nn
import torch


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.hidden_size = hidden_size
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

    def forward(self, x, bsize, time_step, hidden_state, cell_state, out):
        # time_step: number of time_step before an action, so time_step number of obs are needed as input
        outs = out
        x = x.view(bsize, time_step, self.input_size)
        x_new = torch.concat((x[:, 0:1, :], outs), 2)
        atten = outs
        for T in range(time_step):
            previous_hidden_state = hidden_state
            previous_cell_state = cell_state
            # Calculate weighted outputs as attention to make new input by concatenating with the input
            if T > 0:
                score = torch.zeros((outs.shape[0], 1, outs.shape[1]), requires_grad=False)
                for t in range(T + 1):
                    score[:, :, t] = self.atten_layer(torch.concat((outs[:, t, :], hidden_state[0]), 1))
                weight = nn.Softmax(1)(score)
                atten = torch.bmm(weight, outs)
                x_new = torch.concat((x[:, T:T + 1, :], atten), 2)
            lstm_out = self.lstm_layer(x_new, (hidden_state, cell_state))
            hidden_state = lstm_out[1][0]
            cell_state = lstm_out[1][1]
            out = lstm_out[0]
            outs = torch.concat((outs, out), 1)
        return out, (hidden_state, cell_state), (previous_hidden_state, previous_cell_state), atten

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
        h = torch.zeros(1, bsize, self.hidden_size).float().to(self.device)
        c = torch.zeros(1, bsize, self.hidden_size).float().to(self.device)
        o = torch.zeros(bsize, 1, self.hidden_size).float().to(self.device)
        return h, c, o
