import torch.nn as nn
import torch
from LSTM_attention import LSTM
from torchvision.models import resnet18
from DQN import DQN
import numpy as np

class Network(nn.Module):

    def __init__(self, cnn_out_size, lstm_hidden_size, num_action):
        super(Network, self).__init__()
        self.cnn_out_size=cnn_out_size
        self.resnet = resnet18(num_classes=cnn_out_size)
        self.lstm = LSTM(cnn_out_size, lstm_hidden_size)
        self.dqn = DQN(lstm_hidden_size, num_action)

    def forward(self, x, bsize, time_step, hidden_state, cell_state, lstm_out):
        if len(x.shape)==5:
            x=x.transpose((0,1,4,2,3))
            x = x.reshape(bsize * time_step, x.shape[2], x.shape[3], x.shape[4])
        else:
            x = x.reshape(bsize * time_step, 1, x.shape[2], x.shape[3])
        x=torch.tensor(x,dtype=torch.float32)
        resnet_out = self.resnet.forward(x)
        resnet_out=resnet_out.view(bsize, time_step, -1)
        lstm_out = self.lstm.forward(resnet_out, bsize, time_step, hidden_state, cell_state, lstm_out)

        out = lstm_out[0].squeeze(dim=1)
        h_n = lstm_out[1][0]
        c_n = lstm_out[1][1]

        dqn_out=self.dqn(out, bsize)

        return dqn_out, (h_n, c_n)
