import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
from .LSTM_attention import LSTM



class Encoder(nn.Module):
    '''
    action_shape(tuple): action shape for an agent
    '''

    def __init__(self, action_shape, lstm_hidden_size, atten_size):
        super(Encoder, self).__init__()
        self.action_shape = action_shape
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8,
                                     stride=4)  # potential check - in_channels
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=7, stride=1)
        self.lstm = LSTM(512, lstm_hidden_size, atten_size)
        self.relu = nn.ReLU()

    '''
    Input:
        obs(torch.Tensor): observations
            shape: (bsize, time_step, image_width, image_height, image_channel)
        hidden_state(torch.Tensor)
            shape: (bsize, 1, hidden_size)
        cell_state(torch.Tensor)
            shape: (bsize, 1, cell_size)
        lstm_out(torch.Tensor)
            shape: (bsize, available_atten_size, cell_size)
    Output:
        hidden_state(torch.Tensor)
            shape: (bsize, 1, hidden_size)
        cell_state(torch.Tensor)
            shape: (bsize, 1, cell_size)
        lstm_out(torch.Tensor)
            shape: (bsize, available_atten_size, cell_size)
        dqn_out(torch.Tensor)
            shape: (bsize, action_shape[0], action_shape[1], ...)
    '''

    def forward(self, obs, hidden_state, cell_state):
        bsize = obs.shape[0]
        obs = obs / 255
        if len(obs.shape) == 5:
            obs = obs.reshape(bsize * obs.shape[1], obs.shape[4], obs.shape[2], obs.shape[3])
        elif len(obs.shape) == 4:
            obs = obs.reshape(bsize * obs.shape[1], 1, obs.shape[2], obs.shape[3])
        else:
            raise RuntimeError("The observation shape must be (bsize, time_step, width, height, (channel)")
        conv_out = self.conv_layer1(obs)
        conv_out = self.relu(conv_out)
        conv_out = self.conv_layer2(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.conv_layer3(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.conv_layer4(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = conv_out.view(bsize, int(obs.shape[0] / bsize), -1)
        lstm_out, (hidden_state, cell_state) = self.lstm(conv_out, hidden_state, cell_state)
        return lstm_out, (hidden_state, cell_state)

    def generate_action_matrix(self):
        action_matrix = np.zeros(self.action_shape)
        count = 0
        for idx, _ in np.ndenumerate(action_matrix):
            action_matrix[idx] = count
            count += 1
        return action_matrix
