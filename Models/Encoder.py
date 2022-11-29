import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
from .LSTM_attention import LSTM



class Encoder(nn.Module):
    '''
    action_shape(tuple): action shape for an agent
    '''

    def __init__(self, cnn_out_size, action_shape, lstm_hidden_size, atten_size):
        super(Encoder, self).__init__()
        self.action_shape = action_shape
        self.cnn_out_size = cnn_out_size
        self.resnet = resnet18(num_classes=cnn_out_size)
        self.lstm = LSTM(cnn_out_size, lstm_hidden_size, atten_size)

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
        resnet_out = self.resnet(obs)
        resnet_out = resnet_out.view(bsize, int(obs.shape[0] / bsize), -1)
        lstm_out, (hidden_state, cell_state) = self.lstm(resnet_out, hidden_state, cell_state)
        return lstm_out, (hidden_state, cell_state)
        # # todo: original code here mignt not cosider the atten_size in dqn_out
        # if self.method == "DQN" or self.method == "BC":
        #     network_out = self.fc(lstm_out[:, -1, :])
        #     network_out = network_out.view((bsize,) + self.action_shape)
        #     return lstm_out, (hidden_state, cell_state), network_out
        # elif self.method == "PPO":
        #     act_prob = self.fc[0](lstm_out[:, -1, :])
        #     act_prob = act_prob.view((bsize,) + self.action_shape)
        #     value = self.fc[1](lstm_out[:, -1, :])
        #     return lstm_out, (hidden_state, cell_state), (act_prob, value)

    def generate_action_matrix(self):
        action_matrix = np.zeros(self.action_shape)
        count = 0
        for idx, _ in np.ndenumerate(action_matrix):
            action_matrix[idx] = count
            count += 1
        return action_matrix
