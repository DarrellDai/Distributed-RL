import torch.nn as nn
import torch
from LSTM_attention import LSTM
from torchvision.models import resnet18
from DQN import DQN
import numpy as np


class Network(nn.Module):

    def __init__(self, cnn_out_size, lstm_hidden_size, action_shape):
        super(Network, self).__init__()
        self.action_shape = action_shape
        self.cnn_out_size = cnn_out_size
        self.resnet = resnet18(num_classes=cnn_out_size)
        self.lstm = LSTM(cnn_out_size + len(action_shape), lstm_hidden_size)
        self.dqn = DQN(lstm_hidden_size)

    '''
    obs: observations
        shape: (bsize, time_step, image_width, image_height, image_channel)
    acts: actions
        shape: (bsize, time_step, action_branches)
    '''

    def forward(self, obs, acts, bsize, time_step, hidden_state, cell_state, lstm_out):
        if len(obs.shape) == 5:
            obs = obs.transpose((0, 1, 4, 2, 3))
            obs = obs.reshape(bsize * time_step, obs.shape[2], obs.shape[3], obs.shape[4])
        else:
            obs = obs.reshape(bsize * time_step, 1, obs.shape[2], obs.shape[3])
        obs = torch.tensor(obs, dtype=torch.float32)
        resnet_out = self.resnet.forward(obs)
        resnet_out = resnet_out.view(bsize, time_step, -1)
        obs_acts = torch.concat((resnet_out, acts), -1)
        lstm_out, _, (previous_hidden_state, previous_cell_state), atten = self.lstm.forward(obs_acts, bsize, time_step,
                                                                                             hidden_state, cell_state,
                                                                                             lstm_out)

        # Used to retrieve action index
        action_matrix = torch.zeros(self.action_shape)
        # Output of LSTM for all possible actions
        # shape: (bsize, 1 , action_shape[0], action_shape[1], ..., lstm_hidden_size)
        out_per_action = torch.zeros((bsize, 1) + self.action_shape + hidden_state.shape[-1:])
        # Hidden states of LSTM for all possible actions
        # shape: (1 , bsize, action_shape[0], action_shape[1], ..., lstm_hidden_size)
        hidden_state_per_action = torch.zeros((1, bsize,) + self.action_shape + hidden_state.shape[-1:])
        # Cell states of LSTM for all possible actions
        # shape: (1 , bsize, action_shape[0], action_shape[1], ..., lstm_hidden_size)
        cell_state_per_action = torch.zeros((1, bsize,) + self.action_shape + cell_state.shape[-1:])
        for idx, _ in np.ndenumerate(action_matrix):
            last_acts = torch.tensor(idx).repeat(bsize, 1, 1)
            obs_acts = torch.concat((resnet_out[:, -1:, :], last_acts), -1)
            out_per_action[:, :, idx[0], idx[1], :], (hidden_state_per_action[:, :, idx[0], idx[1], :],
                                                      cell_state_per_action[:, :, idx[0], idx[1],
                                                      :]), _, _ = self.lstm.forward(
                obs_acts, bsize, 1, previous_hidden_state, previous_cell_state, lstm_out)

        # Q-value for all possible actions after all previous obseravtions and actions
        dqn_out = self.dqn(out_per_action).squeeze(-1).squeeze(1)

        return dqn_out, out_per_action, (hidden_state_per_action, cell_state_per_action)

    def generate_action_matrix(self):
        action_matrix = np.zeros(self.action_shape)
        count = 0
        for idx, _ in np.ndenumerate(action_matrix):
            action_matrix[idx] = count
            count += 1
        return action_matrix
