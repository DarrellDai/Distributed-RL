import torch.nn as nn
import torch
from LSTM_attention import LSTM
from torchvision.models import resnet18
from DQN import DQN
import numpy as np


class Network(nn.Module):
    '''
    action_shape(tuple): action shape for an agent
    '''

    def __init__(self, cnn_out_size, lstm_hidden_size, atten_size, action_shape):
        super(Network, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_shape = action_shape
        self.cnn_out_size = cnn_out_size
        self.resnet = resnet18(num_classes=cnn_out_size).to(self.device)
        self.lstm = LSTM(cnn_out_size + len(action_shape), lstm_hidden_size, atten_size).to(self.device)
        self.dqn = DQN(lstm_hidden_size).to(self.device)

    '''
    obs(torch.Tensor): observations
        shape: (bsize, time_step, image_width, image_height, image_channel)
    act(torch.Tensor): actions
        shape: (bsize, time_step, action_branches)
    hidden_state(torch.Tensor)
    cell_state(torch.Tensor)
    lstm_out(torch.Tensor)
    '''

    def forward(self, obs, act, bsize, hidden_state, cell_state, lstm_out):
        obs = obs.float().to(self.device)
        act = act.float().to(self.device)
        if len(obs.shape) == 5:
            obs = obs.reshape(bsize * obs.shape[1], obs.shape[4], obs.shape[2], obs.shape[3])
        elif len(obs.shape) == 4:
            obs = obs.reshape(bsize * obs.shape[1], 1, obs.shape[2], obs.shape[3])
        else:
            raise RuntimeError
        resnet_out = self.resnet.forward(obs)
        resnet_out = resnet_out.view(bsize, int(obs.shape[0] / bsize), -1)
        if act.shape[1] != 0:
            if resnet_out.shape[1] > act.shape[1]:
                obs_act = torch.concat((resnet_out[:, :-1, :], act), -1)
            else:
                obs_act = torch.concat((resnet_out, act), -1)
            # Forward until in the last one the batch, so the attention is of the last one.
            # previous_hidden_state and previous_cell_state can be used to calculate the output for last obs and act.
            # The reason of doing this is because, unlike previous ones, the last one is not (obs,act) pair but only obs
            lstm_out, (hidden_state, cell_state) = self.lstm.forward(obs_act, bsize, hidden_state, cell_state,
                                                                     lstm_out)

        if obs.shape[1] > act.shape[1]:
            # Used to retrieve action index
            action_matrix = np.zeros(self.action_shape)
            # Output of LSTM for all possible actions
            # shape: (bsize, 1 , action_shape[0], action_shape[1], ..., lstm_hidden_size)
            out_per_action = torch.zeros((bsize, 1) + self.action_shape + hidden_state.shape[-1:]).float().to(
                self.device)
            # Hidden states of LSTM for all possible actions
            # shape: (1 , bsize, action_shape[0], action_shape[1], ..., lstm_hidden_size)
            hidden_state_per_action = torch.zeros((1, bsize,) + self.action_shape + hidden_state.shape[-1:]).float().to(
                self.device)
            # Cell states of LSTM for all possible actions
            # shape: (1 , bsize, action_shape[0], action_shape[1], ..., lstm_hidden_size)
            cell_state_per_action = torch.zeros((1, bsize,) + self.action_shape + cell_state.shape[-1:]).float().to(
                self.device)
            for idx, _ in np.ndenumerate(action_matrix):
                new_acts = torch.tensor(idx).repeat(bsize, 1, 1).float().to(self.device)
                obs_act = torch.concat((resnet_out[:, -1:, :], new_acts), -1)
                lstm_out_all_act, (hidden_state_per_action[:, :, idx[0], idx[1], :],
                                   cell_state_per_action[:, :, idx[0], idx[1], :]) = self.lstm.forward(obs_act, bsize,
                                                                                                       hidden_state,
                                                                                                       cell_state,
                                                                                                       lstm_out)
                out_per_action[:, :, idx[0], idx[1], :] = lstm_out_all_act[:, -1:, :]
            # Q-value for all possible actions after all previous obseravtions and actions
            dqn_out = self.dqn(out_per_action).squeeze(-1).squeeze(1).float()
            return lstm_out, (hidden_state, cell_state), dqn_out, out_per_action, (
                hidden_state_per_action, cell_state_per_action)
        else:
            out_per_action = torch.zeros((bsize, 1) + (1, 1) + hidden_state.shape[-1:]).float().to(
                self.device)
            out_per_action[:, :, 0, 0, :] = lstm_out[:, -1:, :]
            dqn_out = self.dqn(out_per_action).squeeze(-1).squeeze(1).float()
            return lstm_out, (hidden_state, cell_state), dqn_out, None, (None, None)

    def generate_action_matrix(self):
        action_matrix = np.zeros(self.action_shape)
        count = 0
        for idx, _ in np.ndenumerate(action_matrix):
            action_matrix[idx] = count
            count += 1
        return action_matrix