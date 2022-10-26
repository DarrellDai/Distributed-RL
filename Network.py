import torch.nn as nn
import torch
from LSTM_attention import LSTM
from torchvision.models import resnet18, ResNet18_Weights
from DQN import DQN
from ActNet import ActNet
import numpy as np


class Network(nn.Module):
    '''
    action_shape(tuple): action shape for an agent
    '''

    def __init__(self, cnn_out_size, action_space_shape, action_out_size, lstm_hidden_size, atten_size):
        super(Network, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_shape = action_space_shape
        self.cnn_out_size = cnn_out_size
        self.action_out_size = action_out_size
        self.resnet = resnet18(num_classes=cnn_out_size)
        self.lstm = LSTM(cnn_out_size + action_out_size, lstm_hidden_size, atten_size)
        self.dqn = DQN(lstm_hidden_size)
        self.actnet = ActNet(len(action_space_shape), action_out_size)

    '''
    Input:
        obs(torch.Tensor): observations
            shape: (bsize, time_step, image_width, image_height, image_channel)
        act(torch.Tensor): actions
            shape: (bsize, time_step, action_branches)
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
        out_per_action(torch.Tensor)
            shape: (bsize, 1, action_shape[0], action_shape[1], ..., out_size)
        hidden_state_per_action(torch.Tensor)
            shape: (bsize, 1, action_shape[0], action_shape[1], ..., hidden_size)
        cell_state_per_action(torch.Tensor)
            shape: (bsize, 1, action_shape[0], action_shape[1], ..., cell_size)
        dqn_out(torch.Tensor)
            shape: (bsize, action_shape[0], action_shape[1], ...)
    '''

    def forward(self, obs, act, hidden_state, cell_state, lstm_out):
        self.device=obs.get_device()
        bsize = obs.shape[0]
        obs = obs.float()
        act = act.float()
        obs_len = obs.shape[1]
        act_len = act.shape[1]
        if len(obs.shape) == 5:
            obs = obs.reshape(bsize * obs.shape[1], obs.shape[4], obs.shape[2], obs.shape[3])
        elif len(obs.shape) == 4:
            obs = obs.reshape(bsize * obs.shape[1], 1, obs.shape[2], obs.shape[3])
        else:
            raise RuntimeError("The observation shape must be (bsize, time_step, width, height, (channel)")
        resnet_out = self.resnet(obs)
        resnet_out = resnet_out.view(bsize, int(obs.shape[0] / bsize), -1)
        act_out = self.actnet(act)
        # obs_act: (bsize, time_step, cnn_out_size + act_out_size)
        if act.shape[1] != 0:
            if resnet_out.shape[1] > act.shape[1]:
                obs_act = torch.concat((resnet_out[:, :-1, :], act_out), -1)
            else:
                obs_act = torch.concat((resnet_out, act_out), -1)
            # Forward until in the last one the batch, so the attention is of the last one.
            # previous_hidden_state and previous_cell_state can be used to calculate the output for last obs and act.
            # The reason of doing this is because, unlike previous ones, the last one is not (obs,act) pair but only obs
            lstm_out, (hidden_state, cell_state) = self.lstm(obs_act, hidden_state, cell_state,
                                                             lstm_out)

        if obs_len > act_len:
            # Used to retrieve action index
            action_matrix = np.zeros(self.action_shape)
            # out_per_action: Output of LSTM for all possible actions
            # shape: (bsize, 1 , action_shape[0], action_shape[1], ..., lstm_hidden_size)
            out_per_action = torch.zeros((bsize, 1) + self.action_shape + hidden_state.shape[-1:]).float().to(
                self.device)
            # hidden_state_per_action: Hidden states of LSTM for all possible actions
            # shape: (1 , bsize, action_shape[0], action_shape[1], ..., lstm_hidden_size)
            hidden_state_per_action = torch.zeros((bsize, 1) + self.action_shape + hidden_state.shape[-1:]).float().to(
                self.device)
            # cell_state_per_action: Cell states of LSTM for all possible action
            # shape: (1 , bsize, action_shape[0], action_shape[1], ..., lstm_hidden_size)
            cell_state_per_action = torch.zeros((bsize, 1) + self.action_shape + cell_state.shape[-1:]).float().to(
                self.device)
            for idx, _ in np.ndenumerate(action_matrix):
                proposed_acts = torch.tensor(idx).repeat(bsize, 1, 1).float().to(self.device)
                proposed_act_out = self.actnet(proposed_acts)
                obs_act = torch.concat((resnet_out[:, -1:, :], proposed_act_out), -1)
                lstm_out_all_act, (hidden_state_per_action[:, :, idx[0], idx[1], :],
                                   cell_state_per_action[:, :, idx[0], idx[1], :]) = self.lstm(obs_act,
                                                                                               hidden_state,
                                                                                               cell_state,
                                                                                               lstm_out)
                out_per_action[:, :, idx[0], idx[1], :] = lstm_out_all_act[:, -1:, :]
            # Q-value for all possible actions after all previous observations and actions
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
