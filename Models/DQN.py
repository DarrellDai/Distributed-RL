import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import numpy as np
import torch
import torch.nn as nn
from .Encoder import Encoder
from utils import preprocess_data_from_batch, extract_input_per_episode, sync_grads
from copy import deepcopy

class DQNNet(nn.Module):

    def __init__(self, cnn_out_size, lstm_hidden_size,
                               atten_size, action_shape):
        super(DQNNet, self).__init__()
        self.action_shape=tuple(action_shape)
        self.encoder = Encoder(cnn_out_size, self.action_shape, lstm_hidden_size,
                               atten_size)
        self.adv = nn.Linear(lstm_hidden_size, np.prod(np.array(self.action_shape)))
        self.val = nn.Linear(lstm_hidden_size, 1)

    def forward(self, obs, hidden_state, cell_state):
        bsize = obs.shape[0]
        lstm_out, (hidden_state, cell_state) = self.encoder(obs, hidden_state, cell_state)
        adv_out = self.adv(lstm_out[:, -1, :])
        val_out = self.val(lstm_out[:, -1, :])
        qout = val_out.expand(-1, np.prod(np.array(self.action_shape))) + (
                adv_out - adv_out.mean(dim=-1).unsqueeze(dim=-1).expand(-1, np.prod(np.array(self.action_shape))))
        qout = qout.view((bsize,) + self.action_shape)
        return lstm_out, (hidden_state, cell_state), qout

class DQN(nn.Module):
    def __init__(self, NN_param, method_param):
        super().__init__()
        self.action_shape = tuple(NN_param["action_shape"])
        self.gamma = method_param["RL_gamma"]
        self.target_update_freq = method_param["target_update_freq(epochs)"]
        self.main_model = DQNNet(NN_param["cnn_out_size"], NN_param["lstm_hidden_size"],
                                 NN_param["atten_size"], NN_param["action_shape"])

    def evaluate(self, obs, hidden_state, cell_state):
        return self.main_model(obs, hidden_state, cell_state)

    def initialize_training(self, initial_learning_rate, learning_rate_step_size, learning_rate_gamma):
        self.target_model = deepcopy(self.main_model)
        self.optimizer = torch.optim.Adam(self.main_model.parameters(), lr=initial_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=learning_rate_step_size,
                                                         gamma=learning_rate_gamma)
        self.criterion = nn.SmoothL1Loss()

    def get_initial_value(self, bsize):
        hidden_state, cell_state = self.main_model.encoder.lstm.init_hidden_states_and_outputs(bsize)
        return hidden_state, cell_state

    def learn(self, batches, epoch):

        loss_stat = {}
        loss_stat["SmoothL1Loss"] = []
        for batch in batches:
            pred_values = []
            target_values = []
            for episode_idx in range(len(batch)):
                hidden_state, cell_state = self.get_initial_value(
                    bsize=1)
                hidden_state, cell_state = hidden_state.to(next(self.main_model.parameters()).device), cell_state.to(
                    next(self.main_model.parameters()).device)
                act, current_vector_obs, current_visual_obs, next_vector_obs, next_visual_obs, rewards = preprocess_data_from_batch(
                    batch)
                act_per_episode, current_visual_obs_per_episode, current_vector_obs_per_episode, rewards_per_episode, visual_obs_per_episode, next_visual_obs_per_episode, next_vector_obs_per_episode, _ = extract_input_per_episode(
                    act, episode_idx, current_vector_obs, current_visual_obs, next_vector_obs, next_visual_obs,
                    rewards, next(self.main_model.parameters()).device)

                out, _, Q_s = self.main_model(
                    current_visual_obs_per_episode,
                    hidden_state=hidden_state, cell_state=cell_state)
                if next_vector_obs_per_episode[-1][0] == 0:
                    Q_next_max = 0
                else:
                    out_target, _, Q_next = self.target_model(
                        next_visual_obs_per_episode,
                        hidden_state=hidden_state, cell_state=cell_state)
                    Q_next_max = torch.max(Q_next.reshape(-1).detach())
                pred_values.append(Q_s[0][tuple(np.array(act_per_episode[0, -1].cpu()) + 1)])
                target_values.append(rewards_per_episode[-1] + (self.gamma * Q_next_max))

            pred_values = torch.stack(pred_values)
            target_values = torch.stack(target_values).to(next(self.main_model.parameters()).device)

            loss = self.criterion(pred_values, target_values)

            #  save performance measure
            loss_stat["SmoothL1Loss"].append(loss.item())

            # make previous grad zero
            self.optimizer.zero_grad()

            # backward
            loss.backward()

            # Synchronize gradients from all learners
            sync_grads(self.main_model)

            # update params
            self.optimizer.step()

            # release memory
            torch.cuda.empty_cache()
        self.scheduler.step()
        if (epoch + 1) % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.main_model.state_dict())
        return loss_stat

    def get_model_state_dict(self):
        model_state_dict = {}
        model_state_dict["DQN"] = self.main_model.state_dict()
        return model_state_dict

    def get_optimizer_state_dict(self):
        optimizer_state_dict = {}
        optimizer_state_dict["DQN"] = self.optimizer.state_dict()
        return optimizer_state_dict

    def load_model_state_dict(self, model_state_dict):
        self.main_model.load_state_dict(model_state_dict["DQN"])

    def load_optimizer_state_dict(self, optimizer_state_dict):
        self.optimizer.load_state_dict(optimizer_state_dict["DQN"])
