import numpy as np
import torch
import torch.nn as nn

from Encoder import Encoder
from utils import preprocess_data_from_batch, extract_input_per_episode, sync_grads


class BCNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.bc_net = nn.Sequential(nn.Linear(input_size, int(input_size / 3)),
                                    nn.ReLU(),
                                    nn.Linear(int(input_size / 3), int(input_size / 9)),
                                    nn.ReLU(),
                                    nn.Linear(int(input_size / 9), output_size),
                                    nn.Softmax(-1)
                                    )

    def forward(self, x):
        x = self.bc_net(x)
        return x


class Behavior_Cloning(nn.Module):
    def __init__(self, cnn_out_size, action_shape, lstm_hidden_size, atten_size):
        super().__init__()
        self.action_shape = action_shape
        self.encoder = Encoder(cnn_out_size, action_shape, lstm_hidden_size, atten_size)
        self.bc = BCNet(lstm_hidden_size, np.prod(np.array(self.action_shape)))

    def forward(self, obs, hidden_state, cell_state):
        bsize = obs.shape[0]
        lstm_out, (hidden_state, cell_state) = self.encoder(obs, hidden_state, cell_state)
        act_prob = self.bc(lstm_out[:, -1, :])
        act_prob = act_prob.view((bsize,) + self.action_shape)
        return lstm_out, (hidden_state, cell_state), act_prob
    def evaluate(self, obs, hidden_state, cell_state):
        return self(obs, hidden_state, cell_state)
    def initialize_training(self, initial_learning_rate, learning_rate_step_size, learning_rate_gamma):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=initial_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=learning_rate_step_size,
                                                         gamma=learning_rate_gamma)
        self.criterion = nn.CrossEntropyLoss()
    def get_initial_value(self, bsize):
        hidden_state, cell_state=self.encoder.lstm.init_hidden_states_and_outputs(bsize)
        return hidden_state, cell_state
    def learn(self, batches, epoch):
        loss_stat = {}
        loss_stat["BC_loss"] = []
        for batch in batches:
            pred_values = []
            target_values = []
            for episode_idx in range(len(batch)):
                hidden_state, cell_state = self.encoder.lstm.init_hidden_states_and_outputs(
                    bsize=1)
                hidden_state, cell_state = hidden_state.to(next(self.parameters()).device), cell_state.to(
                    next(self.parameters()).device)
                act, current_vector_obs, current_visual_obs, next_vector_obs, next_visual_obs, rewards = preprocess_data_from_batch(
                    batch)
                act_per_episode, current_visual_obs_per_episode, current_vector_obs_per_episode, rewards_per_episode, visual_obs_per_episode, next_vector_obs_per_episode, _ = extract_input_per_episode(
                    act, episode_idx, current_vector_obs, current_visual_obs, next_vector_obs, next_visual_obs,
                    rewards, next(self.parameters()).device)

                for t in range(len(batch[episode_idx])):
                    out, (hidden_state, cell_state), act_prob = self(
                        current_visual_obs_per_episode[:, t:t + 1],
                        hidden_state=hidden_state, cell_state=cell_state)
                    pred_values.append(act_prob.view(-1))
                    target_values.append(torch.tensor(
                        np.ravel_multi_index(np.array(act_per_episode[0, t].cpu()) + 1,
                                             act_prob[0].shape)).detach())

            pred_values = torch.stack(pred_values)
            target_values = torch.stack(target_values).to(next(self.parameters()).device)

            loss = self.criterion(pred_values, target_values)

            #  save performance measure
            loss_stat["BC_loss"].append(loss.item())

            # make previous grad zero
            self.optimizer.zero_grad()

            # backward
            loss.backward()

            # Synchronize gradients from all learners
            sync_grads(self)

            # update params
            self.optimizer.step()

            # release memory
            torch.cuda.empty_cache()
        self.scheduler.step()

        return loss_stat

    def get_model_state_dict(self):
        model_state_dict = {}
        model_state_dict["BC"] = self.state_dict()
        return model_state_dict

    def get_optimizer_state_dict(self):
        optimizer_state_dict = {}
        optimizer_state_dict["BC"] = self.optimizer.state_dict()
        return optimizer_state_dict

    def load_model_state_dict(self, model_state_dict):
        self.load_state_dict(model_state_dict["BC"])

    def load_optimizer_state_dict(self, optimizer_state_dict):
        self.optimizer.load_state_dict(optimizer_state_dict["BC"])
