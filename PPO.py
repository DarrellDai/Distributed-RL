import copy

import numpy as np
import torch
import torch.nn as nn

from Encoder import Encoder
from utils import preprocess_data_from_batch, extract_input_per_episode, sync_grads


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width=64):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, net_width),
                                   nn.Tanh(),
                                   nn.Linear(net_width, net_width),
                                   nn.Tanh(),
                                   nn.Linear(net_width, action_dim),
                                   nn.Softmax(-1))

    def forward(self, state):
        return self.actor(state)


class Critic(nn.Module):
    def __init__(self, state_dim, net_width=64):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(nn.Linear(state_dim, net_width),
                                    nn.ReLU(),
                                    nn.Linear(net_width, net_width),
                                    nn.ReLU(),
                                    nn.Linear(net_width, 1))

    def forward(self, state):
        return self.critic(state)


class PPO(nn.Module):
    def __init__(self, cnn_out_size, action_shape, lstm_hidden_size, atten_size, net_width=64):
        super().__init__()
        self.action_shape = action_shape
        self.encoder = Encoder(cnn_out_size, action_shape, lstm_hidden_size, atten_size)
        self.actor = Actor(lstm_hidden_size, np.prod(np.array(self.action_shape)), net_width)
        self.critic = Critic(lstm_hidden_size, net_width)

    def forward(self, obs, hidden_state, cell_state):
        bsize = obs.shape[0]
        lstm_out, (hidden_state, cell_state) = self.encoder(obs, hidden_state, cell_state)
        act_prob = self.actor(lstm_out[:, -1, :])
        act_prob = act_prob.view((bsize,) + self.action_shape)
        value = self.critic(lstm_out[:, -1, :])
        return lstm_out, (hidden_state, cell_state), (act_prob, value)

    def evaluate(self, obs, hidden_state, cell_state):
        return self(obs, hidden_state, cell_state)

    def initialize_training(self, initial_learning_rate, learning_rate_step_size, learning_rate_gamma):
        self.actor_optimizer = torch.optim.Adam(self.parameters(), lr=initial_learning_rate)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=learning_rate_step_size,
                                                               gamma=learning_rate_gamma)
        self.critic_optimizer = torch.optim.Adam(self.parameters(), lr=initial_learning_rate)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer,
                                                                step_size=learning_rate_step_size,
                                                                gamma=learning_rate_gamma)
        self.critic_criterion = nn.MSELoss()

    def get_initial_value(self, bsize):
        hidden_state, cell_state = self.encoder.lstm.init_hidden_states_and_outputs(bsize)
        return hidden_state, cell_state

    def learn(self, batches, epoch, num_iter_per_batch, gamma, lambd, clip_rate):
        loss_stat = {}
        loss_stat["actor_loss"] = []
        loss_stat["critic_loss"] = []
        for batch in batches:
            old_act_probs = []
            target_values = []
            advs = []
            for i in range(num_iter_per_batch):
                pred_values = []
                actor_losses = []
                for episode_idx in range(len(batch)):
                    act_prob_per_episode = []
                    value_per_episode = []
                    hidden_state, cell_state = self.encoder.lstm.init_hidden_states_and_outputs(
                        bsize=1)
                    hidden_state, cell_state = hidden_state.to(next(self.parameters()).device), cell_state.to(
                        next(self.parameters()).device)
                    act, current_vector_obs, current_visual_obs, next_vector_obs, next_visual_obs, rewards = preprocess_data_from_batch(
                        batch)
                    act_per_episode, current_visual_obs_per_episode, current_vector_obs_per_episode, rewards_per_episode, visual_obs_per_episode, next_vector_obs_per_episode, done_mask = extract_input_per_episode(
                        act, episode_idx, current_vector_obs, current_visual_obs, next_vector_obs, next_visual_obs,
                        rewards, next(self.parameters()).device)

                    for t in range(len(batch[episode_idx]) + 1):
                        hidden_state = hidden_state.detach()
                        cell_state = cell_state.detach()
                        out, (hidden_state, cell_state), (act_prob, value) = self(
                            visual_obs_per_episode[:, t:t + 1],
                            hidden_state=hidden_state, cell_state=cell_state)
                        if t < len(batch[episode_idx]):
                            act_prob_per_episode.append(act_prob[0][
                                                            tuple(np.array(act_per_episode[0, t].cpu()) + 1)])
                        value_per_episode.append(value[0, 0])
                    act_prob_per_episode = torch.stack(act_prob_per_episode)
                    value_per_episode = torch.stack(value_per_episode)
                    if i == 0:
                        deltas = rewards_per_episode + gamma * value_per_episode[1:] * (
                                1 - done_mask[1:]) - value_per_episode[0:-1]
                        deltas = deltas.detach().cpu().flatten().numpy()
                        done_mask = done_mask.detach().cpu().flatten().numpy()
                        adv = [0]

                        '''done for GAE'''
                        for dlt, mask in zip(deltas[::-1], done_mask[:0:-1]):
                            advantage = dlt + gamma * lambd * adv[-1] * (1 - mask)
                            adv.append(advantage)
                        adv.reverse()
                        adv = copy.deepcopy(adv[0:-1])
                        adv = torch.tensor(adv).float().to(next(self.parameters()).device)
                        target_value = adv + value_per_episode[:-1]
                        target_values.append(target_value.detach())
                        adv = (adv - adv.mean()) / ((adv.std() + 1e-4))
                        advs.append(adv)
                        old_act_prob = act_prob_per_episode
                        old_act_probs.append(old_act_prob.detach())

                    pred_values.append(value_per_episode[:-1])
                    ratio = torch.exp(torch.log(act_prob_per_episode) - torch.log(old_act_probs[episode_idx]))
                    surr1 = ratio * advs[episode_idx]
                    surr2 = torch.clamp(ratio, 1 - clip_rate, 1 + clip_rate) * advs[episode_idx]
                    actor_loss_per_episode = -torch.min(surr1, surr2)
                    actor_loss_per_episode = actor_loss_per_episode.sum()
                    actor_losses.append(actor_loss_per_episode)

                actor_losses = torch.stack(actor_losses)
                actor_loss = actor_losses.mean()

                pred_values = torch.stack(pred_values)
                if i == 0:
                    target_values = torch.stack(target_values).to(next(self.parameters()).device)
                critic_loss = self.critic_criterion(pred_values, target_values)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                critic_loss.backward()
                # Synchronize gradients from all learners
                sync_grads(self)
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                loss_stat["actor_loss"] = actor_loss.detach().cpu()
                loss_stat["critic_loss"] = critic_loss.detach().cpu()
                # release memory
                torch.cuda.empty_cache()

        self.actor_scheduler.step()
        self.critic_scheduler.step()

        return loss_stat

    def get_model_state_dict(self):
        model_state_dict = {}
        model_state_dict["actor"] = self.actor.state_dict()
        model_state_dict["critic"] = self.critic.state_dict()
        return model_state_dict

    def get_optimizer_state_dict(self):
        optimizer_state_dict = {}
        optimizer_state_dict["actor"] = self.actor_optimizer.state_dict()
        optimizer_state_dict["critic"] = self.critic_optimizer.state_dict()
        return optimizer_state_dict

    def load_model_state_dict(self, model_state_dict):
        self.actor.load_state_dict(model_state_dict["actor"])
        self.critic.load_state_dict(model_state_dict["critic"])

    def load_optimizer_state_dict(self, optimizer_state_dict):
        self.actor_optimizer.load_state_dict(optimizer_state_dict["actor"])
        self.critic_optimizer.load_state_dict(optimizer_state_dict["critic"])
