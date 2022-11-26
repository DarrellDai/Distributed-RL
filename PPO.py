import copy

import numpy as np
import torch
import torch.nn as nn

from utils import preprocess_data_from_batch, extract_input_per_episode


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


class Critic:
    def __init__(self, state_dim, action_dim, net_width=64):
        self.actor = Actor(state_dim, action_dim, net_width)
        self.critic = Critic(state_dim, net_width)

    def learn(self, agent_ids, action_shape, main_model, actor_optimizer, critic_optimizer, criterion, batches,
              num_iter_per_batch, gamma, lambd, clip_rate,
              loss_stat, device, scheduler=None):
        for id in agent_ids:
            for batch in batches[id]:
                old_act_probs = []
                pred_values = []
                target_values = []
                advs = []
                actor_losses=[]
                for i in range(num_iter_per_batch):
                    for episode_idx in range(len(batch)):
                        act_prob_per_episode = torch.zeros(len(batch[episode_idx]) + 1)
                        value_per_episode = torch.zeros(len(batch[episode_idx]) + 1)
                        hidden_state, cell_state = main_model[id].lstm.init_hidden_states_and_outputs(
                            bsize=1)
                        hidden_state, cell_state = hidden_state.to(device), cell_state.to(device)
                        act, current_vector_obs, current_visual_obs, next_vector_obs, next_visual_obs, rewards = preprocess_data_from_batch(
                            batch)
                        act_per_episode, current_visual_obs_per_episode, current_vector_obs_per_episode, rewards_per_episode, visual_obs_per_episode, next_vector_obs_per_episode, done_mask = extract_input_per_episode(
                            act, episode_idx, current_vector_obs, current_visual_obs, next_vector_obs, next_visual_obs,
                            rewards)

                        for t in range(len(batch[episode_idx]) + 1):
                            out, (hidden_state, cell_state), (act_prob, value) = main_model[id](
                                visual_obs_per_episode[:, t:t + 1],
                                hidden_state=hidden_state, cell_state=cell_state)
                            act_prob_per_episode[t] = act_prob[0][
                                tuple(np.array(act_per_episode[0, t].cpu()) + 1)]
                            value_per_episode[t] = value[0, 0, 0]
                        if i == 0:
                            deltas = rewards_per_episode + gamma * value_per_episode[1:] * (
                                        1 - done_mask[1:]) - value_per_episode[0:-1]
                            deltas = deltas.cpu().flatten().numpy()
                            adv = [0]

                            '''done for GAE'''
                            for dlt, mask in zip(deltas[::-1], done_mask[1::-1]):
                                advantage = dlt + gamma * lambd * adv[-1] * (1 - mask)
                                adv.append(advantage)
                            adv.reverse()
                            adv = copy.deepcopy(adv[0:-1])
                            adv = torch.tensor(adv).unsqueeze(1).float().to(device)
                            target_value = adv + value_per_episode
                            target_values.append(target_value)
                            pred_values.append(value_per_episode)
                            adv = (adv - adv.mean()) / ((adv.std() + 1e-4))
                            advs.append(adv)
                            old_act_probs.append(act_prob_per_episode)

                        ratio = torch.exp(torch.log(act_prob_per_episode) - torch.log(old_act_probs[episode_idx]))
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1 - clip_rate, 1 + clip_rate) * adv
                        actor_loss_per_episode = -torch.min(surr1, surr2)
                        actor_loss_per_episode=actor_loss_per_episode.sum()
                        actor_losses.append(actor_loss_per_episode)

                    actor_losses=torch.stack(actor_losses)
                    actor_loss=actor_losses.mean()
                    loss_stat[id]["actor_loss"] = actor_loss
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                    actor_optimizer.step()

                    pred_values = torch.stack(pred_values)
                    target_values = torch.stack(target_values).to(device)

                    critic_loss = criterion(pred_values, target_values)

                    #  save performance measure
                    loss_stat[id]["critic_loss"]=critic_loss

                    # make previous grad zero
                    critic_optimizer[id].zero_grad()

                    # backward
                    critic_loss.backward()

                    # # Synchronize gradients from all learners
                    # sync_grads(self.main_model[id])
                    # update params
                    critic_optimizer[id].step()

                    # release memory
                    torch.cuda.empty_cache()
        if scheduler is not None:
            scheduler[id].step()
