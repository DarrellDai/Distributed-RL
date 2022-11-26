import numpy as np
import torch
import torch.nn as nn

from utils import preprocess_data_from_batch, extract_input_per_episode


class DQN(nn.Module):

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        self.qnetwork = nn.Sequential(nn.Linear(input_size, input_size),
                                      nn.ReLU(),
                                      nn.Linear(input_size, input_size),
                                      nn.ReLU(),
                                      nn.Linear(input_size, output_size))

    def forward(self, x):
        qout = self.qnetwork(x)

        return qout

    def learn(self, agent_ids, main_model, target_model, optimizer, criterion, batches, gamma,
              loss_stat, device, scheduler=None):
        for id in agent_ids:
            for batch in batches[id]:
                pred_values = []
                target_values = []
                for episode_idx in range(len(batch)):
                    hidden_state, cell_state = main_model[id].lstm.init_hidden_states_and_outputs(
                        bsize=1)
                    hidden_state, cell_state = hidden_state.to(device), cell_state.to(device)
                    hidden_state_target, cell_state_target = target_model[
                        id].lstm.init_hidden_states_and_outputs(
                        bsize=1)
                    hidden_state_target, cell_state_target = hidden_state_target.to(device), cell_state_target.to(
                        device)
                    act, current_vector_obs, current_visual_obs, next_vector_obs, next_visual_obs, rewards = preprocess_data_from_batch(
                        batch)
                    act_per_episode, current_visual_obs_per_episode, current_vector_obs_per_episode, rewards_per_episode, visual_obs_per_episode, next_vector_obs_per_episode, done_mask = extract_input_per_episode(
                        act, episode_idx, current_vector_obs, current_visual_obs, next_vector_obs, next_visual_obs,
                        rewards)

                    for t in range(len(batch[episode_idx])):
                        if done_mask[t+1] == 0:
                            Q_next_max = 0
                        else:
                            out_target, (hidden_state_target, cell_state_target), Q_next = target_model[
                                id](
                                visual_obs_per_episode[:, t + 1:t + 2],
                                hidden_state=hidden_state_target,
                                cell_state=cell_state_target)
                            Q_next_max = torch.max(Q_next.reshape(-1).detach())
                        out, (hidden_state, cell_state), Q_s = main_model[id](
                            current_visual_obs_per_episode[:, t:t + 1],
                            hidden_state=hidden_state, cell_state=cell_state)
                        pred_values.append(Q_s[0][tuple(np.array(act_per_episode[0, t].cpu()) + 1)])
                        target_values.append(rewards_per_episode[t] + (gamma * Q_next_max))

                pred_values = torch.stack(pred_values)
                target_values = torch.stack(target_values).to(device)

                loss = criterion(pred_values, target_values)

                #  save performance measure
                loss_stat[id]["DQN_loss"]=loss.item()

                # make previous grad zero
                optimizer[id].zero_grad()

                # backward
                loss.backward()

                # # Synchronize gradients from all learners
                # sync_grads(self.main_model[id])
                # update params
                optimizer[id].step()

                # release memory
                torch.cuda.empty_cache()
        if scheduler is not None:
            scheduler[id].step()
