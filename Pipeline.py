import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from Network import Network
from unity_wrappers.envs import MultiUnityWrapper
from mlagents_envs.environment import UnityEnvironment
from copy import deepcopy

import os

from utils import find_name_of_agents, find_optimal_action, \
    convert_to_array, \
    find_hidden_cell_out_of_an_action, \
    combine_out, \
    save_checkpoint, \
    load_checkpoint
from tqdm import tqdm
from collections import deque


class Pipeline:
    def __init__(self):

        self.cnn_out_size = {}
        self.lstm_hidden_size = {}
        self.action_shape = {}
        self.action_out_size = {}
        self.atten_size = {}
        self.cnn_out_size = {}
        self.lstm_hidden_size = {}
        self.action_out_size = {}

    def initialize_model(self, cnn_out_size, lstm_hidden_size, action_shape, action_out_size, atten_size,
                         device_idx):
        self.device_idx = device_idx
        self.device = torch.device('cuda:' + str(self.device_idx[0]) if torch.cuda.is_available() else 'cpu')

        for idx in range(len(self.agent_ids)):
            self.cnn_out_size[self.agent_ids[idx]] = cnn_out_size[idx]
            self.lstm_hidden_size[self.agent_ids[idx]] = lstm_hidden_size[idx]
            self.action_shape[self.agent_ids[idx]] = tuple(action_shape[idx])
            self.atten_size[self.agent_ids[idx]] = atten_size[idx]
            self.action_out_size[self.agent_ids[idx]] = action_out_size[idx]

        self.main_model = {}

        for id in self.agent_ids:
            self.main_model[id] = Network(cnn_out_size=self.cnn_out_size[id],
                                          lstm_hidden_size=self.lstm_hidden_size[id],
                                          atten_size=self.atten_size[id], action_space_shape=self.action_shape[id],
                                          action_out_size=self.action_out_size[id])
            if torch.cuda.is_available():
                self.main_model[id] = nn.DataParallel(self.main_model[id], device_ids=self.device_idx).to(self.device)
            else:
                self.main_model[id] = self.main_model[id].to(self.device)

    def initialize_env(self, env_path):
        unity_env = UnityEnvironment(env_path)
        self.env = MultiUnityWrapper(unity_env=unity_env, uint8_visual=True, allow_multiple_obs=True)
        self.agent_ids = tuple(self.env.agent_id_to_behaviour_name.keys())
        self.id_to_name = find_name_of_agents(self.env.agent_id_to_behaviour_name, self.agent_ids)

    def initialize_training(self, learning_rate):
        criterion = nn.MSELoss()
        target_model = {}
        optimizer = {}
        for id in self.agent_ids:
            target_model[id] = Network(cnn_out_size=self.cnn_out_size[id], lstm_hidden_size=self.lstm_hidden_size[id],
                                       atten_size=self.atten_size[id], action_space_shape=self.action_shape[id],
                                       action_out_size=self.action_out_size[id])
            if torch.cuda.is_available():
                target_model[id] = nn.DataParallel(target_model[id], device_ids=self.device_idx).to(self.device)
            else:
                target_model[id] = target_model[id].to(self.device)
            target_model[id].load_state_dict(self.main_model[id].state_dict())
            optimizer[id] = torch.optim.Adam(self.main_model[id].parameters(), lr=learning_rate)

        return criterion, optimizer, target_model

    def fill_memory_with_random_walk(self, memory, max_steps):
        for _ in tqdm(range(memory.memsize)):
            prev_obs = self.env.reset()
            step_count = 0
            local_memory = {}
            alive = {}
            for id in self.agent_ids:
                local_memory[id] = []
                alive[id] = True
            done = False

            while step_count < max_steps and not done:
                act = {}
                # {0: agent 0's action, 1: ...]
                for id in self.agent_ids:
                    act[id] = self.env.action_space[id].sample() - 1

                obs_dict, reward_dict, done_dict, info_dict = self.env.step(act)
                step_count += 1
                done = done_dict["__all__"]
                for id in self.agent_ids:
                    local_memory[id].append(
                        (deepcopy(prev_obs[id]), deepcopy(act[id]), deepcopy(reward_dict[id]), deepcopy(obs_dict[id])))
                # for id in self.agent_ids:
                #     if alive[id]:
                #         local_memory[id].append(
                #             (deepcopy(prev_obs[id]), deepcopy(act[id]), deepcopy(reward_dict[id]), deepcopy(obs_dict[id])))
                #         alive[id]=prev_obs[id][1][list(prev_obs[id][1].keys())[0][0]]
                prev_obs = obs_dict
            memory.add_episode(local_memory)

        print('\n Populated with %d Episodes' % (len(memory.memory[memory.agent_ids[0]])))

    def find_random_action_while_updating_LSTM(self, prev_obs, hidden_state, cell_state, lstm_out):
        act = {}
        for id in self.agent_ids:
            act[id] = self.env.action_space[id].sample() - 1
            act[id] = torch.tensor(act[id]).reshape(1, 1, len(act[id])).to(self.device)
            prev_obs[id][0] = prev_obs[id][0].reshape(1, 1, prev_obs[id][0].shape[0], prev_obs[id][0].shape[1],
                                                      prev_obs[id][0].shape[2])
            model_out = self.main_model[id](prev_obs[id][0], act[id],
                                            hidden_state=hidden_state[id],
                                            cell_state=cell_state[id], lstm_out=lstm_out[id])

            hidden_state[id] = model_out[1][0]
            cell_state[id] = model_out[1][1]
            lstm_out[id] = model_out[0]
        return act, hidden_state, cell_state, lstm_out

    def find_best_action_by_model(self, prev_obs, hidden_state, cell_state, lstm_out):
        act = {}
        for id in self.agent_ids:
            prev_obs[id][0] = prev_obs[id][0].reshape(1, 1, prev_obs[id][0].shape[0], prev_obs[id][0].shape[1],
                                                      prev_obs[id][0].shape[2])
            lo, _, dqn_out, out_per_action, (
                hidden_state_per_action, cell_state_per_action) = self.main_model[id](prev_obs[id][0],
                                                                                      act=
                                                                                      torch.zeros((1, 0,
                                                                                                   len(
                                                                                                       self.action_shape[
                                                                                                           id]))).to(
                                                                                          self.device),
                                                                                      hidden_state=
                                                                                      hidden_state[id],
                                                                                      cell_state=
                                                                                      cell_state[id],
                                                                                      lstm_out=lstm_out[id])

            act[id] = find_optimal_action(dqn_out)
            act[id] = torch.from_numpy(act[id]).to(self.device)
            hidden_state[id], cell_state[id], lo_new = find_hidden_cell_out_of_an_action(act[id],
                                                                                         hidden_state_per_action,
                                                                                         cell_state_per_action,
                                                                                         out_per_action)
            lstm_out[id] = combine_out(lo, lo_new.to(self.device), self.atten_size[id])
            hidden_state[id] = hidden_state[id]
            cell_state[id] = cell_state[id]
        return act, hidden_state, cell_state, lstm_out

    def resume_training(self, checkpoint_to_load, optimizer, target_model):
        model_state_dicts, optimizer_state_dicts, total_steps, episode_count, epsilon, mem = load_checkpoint(
            checkpoint_to_load, self.device)
        start_episode = episode_count
        for id in self.agent_ids:
            self.main_model[id].load_state_dict(model_state_dicts[id])
            target_model[id].load_state_dict(self.main_model[id].state_dict())
            optimizer[id].load_state_dict(optimizer_state_dicts[id])
        return target_model, optimizer, total_steps, start_episode, episode_count, epsilon, mem

    def train(self, start_episode, episode_count, total_episodes, total_steps, gamma, epsilon, final_epsilon,
              epsilon_vanish_rate, max_steps, target_model,
              memory, batch_size, time_step, learning_rate, target_update_freq, update_freq, optimizer, criterion,
              performance_display_interval, checkpoint_save_interval,
              checkpoint_to_save, name_tensorboard):

        writer = SummaryWriter(os.path.join("runs", name_tensorboard))
        total_reward = {}
        loss_stat = {}
        local_memory = {}
        hidden_state = {}
        cell_state = {}
        lstm_out = {}
        alive = {}
        for episode in tqdm(range(start_episode, total_episodes)):
            episode_count += 1
            step_count = 0
            prev_obs = self.env.reset()


            for id in self.agent_ids:
                total_reward[id] = 0
                loss_stat[id] = []
                local_memory[id] = []
                alive[id] = True
                hidden_state[id], cell_state[id], lstm_out[id] = self.main_model[
                    id].module.lstm.init_hidden_states_and_outputs(
                    bsize=1)
            done = False
            while step_count < max_steps and not done:
                for id in self.agent_ids:
                    prev_obs[id][0] = torch.from_numpy(prev_obs[id][0]).float().to(self.device)
                step_count += 1
                total_steps += 1
                with torch.no_grad():
                    if np.random.rand(1) < epsilon:
                        act, hidden_state, cell_state, lstm_out = self.find_random_action_while_updating_LSTM(prev_obs,
                                                                                                              hidden_state,
                                                                                                              cell_state,
                                                                                                              lstm_out)
                    else:
                        act, hidden_state, cell_state, lstm_out = self.find_best_action_by_model(prev_obs, hidden_state,
                                                                                                 cell_state, lstm_out)
                    obs_dict, reward_dict, done_dict, info_dict = self.env.step(act)

                    done = done_dict["__all__"]

                    for id in self.agent_ids:
                        total_reward[id] += reward_dict[id]

                        prev_obs[id][0] = prev_obs[id][0].reshape(prev_obs[id][0].shape[2], prev_obs[id][0].shape[3],
                                                                  prev_obs[id][0].shape[4])
                        prev_obs[id][0] = np.array(prev_obs[id][0].cpu())
                        act[id] = act[id].reshape(-1)
                        act[id]=np.array(act[id].cpu())

                        local_memory[id].append(
                            (deepcopy(prev_obs[id]), deepcopy(act[id]), deepcopy(reward_dict[id]),
                             deepcopy(obs_dict[id])))

                    prev_obs = deepcopy(obs_dict)

                if (total_steps % target_update_freq) == 0:
                    for id in self.agent_ids:
                        target_model[id].load_state_dict(self.main_model[id].state_dict())

                if (total_steps % update_freq) == 0:
                    self.learn(batch_size, time_step, gamma, memory, criterion, optimizer, target_model, loss_stat)
            # save performance measure
            memory.add_episode(local_memory)

            loss = {}
            for id in self.agent_ids:
                if len(loss_stat[id]) > 0:
                    loss[id] = np.mean(loss_stat[id])
                else:
                    loss[id] = 0
                writer.add_scalar(self.id_to_name[id] + ": Loss/train", loss[id], episode_count)
                writer.add_scalar(self.id_to_name[id] + ": Reward/train", total_reward[id], episode_count)
            writer.flush()

            if epsilon > final_epsilon:
                epsilon *= epsilon_vanish_rate

            if (episode + 1) % performance_display_interval == 0:
                print('\n Episode: [%d | %d] LR: %f, Epsilon : %f \n' % (
                    episode, start_episode + total_episodes, learning_rate, epsilon))
                for id in self.agent_ids:
                    print('\n Agent %d, Reward: %f, Loss: %f \n' % (id, total_reward[id], loss[id]))

            if (episode + 1) % checkpoint_save_interval == 0:
                model_state_dicts = {}
                optimizer_state_dicts = {}
                for id in self.agent_ids:
                    model_state_dicts[id] = self.main_model[id].state_dict()
                    optimizer_state_dicts[id] = optimizer[id].state_dict()

                save_checkpoint({
                    'model_state_dicts': model_state_dicts,
                    'optimizer_state_dicts': optimizer_state_dicts,
                    'epsilon': epsilon,
                    'total_steps': total_steps,
                    "episode_count": episode_count,
                    "memory": memory
                }, filename=checkpoint_to_save)
        writer.close()

    def train_from_human_play(self, batch_size, time_step, gamma, memory, criterion, optimizer, learning_rate,
                              target_model, name_tensorboard, total_epochs, target_update_freq, checkpoint_save_interval, checkpoint_to_save):
        writer = SummaryWriter(os.path.join("runs", name_tensorboard))
        loss_stat = {}
        for epoch in tqdm(range(total_epochs)):
            for id in self.agent_ids:
                loss_stat[id] = []
            self.learn(batch_size, time_step, gamma, memory, criterion, optimizer, target_model,loss_stat)
            print('\n Epoch: [%d | %d] LR: %f \n' % (epoch, total_epochs, learning_rate))
            for id in self.agent_ids:
                print('\n Agent %d, Loss: %f \n' % (id, np.mean(loss_stat[id])))
                writer.add_scalar(self.id_to_name[id] + ": Loss/train", np.mean(loss_stat[id]), epoch)
            writer.flush()
            if (epoch+1) % target_update_freq ==0:
                for id in self.agent_ids:
                    target_model[id].load_state_dict(self.main_model[id].state_dict())
            if (epoch+1) % checkpoint_save_interval == 0:
                model_state_dicts = {}
                optimizer_state_dicts = {}
                for id in self.agent_ids:
                    model_state_dicts[id] = self.main_model[id].state_dict()
                    optimizer_state_dicts[id] = optimizer[id].state_dict()

                save_checkpoint({
                    'model_state_dicts': model_state_dicts,
                    'optimizer_state_dicts': optimizer_state_dicts,
                    'epsilon': 1,
                    'total_steps': 1,
                    "episode_count": 1,
                    "memory": {}
                }, filename=checkpoint_to_save)
    def learn(self, batch_size, time_step, gamma, memory, criterion, optimizer, target_model, loss_stat):
        for id in self.agent_ids:
            batches = memory.get_batch(bsize=batch_size, time_step=time_step, agent_id=id)
            for batch in batches:
                hidden_batch, cell_batch, out_batch = self.main_model[id].module.lstm.init_hidden_states_and_outputs(
                    bsize=len(batch))
                current_visual_obs = []
                current_vector_obs = []
                act = []
                rewards = []
                next_visual_obs = []
                next_vector_obs = []

                for b in batch:
                    cvis, cves, ac, rw, nvis, nves = [], [], [], [], [], []
                    for element in b:
                        cvis.append(element[0][0])
                        cves.append(element[0][1][list(element[0][1].keys())[0]])
                        ac.append(element[1])
                        rw.append(element[2])
                        nvis.append(element[3][0])
                        nves.append(element[3][1][list(element[0][1].keys())[0]])
                    current_visual_obs.append(cvis)
                    current_vector_obs.append(cves)
                    act.append(ac)
                    rewards.append(rw)
                    next_visual_obs.append(nvis)
                    next_vector_obs.append(nves)

                current_visual_obs = np.array(current_visual_obs)
                current_vector_obs = np.array(current_vector_obs)
                act = np.array(act)
                rewards = np.array(rewards)
                next_visual_obs = np.array(next_visual_obs)
                next_vector_obs = np.array(next_vector_obs)

                current_visual_obs = torch.from_numpy(current_visual_obs).float().to(self.device)
                act = torch.from_numpy(act).long().to(self.device)
                rewards = torch.from_numpy(rewards).float().to(self.device)
                next_visual_obs = torch.from_numpy(next_visual_obs).float().to(self.device)
                visual_obs = torch.concat((current_visual_obs, next_visual_obs[:, -1:]), 1)
                Q_next_max = torch.zeros(len(batch)).float().to(self.device)
                for batch_idx in range(len(batch)):
                    _, _, Q_next, _, _ = target_model[id](visual_obs[batch_idx:batch_idx + 1],
                                                          act[batch_idx:batch_idx + 1],
                                                          hidden_state=hidden_batch[
                                                                       batch_idx:batch_idx + 1],
                                                          cell_state=cell_batch[
                                                                     batch_idx:batch_idx + 1],
                                                          lstm_out=out_batch[
                                                                   batch_idx:batch_idx + 1])
                    Q_next_max[batch_idx] = torch.max(Q_next.reshape(-1))
                target_values = rewards[:, time_step - 1] + (gamma * Q_next_max)
                target_values = target_values.float()
                _, _, Q_s, _, _ = self.main_model[id](current_visual_obs, act,
                                                      hidden_state=hidden_batch, cell_state=cell_batch,
                                                      lstm_out=out_batch)

                Q_s_a = Q_s[:, 0, 0]
                loss = criterion(Q_s_a, target_values)

                #  save performance measure
                loss_stat[id].append(loss.item())

                # make previous grad zero
                optimizer[id].zero_grad()

                # backward
                loss.backward()
                # update params
                optimizer[id].step()


    def test(self, max_steps, total_episodes, checkpoint_to_load, name_tensorboard):
        model_state_dicts, _, _, _, _, _ = load_checkpoint(
            checkpoint_to_load, self.device)
        for id in self.agent_ids:
            self.main_model[id].load_state_dict(model_state_dicts[id])
        writer = SummaryWriter(os.path.join("runs", name_tensorboard))
        total_reward = {}
        hidden_state = {}
        cell_state = {}
        lstm_out = {}
        alive = {}
        episode_count = 0
        total_steps = 0
        for episode in tqdm(range(total_episodes)):
            episode_count += 1
            step_count = 0
            prev_obs = self.env.reset()

            for id in self.agent_ids:
                total_reward[id] = 0
                alive[id] = True
                hidden_state[id], cell_state[id], lstm_out[id] = self.main_model[
                    id].module.lstm.init_hidden_states_and_outputs(
                    bsize=1)
            done = False
            while step_count < max_steps and not done:
                for id in self.agent_ids:
                    prev_obs[id][0] = torch.from_numpy(prev_obs[id][0]).float().to(self.device)
                step_count += 1
                total_steps += 1
                with torch.no_grad():

                    act, hidden_state, cell_state, lstm_out = self.find_best_action_by_model(prev_obs, hidden_state,
                                                                                                 cell_state, lstm_out)
                    obs_dict, reward_dict, done_dict, info_dict = self.env.step(act)

                    done = done_dict["__all__"]

                    for id in self.agent_ids:
                        total_reward[id] += reward_dict[id]

                        prev_obs[id][0] = prev_obs[id][0].reshape(prev_obs[id][0].shape[2], prev_obs[id][0].shape[3],
                                                                  prev_obs[id][0].shape[4])
                        prev_obs[id][0] = np.array(prev_obs[id][0].cpu())
                        act[id] = act[id].reshape(-1)
                        act[id] = np.array(act[id].cpu())

                    prev_obs = deepcopy(obs_dict)

            for id in self.agent_ids:
                writer.add_scalar(self.id_to_name[id] + ": Reward/train", total_reward[id], episode_count)
            writer.flush()

            print('\n Episode: [%d | %d] \n' % (episode, total_episodes))
            for id in self.agent_ids:
                print('\n Agent %d, Reward: %f\n' % (id, total_reward[id]))

        writer.close()
