import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import os
from copy import deepcopy
import redis
import time
import yaml
import random
from itertools import count

from unity_wrappers.envs import MultiUnityWrapper
from mlagents_envs.environment import UnityEnvironment

import _pickle as cPickle
from utils import initialize_model, find_optimal_action, \
    wrap_model_with_dataparallel, \
    find_hidden_cell_out_of_an_action, \
    combine_out, wait_until_present, get_agents_id_to_name
from Experience_Replay import Memory


class Actor:
    def __init__(
            self,
            actor_idx=0,
            num_actor=1,
            device_idx=0,
            memsize=100,
            hostname="localhost",
            seed=0

    ):


        self.actor_idx = actor_idx
        self.num_actor = num_actor
        self.device_idx = device_idx
        self.memory_size=memsize
        self.device = torch.device('cuda:' + str(device_idx) if torch.cuda.is_available() else 'cpu')
        self._connect = redis.Redis(host=hostname)
        random.seed(seed)
        torch.set_num_threads(10)

    def initialize_env(self, env_path):
        unity_env = UnityEnvironment(env_path, worker_id=self.actor_idx)
        self.env = MultiUnityWrapper(unity_env=unity_env, uint8_visual=True, allow_multiple_obs=True)
        self.id_to_name = get_agents_id_to_name(self.env)
        self.agent_ids = tuple(self.id_to_name.keys())
        if self.actor_idx==0:
            self._connect.set("id_to_name", cPickle.dumps(self.id_to_name))


    def initialize_model(self, cnn_out_size, lstm_hidden_size, action_shape, action_out_size, atten_size):
        self.memory = Memory(self.memory_size, self.agent_ids)
        self.action_shape = {}
        self.atten_size = {}
        for idx in range(len(self.agent_ids)):
            self.action_shape[self.agent_ids[idx]] = tuple(action_shape[idx])
            self.atten_size[self.agent_ids[idx]] = atten_size[idx]

        self.model = initialize_model(self.agent_ids, cnn_out_size, lstm_hidden_size, action_shape, action_out_size,
                                      atten_size)
        wrap_model_with_dataparallel(self.model, [self.device_idx])
        self._pull_params()

    def find_random_action_while_updating_LSTM(self, prev_obs, hidden_state, cell_state, lstm_out):
        act = {}
        for id in self.agent_ids:
            act[id] = self.env.action_space[id].sample() - 1
            act[id] = torch.tensor(act[id]).reshape(1, 1, len(act[id])).to(self.device)
            prev_obs[id][0] = prev_obs[id][0].reshape(1, 1, prev_obs[id][0].shape[0], prev_obs[id][0].shape[1],
                                                      prev_obs[id][0].shape[2])
            model_out = self.model[id](prev_obs[id][0], act[id],
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
                hidden_state_per_action, cell_state_per_action) = self.model[id](prev_obs[id][0],
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

    def collect_data(self, max_steps, name_tensorboard, actor_update_freq):
        writer = SummaryWriter(os.path.join("runs", name_tensorboard))
        total_reward = {}
        local_memory = {}
        hidden_state = {}
        cell_state = {}
        lstm_out = {}
        alive = {}
        wait_until_present(self._connect, "epsilon")
        epsilon = cPickle.loads(self._connect.get("epsilon"))
        # print("Actor {} got epsilon".format(self.actor_idx))
        prev_epoch = -1

        for _ in count():

            step_count = 0
            prev_obs = self.env.reset()

            for id in self.agent_ids:
                total_reward[id] = 0
                local_memory[id] = []
                alive[id] = True
                hidden_state[id], cell_state[id], lstm_out[id] = self.model[
                    id].module.lstm.init_hidden_states_and_outputs(
                    bsize=1)
            done = False
            while step_count < max_steps and not done:
                for id in self.agent_ids:
                    prev_obs[id][0] = torch.from_numpy(prev_obs[id][0]).float().to(self.device)
                step_count += 1
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
                        act[id] = np.array(act[id].cpu())

                        local_memory[id].append(
                            (deepcopy(prev_obs[id]), deepcopy(act[id]), deepcopy(reward_dict[id]),
                             deepcopy(obs_dict[id])))

                    prev_obs = deepcopy(obs_dict)

            # save performance measure
            # print("Sending memory")
            self.memory.add_episode(local_memory)
            # print("Actor {} got {}/{} episodes".format(self.actor_idx, len(self.memory), int(np.ceil(self.memory.memsize / self.num_actor))))
            if len(self.memory) >= self.memory.memsize / self.num_actor:
                # print("Sending memory")
                with self._connect.lock("Update Experience"):
                    self._connect.rpush("experience", cPickle.dumps(self.memory))
                for id in self.agent_ids:
                    self.memory.replay_buffer[id].clear()

            with self._connect.lock("Update"):
                wait_until_present(self._connect, "episode_count")
                episode_count = cPickle.loads(self._connect.get("episode_count"))
                # print("Actor {} got episode_count".format(self.actor_idx))
                episode_count += + 1
                self._connect.set("episode_count", cPickle.dumps(episode_count))
                success_count = cPickle.loads(self._connect.get("success_count"))
                # print("Actor {} got success_count".format(self.actor_idx))
                if not done:
                    success_count += 1
                self._connect.set("success_count", cPickle.dumps(success_count))
                self._connect.set("epsilon", cPickle.dumps(epsilon))
                wait_until_present(self._connect, "epoch")
                epoch = cPickle.loads(self._connect.get("epoch"))
                # print("Actor {} got epoch".format(self.actor_idx))
                for id in self.agent_ids:
                    writer.add_scalar(self.id_to_name[id] + ": Reward/train", total_reward[id], episode_count)
                writer.flush()




            if epoch % actor_update_freq == 0 and prev_epoch != epoch:
                self._pull_params()

            prev_epoch = epoch

    def _pull_params(self):
        wait_until_present(self._connect, "params")
        # print("Sync params.")
        with self._connect.lock("Update params"):
            params = self._connect.get("params")
            for id in self.agent_ids:
                self.model[id].load_state_dict(cPickle.loads(params)[id])
                self.model[id].to(self.device)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Actor process for distributed reinforcement.')
    parser.add_argument('-n', '--num_actors', type=int, default=1, help='Actor number.')
    parser.add_argument('-i', '--actor_index', type=int, default=0, help="Index of actor")
    parser.add_argument('-r', '--redisserver', type=str, default='localhost', help="Redis's server name.")
    parser.add_argument('-d', '--device', type=int, default=0, help="Index of GPU to use")
    parser.add_argument('-s', '--seed', type=int, default=0, help="Seed for randomization")
    parser.add_argument('-mc', '--model_config', type=str, default='Model.yaml', help="Model config file name")
    parser.add_argument('-rc', '--run_config', type=str, default='Train.yaml', help="Running config file name")
    args = parser.parse_args()

    with open("Config/"+args.model_config) as file:
        model_param = yaml.safe_load(file)
    with open("Config/"+args.run_config) as file:
        run_param = yaml.safe_load(file)
    actor = Actor(
        actor_idx=args.actor_index,
        num_actor=args.num_actors,
        device_idx=args.device, memsize=run_param["memory_size"], hostname=args.redisserver, seed=args.seed)
    actor.initialize_env(run_param["env_path"])
    actor.initialize_model(cnn_out_size=model_param["cnn_out_size"], lstm_hidden_size=model_param["lstm_hidden_size"],
                           action_shape=model_param["action_shape"],
                           action_out_size=model_param["action_out_size"], atten_size=model_param["atten_size"])
    actor.collect_data(max_steps=run_param["max_steps"],
                       name_tensorboard=run_param["name_tensorboard"],
                       actor_update_freq=run_param["actor_update_freq(epochs)"])
    actor.env.close()