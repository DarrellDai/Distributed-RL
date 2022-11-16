import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import os
from copy import deepcopy
import redis
import threading
import yaml
import random
from itertools import count

from unity_wrappers.envs import MultiUnityWrapper
from mlagents_envs.environment import UnityEnvironment

import _pickle as cPickle
from utils import load_checkpoint, initialize_model, find_optimal_action, \
    find_hidden_cell_out_of_an_action, \
    combine_out, wait_until_present, get_agents_id_to_name
from Experience_Replay import Memory


class Actor:
    def __init__(
            self,
            num_actor=1,
            device_idx=[0],
            memsize=100,
            hostname="localhost",

    ):


        self.num_actor = num_actor
        self.device_idx = device_idx
        self.memory_size=memsize
        if device_idx==-1:
            self.device = torch.device('cpu')
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available, please choose CPU by setting device_idx=-1")
            self.device = torch.device('cuda:' + str(device_idx))
        self._connect = redis.Redis(host=hostname)
        torch.set_num_threads(10)

    def initialize_env(self, env_path, actor_idx):
        unity_env = UnityEnvironment(env_path+"_"+str(actor_idx)+"/Hide and Seek", worker_id=actor_idx)
        env = MultiUnityWrapper(unity_env=unity_env, uint8_visual=True, allow_multiple_obs=True)
        if actor_idx==0:
            self.id_to_name = get_agents_id_to_name(env)
            self.agent_ids = tuple(self.id_to_name.keys())
            self._connect.set("id_to_name", cPickle.dumps(self.id_to_name))
        return env
    def initialize_model(self, cnn_out_size, lstm_hidden_size, action_shape, action_out_size, atten_size, mode, checkpoint_to_load=None):
        self.action_shape = {}
        self.atten_size = {}
        for idx in range(len(self.agent_ids)):
            self.action_shape[self.agent_ids[idx]] = tuple(action_shape[idx])
            self.atten_size[self.agent_ids[idx]] = atten_size[idx]

        self.model = initialize_model(self.agent_ids, cnn_out_size, lstm_hidden_size, action_shape, action_out_size,
                                      atten_size, self.device)
        if mode=="train":
            self._pull_params()
        else:
            model_state_dicts, optimizer_state_dicts, episode_count, self.epsilon, self.initial_epoch_count, success_count = load_checkpoint(
                checkpoint_to_load, self.device)
            for id in self.agent_ids:
                self.model[id].load_state_dict(model_state_dicts[id])

    def find_random_action_while_updating_LSTM(self, env, prev_obs, hidden_state, cell_state, lstm_out):
        act = {}
        for id in self.agent_ids:
            act[id] = env.action_space[id].sample() - 1
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

    def collect_data(self, env, max_steps, name_tensorboard, actor_update_freq, actor_idx, mode):
        memory = Memory(self.memory_size, self.agent_ids)
        writer = SummaryWriter(os.path.join("runs", name_tensorboard))
        total_reward = {}
        local_memory = {}
        hidden_state = {}
        cell_state = {}
        lstm_out = {}
        alive = {}
        if mode=="train":
            wait_until_present(self._connect, "epsilon")
            epsilon = cPickle.loads(self._connect.get("epsilon"))
            # print("Actor {} got epsilon".format(self.actor_idx))
            prev_epoch = -1
        else:
            epsilon =0



        for _ in count():

            step_count = 0
            prev_obs = env.reset()

            for id in self.agent_ids:
                total_reward[id] = 0
                local_memory[id] = []
                alive[id] = True
                hidden_state[id], cell_state[id], lstm_out[id] = self.model[
                    id].lstm.init_hidden_states_and_outputs(
                    bsize=1)
            done = False
            while step_count < max_steps and not done:
                for id in self.agent_ids:
                    prev_obs[id][0] = torch.from_numpy(prev_obs[id][0]).float().to(self.device)
                    hidden_state[id]=hidden_state[id].to(self.device)
                    cell_state[id]=cell_state[id].to(self.device)
                    lstm_out[id]=lstm_out[id].to(self.device)
                step_count += 1
                with torch.no_grad():
                    #todo:check this lock
                    with threading.Lock():
                        if np.random.rand(1) < epsilon:

                            act, hidden_state, cell_state, lstm_out = self.find_random_action_while_updating_LSTM(env,
                                                                                                                  prev_obs,
                                                                                                                  hidden_state,
                                                                                                                  cell_state,
                                                                                                                  lstm_out)
                        else:
                            act, hidden_state, cell_state, lstm_out = self.find_best_action_by_model(prev_obs, hidden_state,
                                                                                                     cell_state, lstm_out)
                    obs_dict, reward_dict, done_dict, info_dict = env.step(act)

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
            memory.add_episode(local_memory)
            # print("Actor {} got {}/{} episodes".format(self.actor_idx, len(self.memory), int(np.ceil(self.memory.memsize / self.num_actor))))
            if mode=="train":
                if len(memory) >= memory.memsize / self.num_actor:
                    # print("Sending memory")
                    with self._connect.lock("Update Experience"):
                        self._connect.rpush("experience", cPickle.dumps(memory))
                    for id in self.agent_ids:
                        memory.replay_buffer[id].clear()

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
                        writer.add_scalar(self.id_to_name[id] + ": Reward vs Episode", total_reward[id], episode_count)
                    writer.flush()

                if epoch % actor_update_freq == 0 and prev_epoch != epoch and actor_idx==0:
                    self._pull_params()

                prev_epoch = epoch

    def _pull_params(self):
        wait_until_present(self._connect, "params")
        # print("Sync params.")
        with self._connect.lock("Update params"):
            with threading.Lock():
                params = self._connect.get("params")
                for id in self.agent_ids:
                    self.model[id].load_state_dict(cPickle.loads(params)[id])
                    self.model[id].to(self.device)

    def run(self, seed, env, env_path, actor_idx, max_steps, name_tensorboard, actor_update_freq, mode):
        random.seed(seed)
        if actor_idx!=0:
            env=self.initialize_env(env_path, actor_idx)
        self.collect_data(env=env, max_steps=max_steps,
                          name_tensorboard=name_tensorboard,
                          actor_update_freq=actor_update_freq, actor_idx=actor_idx, mode=mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Actor process for distributed reinforcement.')
    parser.add_argument('-n', '--num_actors', type=int, default=1, help='Actor number.')
    parser.add_argument('-r', '--redisserver', type=str, default='localhost', help="Redis's server name.")
    parser.add_argument('-d', '--device', type=int, default=0, help="Index of GPU to use, -1 is CPU")
    parser.add_argument('-m', '--mode', type=str, default="train", help="Train or test mode")
    parser.add_argument('-mc', '--model_config', type=str, default='Model.yaml', help="Model config file name")
    parser.add_argument('-rc', '--run_config', type=str, default='Train.yaml', help="Running config file name")
    args = parser.parse_args()

    with open("Config/"+args.model_config) as file:
        model_param = yaml.safe_load(file)
    with open("Config/"+args.run_config) as file:
        run_param = yaml.safe_load(file)
    actor = Actor(
        num_actor=args.num_actors,
        device_idx=args.device, memsize=run_param["memory_size"], hostname=args.redisserver)
    env=actor.initialize_env(run_param["env_path"], 0)
    actor.initialize_model(cnn_out_size=model_param["cnn_out_size"], lstm_hidden_size=model_param["lstm_hidden_size"],
                           action_shape=model_param["action_shape"],
                           action_out_size=model_param["action_out_size"], atten_size=model_param["atten_size"], mode=args.mode, checkpoint_to_load=run_param["checkpoint_to_load"])
    threads=[]
    for i in range(args.num_actors):
        if i==0:
            thread=threading.Thread(target=actor.run, args=(i, env, None, i, run_param["max_steps"], run_param["name_tensorboard"], run_param["target_update_freq(epochs)"], args.mode))
        else:
            thread = threading.Thread(target=actor.run, args=(
            i, None, run_param["env_path"], i, run_param["max_steps"], run_param["name_tensorboard"],
            run_param["target_update_freq(epochs)"], args.mode))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()