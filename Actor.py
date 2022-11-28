import _pickle as cPickle
import argparse
import os
import random
import threading
from copy import deepcopy
from itertools import count

import numpy as np
import redis
import torch
import yaml
from mlagents_envs.environment import UnityEnvironment
from torch.utils.tensorboard import SummaryWriter

from Experience_Replay import Memory
from unity_wrappers.envs import MultiUnityWrapper
from utils import load_checkpoint, initialize_model, find_optimal_action, \
    wait_until_present, get_agents_id_to_name


class Actor:
    def __init__(
            self,
            num_actor=1,
            device_idx=[0],
            memsize=100,
            hostname="localhost",
            instance_idx=0
    ):

        self.num_actor = num_actor
        self.device_idx = device_idx
        self.memory_size = memsize
        if device_idx == -1:
            self.device = torch.device('cpu')
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available, please choose CPU by setting device_idx=-1")
            self.device = torch.device('cuda:' + str(device_idx))
        self.instance_idx = instance_idx
        self._connect = redis.Redis(host=hostname, db=instance_idx)
        idx = 0
        while True:
            if not self._connect.get("actor{}".format(idx)) is None:
                self._connect.delete("actor{}".format(idx))
                idx += 1
            else:
                break
        self.locks = []

        for i in range(num_actor):
            self.locks.append(self._connect.lock("actor{}".format(i)))
        torch.set_num_threads(10)

    def initialize_env(self, env_path, actor_idx):
        unity_env = UnityEnvironment(env_path + "_" + str(actor_idx) + "/Hide and Seek",
                                     worker_id=actor_idx + self.num_actor * self.instance_idx)
        env = MultiUnityWrapper(unity_env=unity_env, uint8_visual=True, allow_multiple_obs=True)
        if actor_idx == 0:
            self.id_to_name = get_agents_id_to_name(env)
            self.agent_ids = tuple(self.id_to_name.keys())
            self._connect.set("id_to_name", cPickle.dumps(self.id_to_name))
        return env

    def initialize_model(self, cnn_out_size, lstm_hidden_size, action_shape, atten_size, mode,
                         method, checkpoint_to_load=None):
        self.action_shape = {}
        self.atten_size = {}
        for idx in range(len(self.agent_ids)):
            self.action_shape[self.agent_ids[idx]] = tuple(action_shape[idx])
            self.atten_size[self.agent_ids[idx]] = atten_size[idx]

        self.models = initialize_model(self.agent_ids, cnn_out_size, lstm_hidden_size, action_shape,
                                       atten_size, self.device, method)
        if mode == "train":
            self._pull_params()
        else:
            model_state_dicts, optimizer_state_dicts, episode_count, self.epsilon, self.initial_epoch_count, success_count = load_checkpoint(
                checkpoint_to_load + ".pth.tar", self.device)
            for id in self.agent_ids:
                self.models[id].load_model_state_dict(model_state_dicts[id])

    def find_random_action(self, env):
        act = {}
        for id in self.agent_ids:
            act[id] = env.action_space[id].sample() - 1
            act[id] = torch.tensor(act[id]).reshape(1, 1, len(act[id])).to(self.device)
        return act

    def find_best_action_by_model(self, prev_obs, hidden_state, cell_state):
        act = {}
        for id in self.agent_ids:
            prev_obs[id][0] = prev_obs[id][0].reshape(1, 1, prev_obs[id][0].shape[0], prev_obs[id][0].shape[1],
                                                      prev_obs[id][0].shape[2])
            _, (hidden_state[id], cell_state[id]), (act_prob, value) = self.models[id].evaluate(prev_obs[id][0],
                                                                                                hidden_state=
                                                                                                hidden_state[id],
                                                                                                cell_state=
                                                                                                cell_state[id])

            act[id] = find_optimal_action(act_prob)
            act[id] = torch.from_numpy(act[id]).to(self.device)
            # todo:might be problems here
            # hidden_state[id], cell_state[id], lo_new = find_hidden_cell_out_of_an_action(act[id],
            #                                                                              hidden_state_per_action,
            #                                                                              cell_state_per_action,
            #                                                                              out_per_action)
            # lstm_out[id] = combine_out(lo, lo_new.to(self.device), self.atten_size[id])
        return act, hidden_state, cell_state

    def collect_data(self, env, max_steps, actor_idx, mode):
        if actor_idx == 0:
            self._connect.set("to_update", cPickle.dumps(False))
        memory = Memory(self.memory_size, self.agent_ids)
        wait_until_present(self._connect, "name_tensorboard")
        name_tensorboard = cPickle.loads(self._connect.get("name_tensorboard"))
        writer = SummaryWriter(os.path.join("runs", name_tensorboard))
        total_reward = {}
        local_memory = {}
        hidden_state = {}
        cell_state = {}
        alive = {}
        if mode == "train":
            wait_until_present(self._connect, "epsilon")
            epsilon = cPickle.loads(self._connect.get("epsilon"))
            # print("Actor {} got epsilon".format(self.actor_idx))
        else:
            epsilon = 0

        for _ in count():

            step_count = 0
            prev_obs = env.reset()

            for id in self.agent_ids:
                total_reward[id] = 0
                local_memory[id] = []
                alive[id] = True
                hidden_state[id], cell_state[id] = self.models[
                    id].get_initial_value(bsize=1)
                hidden_state[id], cell_state[id] = hidden_state[id].to(self.device), cell_state[id].to(self.device)
            done = False
            while step_count < max_steps and not done:
                for id in self.agent_ids:
                    prev_obs[id][0] = torch.from_numpy(prev_obs[id][0]).float().to(self.device)
                step_count += 1
                with torch.no_grad():
                    with self._connect.lock("actor{}".format(actor_idx)):
                        act, hidden_state, cell_state = self.find_best_action_by_model(prev_obs,
                                                                                       hidden_state,
                                                                                       cell_state)
                    if np.random.rand(1) < epsilon:
                        act = self.find_random_action(env)
                    obs_dict, reward_dict, done_dict, info_dict = env.step(act)

                    done = done_dict["__all__"]

                    for id in self.agent_ids:
                        if reward_dict[id] == -1:
                            done = True
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
            if mode == "train":
                if len(memory) >= memory.memsize / self.num_actor:
                    # print("Sending memory")
                    with self._connect.lock("Update Experience"):
                        self._connect.rpush("experience", cPickle.dumps(memory))
                    memory.clear_memory()

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
                    to_update = cPickle.loads(self._connect.get("to_update"))
                    for id in self.agent_ids:
                        writer.add_scalar(self.id_to_name[id] + ": Reward vs Episode", total_reward[id], episode_count)
                    writer.flush()

                if to_update and actor_idx == 0:
                    with self._connect.lock("Update params"):
                        self._pull_params()
                        self._connect.set("to_update", cPickle.dumps(False))

    def _pull_params(self):
        wait_until_present(self._connect, "params")
        # print("Sync params.")
        for lock in self.locks:
            lock.acquire()
        params = self._connect.get("params")
        for id in self.agent_ids:
            self.models[id].load_model_state_dict(cPickle.loads(params)[id])
        for lock in self.locks:
            lock.release()

    def run(self, seed, env, env_path, actor_idx, max_steps, mode):
        random.seed(seed)
        if actor_idx != 0:
            env = self.initialize_env(env_path, actor_idx)
        self.collect_data(env=env, max_steps=max_steps,
                          actor_idx=actor_idx, mode=mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Actor process for distributed reinforcement.')
    parser.add_argument('-n', '--num_actors', type=int, default=1, help='Actor number.')
    parser.add_argument('-r', '--redisserver', type=str, default='localhost', help="Redis's server name.")
    parser.add_argument('-ins', '--instance_idx', type=int, default=0, help="The index of instance to run")
    parser.add_argument('-d', '--device', type=int, default=0, help="Index of GPU to use, -1 is CPU")
    parser.add_argument('-m', '--mode', type=str, default="train", help="Train or test mode")
    parser.add_argument('-mc', '--model_config', type=str, default='Model.yaml', help="Model config file name")
    parser.add_argument('-rc', '--run_config', type=str, default='Train.yaml', help="Running config file name")
    args = parser.parse_args()

    with open("Config/" + args.model_config) as file:
        model_param = yaml.safe_load(file)
    with open("Config/" + args.run_config) as file:
        run_param = yaml.safe_load(file)
    actor = Actor(
        num_actor=args.num_actors,
        device_idx=args.device, memsize=run_param["memory_size"], hostname=args.redisserver,
        instance_idx=args.instance_idx)
    env = actor.initialize_env(run_param["env_path"], 0)
    from PPO import PPO

    actor.initialize_model(cnn_out_size=model_param["cnn_out_size"], lstm_hidden_size=model_param["lstm_hidden_size"],
                           action_shape=model_param["action_shape"],
                           atten_size=model_param["atten_size"],
                           mode=args.mode, checkpoint_to_load=run_param["checkpoint_to_load"],
                           method=PPO)
    threads = []
    for i in range(args.num_actors):
        if i == 0:
            thread = threading.Thread(target=actor.run, args=(
                i, env, None, i, run_param["max_steps"],
                args.mode))
        else:
            thread = threading.Thread(target=actor.run, args=(
                i, None, run_param["env_path"], i, run_param["max_steps"],
                args.mode))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
