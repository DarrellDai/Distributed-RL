import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
os.chdir(current)
parent = os.path.dirname(current)
sys.path.append(parent)
import _pickle as cPickle
import argparse
import os
import time
from copy import deepcopy
from datetime import datetime

import keyboard
import numpy as np
import redis
import torch
import yaml
from mlagents_envs.environment import UnityEnvironment
from tqdm import tqdm

from Experience_Collector.Experience_Replay import Memory
from unity_wrappers.envs import MultiUnityWrapper
from utils import save_checkpoint, get_agents_id_to_name


class Human_play:
    def __init__(
            self,
            num_actor=1,
            device_idx=0,
            instance_idx=0
    ):
        self.num_actor = num_actor
        self.device_idx = device_idx
        self.device = torch.device('cuda:' + str(device_idx) if torch.cuda.is_available() else 'cpu')
        self.instance_idx = instance_idx

    def initialize_env(self, env_path, max_steps, total_episodes, worker_id):
        unity_env = UnityEnvironment(env_path, worker_id=worker_id)
        self.env = MultiUnityWrapper(unity_env=unity_env, uint8_visual=True, allow_multiple_obs=True)
        self.id_to_name = get_agents_id_to_name(self.env)
        self.agent_ids = tuple(self.id_to_name.keys())
        self.max_steps = max_steps+15
        self.total_episodes = total_episodes

    def controller(self):
        act = {}
        # {0: agent 0's action, 1: ...]
        for id in self.agent_ids:
            act[id] = np.zeros(2)
        if keyboard.is_pressed("down"):
            act[id][0] = -1
        elif keyboard.is_pressed("up"):
            act[id][0] = 1
        else:
            act[id][0] = 0
        if keyboard.is_pressed("right"):
            act[id][1] = 1
        elif keyboard.is_pressed("left"):
            act[id][1] = -1
        else:
            act[id][1] = 0

        # if keyboard.is_pressed("s"):
        #     act[id][0] = -1
        # elif keyboard.is_pressed("w"):
        #     act[id][0] = 1
        # else:
        #     act[id][0] = 0
        # if keyboard.is_pressed("d"):
        #     act[id][1] = 1
        # elif keyboard.is_pressed("a"):
        #     act[id][1] = -1
        # else:
        #     act[id][1] = 0
        return act

    def play_game(self, start_episode=0, mem=None, num_win=0, real_time_memory_sync=False, hostname='localhost'):
        if real_time_memory_sync:
            self.initialize_server(hostname)

        if mem is None:
            mem = Memory(memsize=self.total_episodes, agent_ids=self.agent_ids)
        total_reward = {}
        exit = False
        for episode in tqdm(range(start_episode, self.total_episodes)):
            prev_obs = self.env.reset()
            step_count = 0
            local_memory = {}
            for id in self.agent_ids:
                local_memory[id] = []
                total_reward[id] = 0
            done = False
            while step_count < self.max_steps and not done:
                act = self.controller()
                if keyboard.is_pressed("esc"):
                    exit = True
                    break
                obs_dict, reward_dict, done_dict, info_dict = self.env.step(act)
                step_count += 1
                done = done_dict["__all__"]
                for id in self.agent_ids:
                    if prev_obs[id][1][1] == 0:
                        local_memory[id].append(
                            (deepcopy(prev_obs[id]), deepcopy(act[id]), deepcopy(reward_dict[id]),
                             deepcopy(obs_dict[id])))
                        total_reward[id] += reward_dict[id]
                    if reward_dict[id] == -1:
                        done = True
                prev_obs = obs_dict
            if not done:
                num_win += 1
                print("You win")
            mem.add_episode(local_memory)
            if real_time_memory_sync:
                self.update()
                if len(mem) >= mem.memsize / self.num_actor:
                    self.connect.rpush("experience", cPickle.dumps(mem))
                    for id in self.agent_ids:
                        mem.replay_buffer[id].clear()
            for id in self.agent_ids:
                print('\n Agent %d, Reward: %f \n' % (id, total_reward[id]))
                print("Win rate: {}/{}".format(num_win, episode+1))
            if exit:
                break
        self.env.close()
        return mem, num_win

    def initialize_server(self, hostname='localhost'):
        self.connect = redis.Redis(host=hostname, db=self.instance_idx)

    def real_time_mode(self, hostname='localhost'):
        self.play_game(real_time_memory_sync=True, hostname=hostname)

    def save_mode(self, checkpoint_to_save, checkpoint_to_load=None):
        if not checkpoint_to_load is None:
            mem, id_to_name, num_win = self.load_checkpoint(checkpoint_to_load)
            new_mem = Memory(memsize=self.total_episodes, agent_ids=mem.agent_ids)
            for id in self.id_to_name:
                stored_memory = 0
                for episode in mem.replay_buffer[id]:
                    new_mem.replay_buffer[id].append(episode)
                    stored_memory += 1
                    if stored_memory == self.total_episodes:
                        break
            mem, num_win = self.play_game(start_episode=len(new_mem), mem=new_mem, num_win=num_win)
        else:
            mem, num_win = self.play_game()
        filename = checkpoint_to_save + "_" + datetime.now().isoformat()[:10] + "_" + datetime.now().isoformat()[
                                                                                      11:13] + "-" + datetime.now().isoformat()[
                                                                                                     14:16] + ".pth.tar"
        save_checkpoint({"memory": mem, "num_win": num_win, "id_to_name": self.id_to_name}, filename=filename)

    def load_mode(self, checkpoint_name, memsize, max_steps=0, hostname='localhost', mode="all"):
        self.initialize_server(hostname)
        memory, self.id_to_name, _ = self.load_checkpoint(checkpoint_name)
        filtered_memory = Memory(memsize=memsize, agent_ids=memory.agent_ids)
        for id in self.id_to_name:
            stored_memory = 0
            for episode in memory.replay_buffer[id]:
                if mode == "success_only":
                    if len(episode) == max_steps:
                        filtered_memory.replay_buffer[id].append(episode)
                        stored_memory += 1
                else:
                    filtered_memory.replay_buffer[id].append(episode)
                    stored_memory += 1
                if stored_memory == memsize:
                    break
        memory = filtered_memory
        self.connect.set("id_to_name", cPickle.dumps(self.id_to_name))
        self.connect.rpush("experience", cPickle.dumps(memory))
        with self.connect.lock("Update"):
            self.connect.set("episode_count", cPickle.dumps(len(memory)))
            self.connect.set("success_count", cPickle.dumps(0))

    def load_checkpoint(self, checkpoint_name):
        filepath = os.path.join('../Checkpoint', checkpoint_name + ".pth.tar")
        checkpoint = torch.load(filepath, self.device)
        memory = checkpoint["memory"]
        id_to_name = checkpoint["id_to_name"]
        num_win = checkpoint["num_win"]
        return memory, id_to_name, num_win

    def update(self):
        with self.connect.lock("Update"):
            self.wait_until_present("episode_count")
            episode_count = cPickle.loads(self.connect.get("episode_count"))
            episode_count = episode_count + 1
            self.connect.set("episode_count", cPickle.dumps(episode_count))

    def wait_until_present(self, name):
        # print("Waiting for "+name)
        while True:
            if not self.connect.get(name) is None:
                # print(name+" received")
                break
            time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learner process for distributed reinforcement.')
    parser.add_argument('-m', '--mode', type=str, default='s',
                        help="Runing mode. s:save_mode, l:load_mode, r: real time mode")
    parser.add_argument('-rc', '--run_config', type=str, default='Human_Play.yaml', help="Running config file name")
    parser.add_argument('-ins', '--instance_idx', type=int, default=0, help="The index of instance to run")
    parser.add_argument('-w', '--worker', type=int, default=0, help="Worker id to run the environment")
    args = parser.parse_args()
    with open("../Config/Run/" + args.run_config) as file:
        param = yaml.safe_load(file)
    human_play = Human_play(device_idx=param["device_idx"], instance_idx=args.instance_idx)
    if args.mode == 'l':
        human_play.load_mode(memsize=param["total_episodes"], max_steps=param["max_steps"],
                             checkpoint_name=param["checkpoint_to_load"], hostname=param["hostname"],
                             mode=param["mode"])
    else:
        human_play.initialize_env(env_path=param["env_path"], max_steps=param["max_steps"],
                                  total_episodes=param["total_episodes"], worker_id=args.worker)
        if args.mode == 's':
            checkpoint_save_name = param["checkpoint_save_name"]
            human_play.save_mode(checkpoint_to_save=param["checkpoint_save_name"],
                                 checkpoint_to_load=param["checkpoint_to_load"])
        else:
            human_play.real_time_mode(param["hostname"])
