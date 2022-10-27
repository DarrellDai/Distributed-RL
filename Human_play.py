import numpy as np
import torch

import os
from copy import deepcopy
import redis
from tqdm import tqdm
from datetime import datetime
import yaml
import keyboard
import argparse
import time

from unity_wrappers.envs import MultiUnityWrapper
from mlagents_envs.environment import UnityEnvironment

import _pickle as cPickle
from utils import save_checkpoint
from Experience_Replay import Memory

class Human_play:
    def __init__(
            self,
            id_to_name,
            device_idx=0,

    ):
        self.id_to_name = id_to_name
        self.agent_ids = tuple(id_to_name.keys())
        self.device_idx = device_idx
        self.device = torch.device('cuda:' + str(device_idx) if torch.cuda.is_available() else 'cpu')
    def initialize_env(self, env_path, max_steps, total_episodes):
        unity_env = UnityEnvironment(env_path)
        self.env = MultiUnityWrapper(unity_env=unity_env, uint8_visual=True, allow_multiple_obs=True)
        self.max_steps=max_steps
        self.total_episodes=total_episodes
    def controller(self):
        act = {}
        # {0: agent 0's action, 1: ...]
        for id in self.agent_ids:
            act[id] = np.zeros(2)
        if keyboard.is_pressed("down"):
            act[1][0] = -1
        elif keyboard.is_pressed("up"):
            act[1][0] = 1
        else:
            act[1][0] = 0
        if keyboard.is_pressed("right"):
            act[1][1] = 1
        elif keyboard.is_pressed("left"):
            act[1][1] = -1
        else:
            act[1][1] = 0

        # if keyboard.is_pressed("s"):
        #     act[0][0] = -1
        # elif keyboard.is_pressed("w"):
        #     act[0][0] = 1
        # else:
        #     act[0][0] = 0
        # if keyboard.is_pressed("d"):
        #     act[0][1] = 1
        # elif keyboard.is_pressed("a"):
        #     act[0][1] = -1
        # else:
        #     act[0][1] = 0
        return act
    def play_game(self, real_time_memory_sync=False, hostname='localhost'):
        if real_time_memory_sync:
            self.initialize_server(hostname)
        mem = Memory(memsize=self.total_episodes, agent_ids=self.agent_ids)
        exit = False
        for _ in tqdm(range(self.total_episodes)):
            prev_obs = self.env.reset()
            step_count = 0
            local_memory = {}
            for id in self.agent_ids:
                local_memory[id] = []
            done = False
            while step_count < self.max_steps and not done:
                act=self.controller()
                if keyboard.is_pressed("esc"):
                    exit = True
                    break
                obs_dict, reward_dict, done_dict, info_dict = self.env.step(act)
                step_count += 1
                done = done_dict["__all__"]
                for id in self.agent_ids:
                    local_memory[id].append(
                        (deepcopy(prev_obs[id]), deepcopy(act[id]), deepcopy(reward_dict[id]), deepcopy(obs_dict[id])))
                prev_obs = obs_dict
            mem.add_episode(local_memory)
            if real_time_memory_sync:
                self.update(local_memory)
            if exit:
                break
        self.env.close()
        return mem
    def initialize_server(self, hostname='localhost'):
        self.connect=redis.Redis(host=hostname)
        self.connect.delete("experience")
    def real_time_mode(self, hostname='localhost'):
        self.play_game(real_time_memory_sync=True, hostname=hostname)
    def save_mode(self,checkpoint_name):
        mem=self.play_game()
        filename = checkpoint_name + "_"+ datetime.now().isoformat()[:10] + "_" + datetime.now().isoformat()[
                                                                     11:13] + "-" + datetime.now().isoformat()[
                                                                                    14:16] + ".pth.tar"
        save_checkpoint({"memory": mem}, filename=filename)

    def load_mode(self, checkpoint_name, hostname='localhost'):
        self.initialize_server(hostname)
        filepath=os.path.join('Checkpoint', checkpoint_name)
        checkpoint = torch.load(filepath, self.device)
        memory=checkpoint["memory"]
        for i in range(len(memory)):
            local_memory = {}
            for id in self.agent_ids:
                local_memory[id] = memory.memory[id][i]
            self.update(local_memory)
    def update(self, local_memory):
        self.connect.rpush("experience", cPickle.dumps(local_memory))
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

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Learner process for distributed reinforcement.')
    parser.add_argument('-m', '--mode', type=str, default='l', help="Runing mode. s:save_mode, l:load_mode, r: real time mode")
    args=parser.parse_args()
    with open("config/Human_Play.yaml") as file:
        param = yaml.safe_load(file)
    env_path=param["env_path"]
    total_episodes=param["total_episodes"]
    max_steps = param["max_steps"]
    device_idx = param["device_idx"]
    id_to_name=param["id_to_name"]
    hostname = param["hostname"]
    human_play=Human_play(id_to_name=id_to_name, device_idx=device_idx)
    if args.mode=='l':
        checkpoint_to_load = param["checkpoint_to_load"]
        human_play.load_mode(checkpoint_name=checkpoint_to_load, hostname=hostname)
    else:
        human_play.initialize_env(env_path=env_path,max_steps=max_steps,total_episodes=total_episodes)
        if args.mode=='s':
            checkpoint_save_name=param["checkpoint_save_name"]
            human_play.save_mode(checkpoint_name=checkpoint_save_name)
        else:
            human_play.real_time_mode(hostname)
        