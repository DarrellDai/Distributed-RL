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
from utils import save_checkpoint, get_agents_id_to_name
from Experience_Replay import Memory

class Human_play:
    def __init__(
            self,
            num_actor=1,
            device_idx=0
    ):
        self.num_actor=num_actor
        self.device_idx = device_idx
        self.device = torch.device('cuda:' + str(device_idx) if torch.cuda.is_available() else 'cpu')
    def initialize_env(self, env_path, max_steps, total_episodes, worker_id):
        unity_env = UnityEnvironment(env_path, worker_id=worker_id)
        self.env = MultiUnityWrapper(unity_env=unity_env, uint8_visual=True, allow_multiple_obs=True)
        self.id_to_name = get_agents_id_to_name(self.env)
        self.agent_ids = tuple(self.id_to_name.keys())
        self.max_steps=max_steps
        self.total_episodes=total_episodes
        self._connect.set("id_to_name", cPickle.dumps(self.id_to_name))
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
    def play_game(self, start_episode=0, mem=None, real_time_memory_sync=False, hostname='localhost'):
        if real_time_memory_sync:
            self.initialize_server(hostname)
        if mem is None:
            mem = Memory(memsize=self.total_episodes, agent_ids=self.agent_ids)
        total_reward = {}
        exit = False
        for _ in tqdm(range(start_episode, self.total_episodes)):
            prev_obs = self.env.reset()
            step_count = 0
            local_memory = {}
            for id in self.agent_ids:
                local_memory[id] = []
                total_reward[id] = 0
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
                    total_reward[id] += reward_dict[id]
                prev_obs = obs_dict
            mem.add_episode(local_memory)
            if real_time_memory_sync:
                self.update()
                if len(mem) >= mem.memsize / self.num_actor:
                    self.connect.rpush("experience", cPickle.dumps(mem))
                    for id in self.agent_ids:
                        mem.replay_buffer[id].clear()
            if exit:
                break
            for id in self.agent_ids:
                print('\n Agent %d, Reward: %f \n' % (id, total_reward[id]))
        self.env.close()
        return mem
    def initialize_server(self, hostname='localhost'):
        self.connect=redis.Redis(host=hostname)
        self.connect.delete("experience")
    def real_time_mode(self, hostname='localhost'):
        self.play_game(real_time_memory_sync=True, hostname=hostname)
    def save_mode(self, checkpoint_to_save, checkpoint_to_load=None):
        if not checkpoint_to_load is None:
            mem=self.load_checkpoint(checkpoint_to_load)
            print(len(mem))
            mem = self.play_game(start_episode=len(mem), mem=mem)
        else:
            mem=self.play_game()
        filename = checkpoint_to_save + "_" + datetime.now().isoformat()[:10] + "_" + datetime.now().isoformat()[
                                                                     11:13] + "-" + datetime.now().isoformat()[
                                                                                    14:16] + ".pth.tar"
        save_checkpoint({"memory": mem}, filename=filename)

    def load_mode(self, checkpoint_name, hostname='localhost'):
        self.initialize_server(hostname)
        memory,self.id_to_name=self.load_checkpoint(checkpoint_name)
        self.connect.set("id_to_name", cPickle.dumps(self.id_to_name))
        self.connect.rpush("experience", cPickle.dumps(memory))
        with self.connect.lock("Update"):
            self.connect.set("episode_count", cPickle.dumps(len(memory)))
            self.connect.set("success_count", cPickle.dumps(0))
    def load_checkpoint(self, checkpoint_name):
        filepath = os.path.join('Checkpoint', checkpoint_name)
        checkpoint = torch.load(filepath, self.device)
        memory = checkpoint["memory"]
        id_to_name = checkpoint["id_to_name"]
        return memory, id_to_name


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

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Learner process for distributed reinforcement.')
    parser.add_argument('-m', '--mode', type=str, default='s', help="Runing mode. s:save_mode, l:load_mode, r: real time mode")
    parser.add_argument('-rc', '--run_config', type=str, default='Human_play.yaml', help="Running config file name")
    parser.add_argument('-w', '--worker', type=int, default=0, help="Worker id to run the environment")
    args=parser.parse_args()
    with open("Config/"+args.run_config) as file:
        param = yaml.safe_load(file)
    human_play=Human_play(device_idx=param["device_idx"])
    if args.mode=='l':
        human_play.load_mode(checkpoint_name=param["checkpoint_to_load"], hostname=param["hostname"])
    else:
        human_play.initialize_env(env_path=param["env_path"],max_steps=param["max_steps"],total_episodes=param["total_episodes"], worker_id=args.worker)
        if args.mode=='s':
            checkpoint_save_name=param["checkpoint_save_name"]
            human_play.save_mode(checkpoint_to_save=param["checkpoint_save_name"], checkpoint_to_load=param["checkpoint_to_load"])
        else:
            human_play.real_time_mode(param["hostname"])
        