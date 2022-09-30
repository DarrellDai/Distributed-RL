import keyboard
import numpy as np
from unity_wrappers.envs import MultiUnityWrapper
from mlagents_envs.environment import UnityEnvironment
from utils import save_checkpoint
from tqdm import tqdm
from copy import deepcopy
from Experience_Replay import Memory
from datetime import datetime
import yaml
import os
import torch

with open("config/Human_Play.yaml") as file:
    param = yaml.safe_load(file)

checkpoint = "Human_play_Nav_" + datetime.now().isoformat()[:10] + "_" + datetime.now().isoformat()[
                                                                     11:13] + "-" + datetime.now().isoformat()[
                                                                                    14:16] + ".pth.tar"
device = torch.device('cuda:' + str(param["device_idx"][0]) if torch.cuda.is_available() else 'cpu')
unity_env = UnityEnvironment(param["env_path"])
env = MultiUnityWrapper(unity_env=unity_env, uint8_visual=True, allow_multiple_obs=True)
agent_ids = tuple(env.agent_id_to_behaviour_name.keys())
max_steps = param["max_steps"]
mem = Memory(memsize=param["total_episodes"], agent_ids=agent_ids)
exit = False
for _ in tqdm(range(param["total_episodes"])):
    prev_obs = env.reset()
    step_count = 0
    local_memory = {}
    for id in agent_ids:
        local_memory[id]=[]
    done = False
    while step_count < max_steps and not done:
        act = {}
        # {0: agent 0's action, 1: ...]
        for id in agent_ids:
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
        if keyboard.is_pressed("esc"):
            exit = True
            break
        obs_dict, reward_dict, done_dict, info_dict = env.step(act)
        step_count += 1
        done = done_dict["__all__"]
        for id in agent_ids:
            local_memory[id].append(
                (deepcopy(prev_obs[id]), deepcopy(act[id]), deepcopy(reward_dict[id]), deepcopy(obs_dict[id])))
        prev_obs = obs_dict
    if exit:
        save_checkpoint({"memory": mem}, filename=checkpoint)
        env.close()
        break

    mem.add_episode(local_memory)
if not exit:
    save_checkpoint({"memory": mem}, filename=checkpoint)
    env.close()
filepath = os.path.join('Checkpoint', checkpoint)
checkpoint = torch.load(filepath, device)
